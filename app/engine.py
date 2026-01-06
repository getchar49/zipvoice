import os, json, uuid, asyncio, datetime as dt, torch, safetensors.torch, time
import gc  # <--- NEW: Needed for garbage collection
from dataclasses import dataclass, field
from typing import Optional, Dict
from huggingface_hub import hf_hub_download
from torch.amp import autocast
from zipvoice.utils.checkpoint import load_checkpoint
from zipvoice.bin.infer_zipvoice import (
    generate_sentence, get_vocoder, VocosFbank,
    ZipVoice, ZipVoiceDistill, EmiliaTokenizer, EspeakTokenizer,
    load_trt, HUGGINGFACE_REPO, MODEL_DIR
)
from app.normalizer.processing import normalize_vietnamese_text

from .settings import (
    RESULTS_DIR, MODEL_NAME, ZIPVOICE_MODEL_DIR, VOCOS_LOCAL_DIR,
    DEVICE, TOKENIZER, LANG_TOKENIZER, MAX_DURATION
)
from .registry import VoiceRegistry, Voice

class JobCancelledError(Exception):
    pass

@dataclass
class TTSJob:
    id: str
    text: str
    voice_id: str
    out_wav_path: str
    speed: float = 1.0
    num_step: Optional[int] = None
    guidance_scale: Optional[float] = None
    remove_long_sil: bool = False
    audio_type: str = "mp3"
    status: str = "queued" 
    progress: float = 0.0
    error: Optional[str] = None
    created_at: dt.datetime = field(default_factory=dt.datetime.utcnow)
    finished_at: Optional[dt.datetime] = None

class ZipVoiceEngine:
    def __init__(self, registry: VoiceRegistry):
        self.device = torch.device(DEVICE if torch.cuda.is_available() else "cpu")
        os.makedirs(RESULTS_DIR, exist_ok=True)
        self.registry = registry
        self.max_concurrent = int(os.getenv("MAX_CONCURRENT", "2"))
        self.sema = asyncio.Semaphore(self.max_concurrent)

        # -- Load model config & weights --
        if ZIPVOICE_MODEL_DIR:
            model_ckpt  = os.path.join(ZIPVOICE_MODEL_DIR, "model.pt")
            model_cfg   = os.path.join(ZIPVOICE_MODEL_DIR, "model.json")
            token_file  = os.path.join(ZIPVOICE_MODEL_DIR, "tokens.txt")
        else:
            model_ckpt  = hf_hub_download(HUGGINGFACE_REPO, filename=f"{MODEL_DIR[MODEL_NAME]}/model.pt")
            model_cfg   = hf_hub_download(HUGGINGFACE_REPO, filename=f"{MODEL_DIR[MODEL_NAME]}/model.json")
            token_file  = hf_hub_download(HUGGINGFACE_REPO, filename=f"{MODEL_DIR[MODEL_NAME]}/tokens.txt")

        with open(model_cfg, "r", encoding="utf-8") as f:
            cfg = json.load(f)

        # Tokenizer
        if TOKENIZER == "espeak":
            self.tokenizer = EspeakTokenizer(token_file=token_file, lang=LANG_TOKENIZER)
        else:
            self.tokenizer = EmiliaTokenizer(token_file=token_file)

        tokenizer_config = {"vocab_size": self.tokenizer.vocab_size, "pad_id": self.tokenizer.pad_id}
        if MODEL_NAME == "zipvoice":
            self.model = ZipVoice(**cfg["model"], **tokenizer_config)
            self.defaults = {"num_step": 16, "guidance_scale": 1.0}
        else:
            self.model = ZipVoiceDistill(**cfg["model"], vocab_size=None, pad_id=None)
            self.defaults = {"num_step": 8, "guidance_scale": 3.0}

        if model_ckpt.endswith(".safetensors"):
            safetensors.torch.load_model(self.model, model_ckpt)
        else:
            load_checkpoint(filename=model_ckpt, model=self.model, strict=True)

        self.model = self.model.to(self.device, dtype=torch.float16).eval()

        self.vocoder = get_vocoder(VOCOS_LOCAL_DIR).to(self.device).eval()
        self.feature_extractor = VocosFbank()
        self.sampling_rate = cfg["feature"]["sampling_rate"]

        self.queue: asyncio.Queue[str] = asyncio.Queue()
        self.jobs : Dict[str, TTSJob] = {}

    def get_job(self, job_id: str) -> Optional[TTSJob]:
        if job_id in self.jobs:
            return self.jobs[job_id]
        
        # Check for wav
        wav_path = os.path.join(RESULTS_DIR, f"{job_id}.wav")
        if os.path.exists(wav_path):
            return TTSJob(
                id=job_id,
                text="<restored_from_disk>",
                voice_id="<unknown>",
                out_wav_path=wav_path,
                audio_type="wav",
                status="done",
                progress=1.0,
                finished_at=dt.datetime.fromtimestamp(os.path.getmtime(wav_path)),
                error=None
            )

        # Check for mp3
        mp3_path = os.path.join(RESULTS_DIR, f"{job_id}.mp3")
        if os.path.exists(mp3_path):
            return TTSJob(
                id=job_id,
                text="<restored_from_disk>",
                voice_id="<unknown>",
                out_wav_path=mp3_path,
                audio_type="mp3",
                status="done",
                progress=1.0,
                finished_at=dt.datetime.fromtimestamp(os.path.getmtime(mp3_path)),
                error=None
            )
            
        return None

    async def submit(self, text: str, voice_id: str,
                     speed=1.0, num_step=None, guidance_scale=None, remove_long_sil=False, audio_type="mp3") -> str:
        job_id = str(uuid.uuid4())
        out_path = os.path.join(RESULTS_DIR, f"{job_id}.wav")
        job = TTSJob(job_id, text, voice_id, out_path, speed, num_step, guidance_scale, remove_long_sil, audio_type=audio_type)
        self.jobs[job_id] = job
        await self.queue.put(job_id)
        return job_id

    def cancel_job(self, job_id: str) -> bool:
        if job_id not in self.jobs: return False
        job = self.jobs[job_id]
        if job.status in ["done", "error", "cancelled"]: return True
        job.status = "cancelled"
        job.finished_at = dt.datetime.utcnow()
        return True

    def cancel_all_jobs(self) -> int:
        count = 0
        for jid, job in self.jobs.items():
            if job.status in ["queued", "running"]:
                job.status = "cancelled"
                job.finished_at = dt.datetime.utcnow()
                count += 1
        return count

    async def run(self):
        asyncio.create_task(self._memory_cleanup_loop())
        while True:
            job_id = await self.queue.get()
            asyncio.create_task(self._admit(job_id))

    async def _memory_cleanup_loop(self):
        while True:
            await asyncio.sleep(3600)
            now = dt.datetime.utcnow()
            to_remove = []
            for jid, job in self.jobs.items():
                if job.status in ["done", "error", "cancelled"] and job.finished_at:
                    age = (now - job.finished_at).total_seconds()
                    if age > 86400: 
                        to_remove.append(jid)
            for jid in to_remove:
                self.jobs.pop(jid, None)
            if to_remove:
                print(f"[Memory Cleaner] Removed {len(to_remove)} old jobs from RAM.")

    def prune_disk_files(self, days: int) -> int:
        if days < 3: raise ValueError("Safety guard: Minimum retention is 3 days.")
        now = time.time()
        cutoff = now - (days * 86400)
        count = 0
        for filename in os.listdir(RESULTS_DIR):
            if not (filename.endswith(".wav") or filename.endswith(".mp3")): continue
            filepath = os.path.join(RESULTS_DIR, filename)
            try:
                mtime = os.path.getmtime(filepath)
                if mtime < cutoff:
                    os.remove(filepath)
                    count += 1
                    job_id = os.path.splitext(filename)[0]
                    self.jobs.pop(job_id, None)
            except OSError: pass
        return count

    async def _admit(self, job_id):
        async with self.sema:
            job = self.jobs.get(job_id)
            if not job: return
            if job.status == "cancelled": return
            await asyncio.to_thread(self._execute_sync, job)

    @torch.inference_mode()
    def _execute_sync(self, job: TTSJob):
        if job.status == "cancelled": return

        job.status = "running"
        try:
            voice: Voice = self.registry.get(job.voice_id)
        except KeyError as e:
            job.status = "error"
            job.error = str(e)
            job.finished_at = dt.datetime.utcnow()
            return

        num_step = job.num_step if job.num_step is not None else self.defaults["num_step"]
        guidance = job.guidance_scale if job.guidance_scale is not None else self.defaults["guidance_scale"]

        def on_progress(done: int, total: int):
            if job.status == "cancelled":
                raise JobCancelledError("Job was cancelled by user request.")
            job.progress = float(done) / max(total, 1)

        input_text = normalize_vietnamese_text(job.text)

        try:
            with autocast(device_type=self.device.type):
                with torch.inference_mode():
                    _ = generate_sentence(
                        save_path=job.out_wav_path,
                        prompt_text=voice.prompt_text,
                        prompt_wav=voice.prompt_wav,
                        text = input_text,
                        model=self.model,
                        vocoder=self.vocoder,
                        tokenizer=self.tokenizer,
                        feature_extractor=self.feature_extractor,
                        device=self.device,
                        num_step=num_step,
                        guidance_scale=guidance,
                        speed=job.speed,
                        sampling_rate=self.sampling_rate,
                        max_duration=MAX_DURATION,
                        remove_long_sil=job.remove_long_sil,
                        progress_cb=on_progress, 
                    )
                
                if job.status == "cancelled":
                    raise JobCancelledError("Job cancelled at finalization.")

                if job.audio_type == "mp3":
                    mp3_path = job.out_wav_path.replace(".wav", ".mp3")
                    import subprocess
                    # specific ffmpeg command for mp3 conversion
                    cmd = ["ffmpeg", "-y", "-i", job.out_wav_path, "-c:a", "libmp3lame", "-b:a", "64k", mp3_path]
                    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                    if os.path.exists(job.out_wav_path):
                        os.remove(job.out_wav_path)
                    job.out_wav_path = mp3_path

                job.progress = 1.0
                job.status = "done"
            
        except JobCancelledError:
            # --- CANCELLATION CLEANUP ---
            job.status = "cancelled"
            job.progress = 0.0
            if os.path.exists(job.out_wav_path):
                try: os.remove(job.out_wav_path)
                except: pass
            
            # --- VRAM/GPU CLEANUP ---
            # 1. Force Python Garbage Collector to release tensor references
            gc.collect() 
            
            # 2. Force PyTorch to release cached allocator memory back to GPU
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            print(f"[Engine] Job {job.id} cancelled. VRAM cleared.")
            # ---------------------------

        except Exception as e:
            job.status = "error"
            job.error = str(e)
            # Optional: Clear cache on error too if you suspect OOM caused it
            if "CUDA out of memory" in str(e):
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
        finally:
            job.finished_at = dt.datetime.utcnow()