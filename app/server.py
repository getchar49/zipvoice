import os, asyncio
from fastapi import FastAPI, HTTPException, status, Query, Path
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from sse_starlette.sse import EventSourceResponse

from .settings import RESULTS_DIR
from .registry import VoiceRegistry
from .engine import ZipVoiceEngine
from .schemas import TTSJobCreate, JobCreateResponse, JobStatusResponse

app = FastAPI(title="ZipVoice TTS (local-only, JSON API)")

# (CORS and Init code remains the same...)
# CORS
try:
    from fastapi.middleware.cors import CORSMiddleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],  
        allow_methods=["*"],
        allow_headers=["*"],
    )
except Exception:
    pass

registry = VoiceRegistry()
engine = ZipVoiceEngine(registry=registry)
asyncio.get_event_loop().create_task(engine.run())

os.makedirs(RESULTS_DIR, exist_ok=True)
app.mount("/files", StaticFiles(directory=RESULTS_DIR), name="files")

@app.get("/v1/voices")
def list_voices():
    return {"voices": registry.list()}

@app.post("/v1/tts/jobs", response_model=JobCreateResponse, status_code=status.HTTP_202_ACCEPTED)
async def create_job(req: TTSJobCreate):
    # (Existing code...)
    try:
        registry.get(req.voice_id)
    except KeyError as e:
        raise HTTPException(404, str(e))

    job_id = await engine.submit(
        text=req.text,
        voice_id=req.voice_id,
        speed=req.speed,
        remove_long_sil=req.remove_long_sil,
        num_step=req.num_step,
        guidance_scale=req.guidance_scale,
        audio_type=req.audio_type,
    )
    return JobCreateResponse(
        job_id=job_id,
        status_url=f"/v1/jobs/{job_id}",
        audio_url=f"/v1/jobs/{job_id}/audio",
    )

@app.get("/v1/jobs/{job_id}", response_model=JobStatusResponse)
async def job_status(job_id: str):
    # (Existing code...)
    j = engine.get_job(job_id)
    if not j:
        raise HTTPException(404, "job not found")
        
    resp = JobStatusResponse(job_id=j.id, status=j.status, progress=j.progress)
    
    if j.status == "done":
        ext = "mp3" if j.out_wav_path.endswith(".mp3") else "wav"
        resp.file_path = f"/files/{job_id}.{ext}"
        resp.audio_url = f"/v1/jobs/{job_id}/audio"
    if j.status == "error":
        resp.error = j.error
    # Note: 'cancelled' status will just return the status string with no file_path
    return resp

# --- NEW: Cancel Specific Job ---
@app.delete("/v1/jobs/{job_id}", status_code=status.HTTP_200_OK)
async def cancel_job(job_id: str = Path(..., description="The ID of the job to cancel")):
    """
    Cancels a specific job. 
    If running, it stops generation. If queued, it removes it from processing line.
    """
    found = engine.cancel_job(job_id)
    if not found:
        raise HTTPException(status_code=404, detail="Job not found")
    
    return {"message": "Job cancellation requested", "job_id": job_id, "status": "cancelled"}

# --- NEW: Cancel All Jobs ---
@app.delete("/v1/jobs", status_code=status.HTTP_200_OK)
async def cancel_all_jobs():
    """
    Emergency Stop: Cancels ALL queued and running jobs.
    """
    count = engine.cancel_all_jobs()
    return {"message": "All active jobs cancelled", "count": count}
# ------------------------------

@app.get("/v1/jobs/{job_id}/events")
async def events(job_id: str):
    async def gen():
        while True:
            j = engine.get_job(job_id)
            if not j:
                yield {"event":"error","data":"not found"}; break
            
            yield {"event":"progress","data": str(j.progress)}
            
            # Check for cancelled status too
            if j.status in ("done", "error", "cancelled"):
                yield {"event":"status","data": j.status}
                break
            
            await asyncio.sleep(0.25)
    return EventSourceResponse(gen())

@app.get("/v1/jobs/{job_id}/audio")
async def get_audio(job_id: str):
    # (Existing code...)
    j = engine.get_job(job_id)
    if not j:
        raise HTTPException(404, "job not found")
        
    if j.status == "error":
        raise HTTPException(400, f"Job failed: {j.error}")
        
    if j.status == "cancelled":
        raise HTTPException(409, "Job was cancelled")

    if j.status != "done":
        raise HTTPException(409, "job not finished yet")
        
    ext = "mp3" if j.out_wav_path.endswith(".mp3") else "wav"
    media_type = "audio/mpeg" if ext == "mp3" else "audio/wav"
    return FileResponse(j.out_wav_path, media_type=media_type, filename=f"{job_id}.{ext}")

@app.delete("/v1/files", status_code=status.HTTP_200_OK)
def prune_audio_files(days: int = Query(..., description="Delete files older than these many days (min 3)")):
    # (Existing code...)
    if days < 3:
        raise HTTPException(status_code=400, detail="For safety, minimum retention period is 3 days.")
    try:
        count = engine.prune_disk_files(days)
        return {"message": "Cleanup successful", "deleted_files": count}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))