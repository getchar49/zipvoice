"""
Cached inference for ZipVoice TTS.

Provides generate_sentence_cached() which is equivalent to
zipvoice.bin.infer_zipvoice.generate_sentence() but accepts
pre-processed prompt tensors instead of a file path.

This avoids redundant I/O (torchaudio.load), resampling, silence removal,
RMS normalization, and feature extraction on every inference call for the
same voice.
"""

import logging
from typing import Optional, Callable

import torch
import torchaudio
import datetime as dt

from zipvoice.utils.infer import (
    add_punctuation,
    chunk_tokens_punctuation,
    batchify_tokens,
    cross_fade_concat,
    remove_silence,
)

logger = logging.getLogger(__name__)


@torch.inference_mode()
def generate_sentence_cached(
    save_path: str,
    prompt_text: str,
    # --- Pre-cached prompt data (replaces prompt_wav: str) ---
    prompt_wav_tensor: torch.Tensor,       # (C, T) — already loaded, silence-removed, rms-normed
    prompt_rms: float,                      # Original RMS before normalization
    prompt_features: torch.Tensor,          # (1, T, C) — already extracted, unsqueezed, scaled
    # --- Standard parameters ---
    text: str,
    model: torch.nn.Module,
    vocoder: torch.nn.Module,
    tokenizer,
    feature_extractor,
    device: torch.device,
    num_step: int = 16,
    guidance_scale: float = 1.0,
    speed: float = 1.0,
    t_shift: float = 0.5,
    target_rms: float = 0.1,
    feat_scale: float = 0.1,
    sampling_rate: int = 24000,
    max_duration: float = 100,
    remove_long_sil: bool = False,
    progress_cb: Optional[Callable[[int, int], None]] = None,
):
    """
    Generate waveform using pre-cached prompt data.

    Equivalent to zipvoice.bin.infer_zipvoice.generate_sentence() but
    skips the following per-call overhead:
      - torchaudio.load(prompt_wav)
      - remove_silence(prompt_wav, ...)
      - rms_norm(prompt_wav, target_rms)
      - feature_extractor.extract(prompt_wav, ...)

    All other logic (punctuation, tokenization, chunking, batching,
    generation, vocoder, cross-fade) is identical.

    Args:
        save_path: Path to save the generated wav.
        prompt_text: Transcription of the prompt wav.
        prompt_wav_tensor: Pre-processed prompt waveform tensor (C, T).
        prompt_rms: Original RMS of the prompt waveform.
        prompt_features: Pre-extracted features tensor (1, T_feat, C_feat),
                         already unsqueezed and scaled (* feat_scale).
        text: Text to synthesize.
        model, vocoder, tokenizer, feature_extractor: Model components.
        device: Torch device.
        num_step: Number of sampling steps.
        guidance_scale: Classifier-free guidance scale.
        speed: Speech speed control.
        t_shift: Time shift parameter.
        target_rms: Target RMS for volume normalization.
        feat_scale: Feature scale factor.
        sampling_rate: Audio sampling rate.
        max_duration: Max duration per batch (seconds).
        remove_long_sil: Whether to remove long silences.
        progress_cb: Progress callback.

    Returns:
        metrics dict with timing information.
    """
    # Move prompt features to device
    prompt_features_dev = prompt_features.to(device)

    prompt_duration = prompt_wav_tensor.shape[-1] / sampling_rate

    # Add punctuation in the end if there is not
    text = add_punctuation(text)
    prompt_text = add_punctuation(prompt_text)

    # Tokenize text (str tokens), punctuations will be preserved.
    tokens_str = tokenizer.texts_to_tokens([text])[0]
    prompt_tokens_str = tokenizer.texts_to_tokens([prompt_text])[0]

    # Chunk text so that each len(prompt wav + generated wav) is around 25 seconds.
    token_duration = (prompt_wav_tensor.shape[-1] / sampling_rate) / (
        len(prompt_tokens_str) * speed
    )
    max_tokens = int((25 - prompt_duration) / token_duration)
    chunked_tokens_str = chunk_tokens_punctuation(tokens_str, max_tokens=max_tokens)

    # Tokenize text (int tokens)
    chunked_tokens = tokenizer.tokens_to_token_ids(chunked_tokens_str)
    prompt_tokens = tokenizer.tokens_to_token_ids([prompt_tokens_str])

    # Batchify chunked texts for faster processing
    tokens_batches, chunked_index = batchify_tokens(
        chunked_tokens, max_duration, prompt_duration, token_duration
    )

    GEN_W, VOC_W = 95, 5
    total_gen_units = sum(len(b) for b in tokens_batches) or 1
    total_voc_units = total_gen_units
    total_units = GEN_W * total_gen_units + VOC_W * total_voc_units
    done_units = 0

    # Start predicting features
    start_t = dt.datetime.now()
    chunked_wavs_cpu = []

    for batch_idx, batch_tokens in enumerate(tokens_batches):
        batch_prompt_tokens = prompt_tokens * len(batch_tokens)

        batch_prompt_features = prompt_features_dev.repeat(len(batch_tokens), 1, 1)
        batch_prompt_features_lens = torch.full(
            (len(batch_tokens),), prompt_features_dev.size(1), device=device
        )

        # Generate features
        (
            pred_features,
            pred_features_lens,
            pred_prompt_features,
            pred_prompt_features_lens,
        ) = model.sample(
            tokens=batch_tokens,
            prompt_tokens=batch_prompt_tokens,
            prompt_features=batch_prompt_features,
            prompt_features_lens=batch_prompt_features_lens,
            speed=speed,
            t_shift=t_shift,
            duration="predict",
            num_step=num_step,
            guidance_scale=guidance_scale,
        )

        # Postprocess predicted features
        pred_features = pred_features.permute(0, 2, 1) / feat_scale  # (B, C, T)
        for i in range(pred_features.size(0)):
            feat_i = pred_features[i][None, :, : pred_features_lens[i]]
            wav_gpu = vocoder.decode(feat_i).squeeze(1).clamp(-1, 1)
            if prompt_rms < target_rms:
                wav_gpu = wav_gpu * prompt_rms / target_rms
            wav_cpu = wav_gpu.cpu()
            global_chunk_index = chunked_index[
                sum(len(b) for b in tokens_batches[:batch_idx]) + i
            ]
            chunked_wavs_cpu.append((global_chunk_index, wav_cpu))
        del pred_features, pred_features_lens, pred_prompt_features, pred_prompt_features_lens
        torch.cuda.empty_cache()
        if progress_cb:
            try:
                done_units += GEN_W * len(batch_tokens)
                progress_cb(done_units, total_units)
            except Exception:
                pass

    # Merge chunked wavs
    start_vocoder_t = dt.datetime.now()
    t = (dt.datetime.now() - start_t).total_seconds()

    sequential_indexed_chunked_wavs = sorted(chunked_wavs_cpu, key=lambda x: x[0])
    sequential_chunked_wavs = [w[1] for w in sequential_indexed_chunked_wavs]
    final_wav = cross_fade_concat(
        sequential_chunked_wavs, fade_duration=0.1, sample_rate=sampling_rate
    )
    final_wav = remove_silence(
        final_wav, sampling_rate, only_edge=(not remove_long_sil), trail_sil=0
    )

    # Calculate metrics
    t_no_vocoder = (start_vocoder_t - start_t).total_seconds()
    t_vocoder = (dt.datetime.now() - start_vocoder_t).total_seconds()
    wav_seconds = final_wav.shape[-1] / sampling_rate
    rtf = t / wav_seconds
    rtf_no_vocoder = t_no_vocoder / wav_seconds
    rtf_vocoder = t_vocoder / wav_seconds
    metrics = {
        "t": t,
        "t_no_vocoder": t_no_vocoder,
        "t_vocoder": t_vocoder,
        "wav_seconds": wav_seconds,
        "rtf": rtf,
        "rtf_no_vocoder": rtf_no_vocoder,
        "rtf_vocoder": rtf_vocoder,
    }

    torchaudio.save(save_path, final_wav.cpu(), sample_rate=sampling_rate)
    if progress_cb:
        try:
            progress_cb(total_units, total_units)
        except Exception:
            pass
    return metrics
