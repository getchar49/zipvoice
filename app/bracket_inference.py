"""
Bracket-aware inference for ZipVoice TTS.

Parses normalized text containing 【X】 markers (letters/words in CJK brackets)
and generates audio with different speed/step parameters for bracket segments
vs normal segments, then concatenates all segments via cross-fade.

Bracket segments (【ây ai】, 【kây pi ai】, 【C】 【E】 【O】, etc.) are generated with:
  - speed = BRACKET_SPEED (default 0.5)
  - num_step = BRACKET_NUM_STEP (default 64)

Normal segments are generated with the user's requested speed/step.
"""

import logging
import re
from dataclasses import dataclass
from typing import List, Optional, Callable

import torch
import torchaudio

from zipvoice.utils.infer import cross_fade_concat

logger = logging.getLogger(__name__)


@dataclass
class TextSegment:
    """A segment of text, either normal or bracket-marked."""
    text: str
    is_bracket: bool


def parse_bracketed_text(text: str) -> List[TextSegment]:
    """
    Parse text containing 【X】 bracket markers into segments.

    Consecutive 【X】 markers (possibly separated by spaces, commas, or dots)
    are grouped into a single bracket segment. Everything else is a normal segment.

    Args:
        text: Normalized text potentially containing 【X】 markers.

    Returns:
        List of TextSegment objects in order. Empty segments are excluded.

    Examples:
        >>> parse_bracketed_text("Kiểm tra 【A】 【I】, Hà Nội.")
        [
            TextSegment(text="Kiểm tra", is_bracket=False),
            TextSegment(text="A I,", is_bracket=True),
            TextSegment(text="Hà Nội.", is_bracket=False),
        ]
    """
    if not text or "【" not in text:
        return [TextSegment(text=text.strip(), is_bracket=False)] if text and text.strip() else []

    # Pattern to match a single 【X】 token (letters/words inside 【】)
    # Supports single letters (【A】) and multi-word transliterations (【kây pi ai】)
    # Content must start with a letter (not digit/space) to avoid matching non-markers
    bracket_pattern = re.compile(r'【([a-zA-Z\u00C0-\u1EF9][^】]{0,49})】')

    segments: List[TextSegment] = []
    pos = 0
    current_bracket_parts: List[str] = []
    trailing_punct = ""

    while pos < len(text):
        m = bracket_pattern.search(text, pos)
        if m is None:
            # No more brackets — rest is normal text
            remaining = text[pos:].strip()
            # Strip stray 【】 from normal text (they can't be rendered)
            remaining = re.sub(r'[【】]', '', remaining).strip()
            if current_bracket_parts:
                # Flush bracket segment
                bracket_text = " ".join(current_bracket_parts)
                if trailing_punct:
                    bracket_text += trailing_punct
                    trailing_punct = ""
                segments.append(TextSegment(text=bracket_text, is_bracket=True))
                current_bracket_parts = []
            if remaining:
                segments.append(TextSegment(text=remaining, is_bracket=False))
            break

        # Text before this bracket
        before = text[pos:m.start()]

        # Check if 'before' is just whitespace/punctuation between consecutive brackets
        before_stripped = before.strip()
        is_separator = bool(re.match(r'^[\s,.\u3001\uff0c]*$', before_stripped)) or not before_stripped

        if is_separator and current_bracket_parts:
            # This is continuation of a bracket sequence — capture any punctuation
            punct_match = re.search(r'[,.\u3001\uff0c]', before_stripped)
            if punct_match:
                trailing_punct = punct_match.group()
            current_bracket_parts.append(m.group(1))
        else:
            # Flush previous bracket segment if any
            if current_bracket_parts:
                bracket_text = " ".join(current_bracket_parts)
                if trailing_punct:
                    bracket_text += trailing_punct
                    trailing_punct = ""
                segments.append(TextSegment(text=bracket_text, is_bracket=True))
                current_bracket_parts = []

            # Add normal text before this bracket (strip stray 【】)
            normal_text = re.sub(r'[【】]', '', before_stripped).strip()
            if normal_text:
                segments.append(TextSegment(text=normal_text, is_bracket=False))

            # Start new bracket sequence
            current_bracket_parts.append(m.group(1))

        pos = m.end()

    # Flush any remaining bracket segment
    if current_bracket_parts:
        bracket_text = " ".join(current_bracket_parts)
        if trailing_punct:
            bracket_text += trailing_punct
        segments.append(TextSegment(text=bracket_text, is_bracket=True))

    # Final cleanup: strip 【】 from normal segments, drop empty ones
    cleaned = []
    for s in segments:
        if not s.is_bracket:
            s.text = re.sub(r'[【】]', '', s.text).strip()
        if s.text.strip():
            cleaned.append(s)
    return cleaned


def has_brackets(text: str) -> bool:
    """Check if text contains any 【X】 bracket markers."""
    return bool(re.search(r'【[a-zA-Z\u00C0-\u1EF9][^】]{0,49}】', text))


@torch.inference_mode()
def generate_sentence_with_brackets(
    save_path: str,
    prompt_text: str,
    prompt_wav: str,
    text: str,
    model: torch.nn.Module,
    vocoder: torch.nn.Module,
    tokenizer,
    feature_extractor,
    device: torch.device,
    num_step: int = 32,
    guidance_scale: float = 1.0,
    speed: float = 1.0,
    sampling_rate: int = 24000,
    max_duration: float = 100,
    remove_long_sil: bool = False,
    progress_cb: Optional[Callable[[int, int], None]] = None,
    bracket_speed: float = 0.5,
    bracket_num_step: int = 64,
):
    """
    Generate audio for text containing 【X】 bracket markers.

    Splits text into bracket/normal segments, generates each with appropriate
    speed/step parameters, then cross-fade concatenates all segments.

    If text has no brackets, falls back to standard generate_sentence().

    Args:
        save_path: Path to save the final concatenated wav.
        prompt_text: Transcription of the prompt wav.
        prompt_wav: Path to the prompt wav file.
        text: Normalized text with potential 【X】 markers.
        model: The ZipVoice model.
        vocoder: The vocoder model.
        tokenizer: Text tokenizer.
        feature_extractor: Audio feature extractor.
        device: Torch device.
        num_step: Default number of sampling steps for normal segments.
        guidance_scale: Classifier-free guidance scale.
        speed: Default speed for normal segments.
        sampling_rate: Audio sampling rate.
        max_duration: Max duration per batch.
        remove_long_sil: Whether to remove long silences.
        progress_cb: Progress callback.
        bracket_speed: Speed for bracket segments (default 0.5).
        bracket_num_step: Number of steps for bracket segments (default 64).

    Returns:
        metrics dict with timing information.
    """
    from zipvoice.bin.infer_zipvoice import generate_sentence

    # Parse segments
    segments = parse_bracketed_text(text)

    # If no brackets or only one segment, use standard generation
    if not any(s.is_bracket for s in segments):
        plain_text = " ".join(s.text for s in segments)
        return generate_sentence(
            save_path=save_path,
            prompt_text=prompt_text,
            prompt_wav=prompt_wav,
            text=plain_text,
            model=model,
            vocoder=vocoder,
            tokenizer=tokenizer,
            feature_extractor=feature_extractor,
            device=device,
            num_step=num_step,
            guidance_scale=guidance_scale,
            speed=speed,
            sampling_rate=sampling_rate,
            max_duration=max_duration,
            remove_long_sil=remove_long_sil,
            progress_cb=progress_cb,
        )

    import datetime as dt
    import tempfile
    import os

    logger.info(f"[Bracket Inference] Parsed {len(segments)} segments: "
                f"{sum(1 for s in segments if s.is_bracket)} bracket, "
                f"{sum(1 for s in segments if not s.is_bracket)} normal")

    start_t = dt.datetime.now()
    segment_wavs: List[torch.Tensor] = []

    # Calculate total segments for progress
    total_segments = len(segments)
    done_segments = 0

    for i, seg in enumerate(segments):
        seg_speed = bracket_speed if seg.is_bracket else speed
        seg_step = bracket_num_step if seg.is_bracket else num_step
        seg_type = "BRACKET" if seg.is_bracket else "NORMAL"

        # Skip segments with no speakable content (only punctuation/whitespace)
        speakable = re.sub(r'[\s.,;:!?…—–\-\'"()\[\]{}]', '', seg.text)
        if not speakable:
            logger.warning(f"[Bracket Inference] Skipping segment {i+1}/{total_segments} "
                           f"with no speakable content: '{seg.text}'")
            done_segments += 1
            continue

        logger.info(f"[Bracket Inference] Segment {i+1}/{total_segments} "
                     f"({seg_type}): '{seg.text[:50]}...' "
                     f"speed={seg_speed}, step={seg_step}")

        # Generate to temp file
        tmp_dir = os.path.dirname(save_path) or "."
        tmp_path = os.path.join(tmp_dir, f"_bracket_seg_{i}_{os.getpid()}.wav")

        try:
            def segment_progress(done, total):
                if progress_cb:
                    # Map segment progress to overall progress
                    overall = (done_segments + done / max(total, 1)) / max(total_segments, 1)
                    progress_cb(int(overall * 100), 100)

            generate_sentence(
                save_path=tmp_path,
                prompt_text=prompt_text,
                prompt_wav=prompt_wav,
                text=seg.text,
                model=model,
                vocoder=vocoder,
                tokenizer=tokenizer,
                feature_extractor=feature_extractor,
                device=device,
                num_step=seg_step,
                guidance_scale=guidance_scale,
                speed=seg_speed,
                sampling_rate=sampling_rate,
                max_duration=max_duration,
                remove_long_sil=remove_long_sil,
                progress_cb=segment_progress,
            )

            # Load generated segment
            wav, sr = torchaudio.load(tmp_path)
            if sr != sampling_rate:
                resampler = torchaudio.transforms.Resample(sr, sampling_rate)
                wav = resampler(wav)
            segment_wavs.append(wav)

        except Exception as e:
            logger.error(f"[Bracket Inference] Failed to generate segment {i+1}/{total_segments} "
                         f"('{seg.text[:50]}'): {e}")
            # Skip failed segment instead of crashing the entire synthesis
            logger.warning(f"[Bracket Inference] Skipping failed segment {i+1} and continuing")
            done_segments += 1
            continue
        finally:
            # Clean up temp file
            if os.path.exists(tmp_path):
                try:
                    os.remove(tmp_path)
                except OSError:
                    pass

        done_segments += 1

    # Cross-fade concatenate all segments
    if segment_wavs:
        final_wav = cross_fade_concat(
            segment_wavs, fade_duration=0.1, sample_rate=sampling_rate
        )
    else:
        final_wav = torch.zeros(1, sampling_rate)  # 1s silence fallback

    # Save final audio
    torchaudio.save(save_path, final_wav.cpu(), sample_rate=sampling_rate)

    # Calculate metrics
    t = (dt.datetime.now() - start_t).total_seconds()
    wav_seconds = final_wav.shape[-1] / sampling_rate
    metrics = {
        "t": t,
        "t_no_vocoder": t,   # approximate — vocoder is inside generate_sentence
        "t_vocoder": 0.0,
        "wav_seconds": wav_seconds,
        "rtf": t / max(wav_seconds, 0.001),
        "rtf_no_vocoder": t / max(wav_seconds, 0.001),
        "rtf_vocoder": 0.0,
    }

    if progress_cb:
        try:
            progress_cb(100, 100)
        except Exception:
            pass

    logger.info(f"[Bracket Inference] Done. Total time: {t:.2f}s, "
                f"Audio: {wav_seconds:.2f}s, RTF: {metrics['rtf']:.4f}")

    return metrics
