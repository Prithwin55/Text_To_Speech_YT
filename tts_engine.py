"""Core TTS engine: Coqui XTTS-v2 with voice cloning and multilingual support."""

import os
import re
import tempfile
from datetime import datetime
from pathlib import Path

import torch
from pydub import AudioSegment

# Accept Coqui TOS automatically (model download)
os.environ.setdefault("COQUI_TOS_AGREED", "1")

SUPPORTED_LANGUAGES: dict[str, str] = {
    "English": "en",
    "Hindi": "hi",
    "Spanish": "es",
    "French": "fr",
    "German": "de",
    "Italian": "it",
    "Portuguese": "pt",
    "Polish": "pl",
    "Turkish": "tr",
    "Russian": "ru",
    "Dutch": "nl",
    "Czech": "cs",
    "Arabic": "ar",
    "Chinese": "zh-cn",
    "Japanese": "ja",
    "Korean": "ko",
    "Hungarian": "hu",
}

# Built-in speakers used when no reference audio is supplied
DEFAULT_SPEAKERS: dict[str, str] = {
    "Female": "Claribel Dervla",
    "Male": "Damien Black",
}
# XTTS-v2 hard limit is ~400 chars; stay well below for reliable output
MAX_CHUNK_CHARS = 220
# Silence (ms) inserted between chunks in the final audio
CHUNK_PAUSE_MS = 380

_tts_model = None


def get_model():
    global _tts_model
    if _tts_model is None:
        from TTS.api import TTS  # lazy import so the module loads fast

        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"[TTS] Loading XTTS-v2 on {device} (first run downloads ~1.8 GB)…")
        _tts_model = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to(device)
        print("[TTS] Model ready.")
    return _tts_model


def split_into_chunks(text: str, max_chars: int = MAX_CHUNK_CHARS) -> list[str]:
    """
    Split text into ≤max_chars chunks, respecting sentence boundaries.
    Handles both Latin punctuation (. ! ?) and Devanagari danda (।).
    """
    # Normalise whitespace
    text = re.sub(r"[ \t]+", " ", text).strip()

    # Split on sentence-ending punctuation
    sentences = re.split(r"(?<=[.!?।])\s+", text)
    sentences = [s.strip() for s in sentences if s.strip()]

    chunks: list[str] = []
    buf = ""

    for sent in sentences:
        if not buf:
            buf = sent
        elif len(buf) + 1 + len(sent) <= max_chars:
            buf += " " + sent
        else:
            _flush(buf, max_chars, chunks)
            buf = sent

    if buf:
        _flush(buf, max_chars, chunks)

    return [c for c in chunks if c]


def _flush(text: str, max_chars: int, out: list[str]) -> None:
    """Append text to out, splitting on commas if it exceeds max_chars."""
    if len(text) <= max_chars:
        out.append(text)
        return
    # Force-split on commas / semicolons / Arabic comma (،)
    for part in re.split(r"(?<=[,;،])\s*", text):
        part = part.strip()
        if part:
            out.append(part)


def generate_speech(
    text: str,
    language: str,
    reference_audio: str | None,
    gender: str,
    output_format: str,
    output_dir: str,
    progress_callback=None,
) -> str:
    """
    Generate speech for *text* and write the result to output_dir.
    Returns the absolute path of the output file.

    Args:
        text:             Full story text.
        language:         BCP-47 language code (e.g. "hi", "en").
        reference_audio:  Path to reference WAV/MP3 for voice cloning, or None.
        gender:           "Female" or "Male" — used when reference_audio is absent.
        output_format:    "mp3" or "wav" (case-insensitive).
        output_dir:       Directory where the output file will be written.
        progress_callback: Optional callable(fraction, desc=str).
    """
    tts = get_model()
    chunks = split_into_chunks(text)
    total = len(chunks)

    if total == 0:
        raise ValueError("No speakable text found after splitting.")

    fmt = output_format.lower()
    segments: list[AudioSegment] = []

    with tempfile.TemporaryDirectory() as tmp:
        for i, chunk in enumerate(chunks):
            if progress_callback:
                progress_callback((i + 1) / total, desc=f"Generating chunk {i + 1}/{total}…")

            chunk_path = os.path.join(tmp, f"chunk_{i:04d}.wav")
            kwargs: dict = {"text": chunk, "language": language, "file_path": chunk_path}

            if reference_audio and os.path.isfile(reference_audio):
                kwargs["speaker_wav"] = reference_audio
            else:
                kwargs["speaker"] = DEFAULT_SPEAKERS.get(gender, DEFAULT_SPEAKERS["Female"])

            tts.tts_to_file(**kwargs)
            segments.append(AudioSegment.from_wav(chunk_path))

        combined = segments[0]
        for seg in segments[1:]:
            combined += AudioSegment.silent(duration=CHUNK_PAUSE_MS) + seg

        os.makedirs(output_dir, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_path = os.path.join(output_dir, f"story_{ts}.{fmt}")
        combined.export(out_path, format=fmt)

    return out_path
