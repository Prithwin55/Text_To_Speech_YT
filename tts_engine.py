"""Core TTS engine: Coqui XTTS-v2 with voice cloning, duration control, and BGM mixing."""

import os
import re
import tempfile
from datetime import datetime

import numpy as np
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

DEFAULT_SPEAKERS: dict[str, str] = {
    "Female": "Claribel Dervla",
    "Male": "Damien Black",
}

MAX_CHUNK_CHARS = 220
CHUNK_PAUSE_MS = 380

_tts_model = None


def get_model():
    global _tts_model
    if _tts_model is None:
        from TTS.api import TTS
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"[TTS] Loading XTTS-v2 on {device} (first run downloads ~1.8 GB)…")
        _tts_model = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to(device)
        print("[TTS] Model ready.")
    return _tts_model


def split_into_chunks(text: str, max_chars: int = MAX_CHUNK_CHARS) -> list[str]:
    """Split on sentence boundaries, handling Latin (. ! ?) and Devanagari (।)."""
    text = re.sub(r"[ \t]+", " ", text).strip()
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
    if len(text) <= max_chars:
        out.append(text)
        return
    for part in re.split(r"(?<=[,;،])\s*", text):
        part = part.strip()
        if part:
            out.append(part)


# ──────────────────────────────────────────────────────────────────────────────
# Duration control
# ──────────────────────────────────────────────────────────────────────────────

def _stretch_to_duration(seg: AudioSegment, target_ms: int) -> AudioSegment:
    """Time-stretch seg to target_ms milliseconds (pitch-preserving via librosa)."""
    import librosa

    current_ms = len(seg)
    if abs(current_ms - target_ms) < 200:
        return seg  # already close enough

    rate = current_ms / target_ms  # >1 = speed up, <1 = slow down
    sr = seg.frame_rate

    samples = np.array(seg.get_array_of_samples(), dtype=np.float32)
    samples /= 2 ** (seg.sample_width * 8 - 1)

    if seg.channels == 2:
        samples = samples.reshape(-1, 2).mean(axis=1)

    stretched = librosa.effects.time_stretch(y=samples, rate=rate)

    # Trim or pad to exact target length
    target_samples = int(target_ms * sr / 1000)
    if len(stretched) > target_samples:
        stretched = stretched[:target_samples]
    elif len(stretched) < target_samples:
        stretched = np.pad(stretched, (0, target_samples - len(stretched)))

    int_samples = (np.clip(stretched, -1.0, 1.0) * 32767).astype(np.int16)
    return AudioSegment(
        int_samples.tobytes(),
        frame_rate=sr,
        sample_width=2,
        channels=1,
    )


# ──────────────────────────────────────────────────────────────────────────────
# Trimming
# ──────────────────────────────────────────────────────────────────────────────

def trim_audio(audio_path: str, start_sec: float, end_sec: float, output_dir: str) -> str:
    """Trim audio_path to [start_sec, end_sec] and save as WAV. Returns new path."""
    audio = AudioSegment.from_file(audio_path)
    dur = len(audio) / 1000.0
    start_ms = max(0, int(start_sec * 1000))
    end_ms = min(int(end_sec * 1000), len(audio))
    if start_ms >= end_ms:
        raise ValueError(
            f"Invalid trim range {start_sec:.1f}s–{end_sec:.1f}s "
            f"(audio is {dur:.1f}s)."
        )
    trimmed = audio[start_ms:end_ms]
    os.makedirs(output_dir, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = os.path.join(output_dir, f"trimmed_{ts}.wav")
    trimmed.export(out_path, format="wav")
    return out_path


# ──────────────────────────────────────────────────────────────────────────────
# BGM mixing
# ──────────────────────────────────────────────────────────────────────────────

def mix_with_bgm(
    speech_path: str,
    bgm_path: str,
    bgm_volume_db: float,
    output_dir: str,
) -> str:
    """
    Mix background music under the speech audio.
    BGM is looped if shorter than speech, then volume-reduced by bgm_volume_db.
    Speech stays dominant. Returns path to the mixed MP3.
    """
    speech = AudioSegment.from_file(speech_path)
    bgm = AudioSegment.from_file(bgm_path)

    # Normalise BGM to speech format so overlay works cleanly
    bgm = (
        bgm
        .set_frame_rate(speech.frame_rate)
        .set_channels(speech.channels)
        .set_sample_width(speech.sample_width)
    )

    # Loop BGM until it covers the full speech duration
    if len(bgm) < len(speech):
        repeats = -(-len(speech) // len(bgm))  # ceiling division
        bgm = bgm * repeats
    bgm = bgm[: len(speech)]

    # Reduce BGM volume (bgm_volume_db is negative, e.g. -18)
    bgm = bgm.apply_gain(bgm_volume_db)

    mixed = bgm.overlay(speech)

    os.makedirs(output_dir, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = os.path.join(output_dir, f"mixed_{ts}.mp3")
    mixed.export(out_path, format="mp3")
    return out_path


# ──────────────────────────────────────────────────────────────────────────────
# Main generation
# ──────────────────────────────────────────────────────────────────────────────

def generate_speech(
    text: str,
    language: str,
    reference_audio: str | None,
    gender: str,
    output_format: str,
    output_dir: str,
    target_duration: float | None = None,
    progress_callback=None,
) -> str:
    """
    Generate speech, optionally time-stretched to target_duration seconds.
    Returns path to the output file.
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

        # Optional duration stretch
        if target_duration and target_duration > 0:
            if progress_callback:
                progress_callback(1.0, desc="Adjusting duration…")
            combined = _stretch_to_duration(combined, int(target_duration * 1000))

        os.makedirs(output_dir, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_path = os.path.join(output_dir, f"story_{ts}.{fmt}")
        combined.export(out_path, format=fmt)

    return out_path
