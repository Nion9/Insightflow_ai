"""YouTube audio download + Whisper transcription.

Cross-platform ffmpeg detection: PATH first, then common install locations.
Whisper models are cached in-process so repeated transcriptions reuse them.
"""
from __future__ import annotations

import os
import shutil
import tempfile
from pathlib import Path
from typing import Dict, Optional

import whisper
import yt_dlp


# ---------- ffmpeg discovery ----------

class FFmpegNotFoundError(RuntimeError):
    """Raised when ffmpeg cannot be located on the system."""


def _find_ffmpeg_dir() -> Optional[str]:
    """Return the directory containing ffmpeg, or None if not found."""
    on_path = shutil.which("ffmpeg")
    if on_path:
        return str(Path(on_path).parent)

    exe = "ffmpeg.exe" if os.name == "nt" else "ffmpeg"
    candidates = [
        Path("C:/ffmpeg/bin"),
        Path("C:/Program Files/ffmpeg/bin"),
        Path("/usr/local/bin"),
        Path("/usr/bin"),
        Path("/opt/homebrew/bin"),
        Path.home() / "ffmpeg" / "bin",
    ]
    for c in candidates:
        if (c / exe).exists():
            return str(c)
    return None


FFMPEG_DIR = _find_ffmpeg_dir()
if FFMPEG_DIR and FFMPEG_DIR not in os.environ.get("PATH", ""):
    os.environ["PATH"] = FFMPEG_DIR + os.pathsep + os.environ.get("PATH", "")


# ---------- Audio download ----------

def download_audio(url: str, output_dir: Optional[str] = None) -> str:
    """Download a YouTube video's audio as MP3.

    Args:
        url: YouTube URL.
        output_dir: Where to write the MP3. A new temp dir is used if omitted.

    Returns:
        Absolute path to the MP3 file.
    """
    if FFMPEG_DIR is None:
        raise FFmpegNotFoundError(
            "ffmpeg not found on PATH. Install it and retry:\n"
            "  • Windows: https://www.gyan.dev/ffmpeg/builds/\n"
            "  • macOS:   brew install ffmpeg\n"
            "  • Linux:   sudo apt install ffmpeg"
        )

    workdir = output_dir or tempfile.mkdtemp(prefix="insightflow_")
    template = str(Path(workdir) / "audio.%(ext)s")

    ydl_opts = {
        "format": "bestaudio/best",
        "outtmpl": template,
        "postprocessors": [{
            "key": "FFmpegExtractAudio",
            "preferredcodec": "mp3",
            "preferredquality": "192",
        }],
        "ffmpeg_location": FFMPEG_DIR,
        "quiet": True,
        "no_warnings": True,
    }

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.download([url])

    mp3 = Path(workdir) / "audio.mp3"
    if not mp3.exists():
        raise FileNotFoundError(f"Audio extraction failed: {mp3}")
    return str(mp3)


# ---------- Whisper transcription ----------

_MODEL_CACHE: Dict[str, "whisper.Whisper"] = {}


def transcribe(audio_path: str, model_size: str = "base") -> str:
    """Transcribe an audio file with Whisper. Models are cached per size."""
    if not Path(audio_path).exists():
        raise FileNotFoundError(audio_path)

    if model_size not in _MODEL_CACHE:
        _MODEL_CACHE[model_size] = whisper.load_model(model_size)

    result = _MODEL_CACHE[model_size].transcribe(audio_path)
    return result["text"].strip()


# ---------- Cleanup ----------

def cleanup(audio_path: str) -> None:
    """Delete the audio file and its parent dir if it was a temp dir."""
    p = Path(audio_path)
    if not p.exists():
        return
    parent = p.parent
    try:
        p.unlink()
    except OSError:
        pass
    if parent.name.startswith("insightflow_"):
        shutil.rmtree(parent, ignore_errors=True)


# ---------- CLI ----------

if __name__ == "__main__":
    url = input("YouTube URL: ").strip()
    audio = None
    try:
        audio = download_audio(url)
        print(f"✓ Audio: {audio}")
        text = transcribe(audio)
        print("\n--- Transcript (first 500 chars) ---")
        print(text[:500])
        Path("transcript.txt").write_text(text, encoding="utf-8")
        print("\n✓ Saved to transcript.txt")
    finally:
        if audio:
            cleanup(audio)
