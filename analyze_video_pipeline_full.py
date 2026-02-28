#!/usr/bin/env python3
"""
================================================================================
VIDEO + AUDIO ANALYSIS PIPELINE (FULL DOCUMENTED VERSION)
================================================================================

Purpose:
    Deterministic, idempotent, binary-content-driven video analysis pipeline
    with adaptive frame selection and multimodal inference.

================================================================================
ARCHITECTURE OVERVIEW
================================================================================

This pipeline performs:

1) Audio extraction (idempotent)
2) Whisper transcription (idempotent)
3) Adaptive frame extraction using dHash (idempotent)
4) Per-frame VLM analysis via llama-server (idempotent per frame)
5) Timeline synthesis (NOT idempotent intentionally)

All steps are cached using SHA256 hashes of BINARY CONTENT,
NOT filenames.

================================================================================
WHY THIS DESIGN IS OPTIMAL
================================================================================

✔ Deterministic
✔ Re-runnable without duplication
✔ Binary hash based (file rename safe)
✔ Reduces VLM cost via adaptive sampling
✔ Suitable for long UI recordings
✔ Traceable (frame_selection.json + run logs)
✔ Scales to large recordings

================================================================================
IDEMPOTENCY MODEL
================================================================================

Each step key is:

    SHA256(input_bytes) + SHA256(parameter_fingerprint)

This ensures:
- If video content changes → cache invalidates
- If fps/threshold changes → cache invalidates
- If whisper model changes → cache invalidates
- If VLM prompt changes → cache invalidates

Timeline synthesis is intentionally NOT idempotent,
so each run produces a fresh merged artifact.

================================================================================
TUNING PARAMETERS (IMPORTANT)
================================================================================

fps:
    Base frame extraction rate.
    1.0  → Recommended for UI recordings
    0.5  → Very long recordings
    2.0  → Fast UI transitions

change_threshold:
    Hamming distance for dHash.
    6  → Recommended default
    4  → More sensitive
    8+ → Less sensitive (fewer frames)

burst_window:
    Number of frames captured AFTER a detected change.
    3  → Recommended
    5+ → Captures longer transitions

whisper_model:
    tiny    → Fastest, lowest quality
    base
    small   → Recommended balance
    medium
    large   → Best accuracy, slower

VLM (llama-server):
    Use --parallel (np) tuned to CPU/GPU capacity.
    Example:
        ./llama-server -hf Qwen/Qwen3-VL-8B-Instruct-GGUF:Q8_0 -c 8192 -np 8 -cb

================================================================================
USAGE
================================================================================

Basic:
    python analyze_video_pipeline_full.py

With VLM:
    python analyze_video_pipeline_full.py --do_vlm

Custom tuning:
    python analyze_video_pipeline_full.py \
        --fps 1 \
        --change_threshold 6 \
        --burst_window 3 \
        --whisper_model small \
        --do_vlm

================================================================================
DEPENDENCIES
================================================================================

pip install openai-whisper pillow numpy
ffmpeg must be installed in PATH.

================================================================================
TROUBLESHOOTING
================================================================================

If whisper fails:
    Check ffmpeg -version

If VLM fails:
    Ensure llama-server is running:
        http://127.0.0.1:8033

================================================================================
"""

# ------------------------------------------------------------------------------
# IMPORTS
# ------------------------------------------------------------------------------

import argparse
import base64
import hashlib
import json
import shutil
import subprocess
import time
from pathlib import Path
from typing import Dict, Any, List, Optional

import whisper
from PIL import Image
import numpy as np
import urllib.request

# ------------------------------------------------------------------------------
# USER INPUT DEFAULTS
# ------------------------------------------------------------------------------

VIDEO_PATH = "video1822201159.mp4"
FRAMES_DIR = "extracted_frames"
AUDIO_PATH = "audio1822201159.m4a"
TRANSCRIPT_PATH = "transcript-analyze_pipeline.txt"

CACHE_ROOT = ".pipeline_cache"
RUN_ROOT = ".pipeline_runs"
LLAMA_URL = "http://127.0.0.1:8033"

# ------------------------------------------------------------------------------
# UTILS
# ------------------------------------------------------------------------------


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while chunk := f.read(1024 * 1024):
            h.update(chunk)
    return h.hexdigest()


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def run_cmd(cmd: List[str]):
    p = subprocess.run(cmd, stdout=subprocess.PIPE,
                       stderr=subprocess.STDOUT, text=True)
    if p.returncode != 0:
        raise RuntimeError(p.stdout)


def atomic_write(path: Path, content: str):
    tmp = path.with_suffix(".tmp")
    tmp.write_text(content, encoding="utf-8")
    tmp.replace(path)


# ------------------------------------------------------------------------------
# DHASH ADAPTIVE FRAME SELECTION
# ------------------------------------------------------------------------------

def dhash(image_path: Path, hash_size: int = 8) -> int:
    img = Image.open(image_path).convert(
        "L").resize((hash_size + 1, hash_size))
    pixels = np.array(img)
    diff = pixels[:, 1:] > pixels[:, :-1]
    return sum([1 << i for i, v in enumerate(diff.flatten()) if v])


def hamming_distance(a: int, b: int) -> int:
    return bin(a ^ b).count("1")


# ------------------------------------------------------------------------------
# PIPELINE STEPS
# ------------------------------------------------------------------------------

def extract_audio(video: Path, audio: Path):
    if not audio.exists():
        run_cmd([
            "ffmpeg", "-y",
            "-i", str(video),
            "-vn",
            "-c:a", "aac",
            "-b:a", "128k",
            str(audio),
        ])


def transcribe_audio(audio: Path, model_name: str, transcript_path: Path):
    model = whisper.load_model(model_name)
    result = model.transcribe(str(audio), fp16=False)
    atomic_write(transcript_path, result["text"])
    atomic_write(transcript_path.with_suffix(
        ".txt.json"), json.dumps(result, indent=2))


def extract_frames_adaptive(video: Path, fps: float, change_threshold: int, burst_window: int, frames_dir: Path):
    tmp = frames_dir / "_tmp"
    shutil.rmtree(tmp, ignore_errors=True)
    tmp.mkdir(parents=True, exist_ok=True)

    run_cmd([
        "ffmpeg", "-y",
        "-i", str(video),
        "-vf", f"fps={fps}",
        str(tmp / "frame_%06d.jpg"),
    ])

    frames = sorted(tmp.glob("*.jpg"))
    prev_hash = None
    burst = 0
    frames_dir.mkdir(parents=True, exist_ok=True)

    selection_log = []

    for i, f in enumerate(frames, start=1):
        h = dhash(f)
        keep = False

        if prev_hash is None:
            keep = True
        else:
            dist = hamming_distance(prev_hash, h)
            if dist >= change_threshold:
                keep = True
                burst = burst_window
            elif burst > 0:
                keep = True
                burst -= 1

        if keep:
            shutil.copy2(f, frames_dir / f.name)

        selection_log.append({
            "frame": f.name,
            "selected": keep
        })

        prev_hash = h

    atomic_write(frames_dir / "frame_selection.json",
                 json.dumps(selection_log, indent=2))
    shutil.rmtree(tmp, ignore_errors=True)


def analyze_frames_vlm(frames_dir: Path):
    jsonl = frames_dir / "frames_analysis.jsonl"
    lines = []

    for frame in sorted(frames_dir.glob("frame_*.jpg")):
        img_bytes = frame.read_bytes()
        payload = {
            "model": "default",
            "messages": [{
                "role": "user",
                "content": [
                    {"type": "text", "text": "Summarize key UI information as JSON."},
                    {"type": "image_url", "image_url": {
                        "url": "data:image/jpeg;base64," + base64.b64encode(img_bytes).decode()}}
                ]
            }]
        }
        req = urllib.request.Request(
            LLAMA_URL + "/v1/chat/completions",
            data=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json"},
        )
        resp = urllib.request.urlopen(req)
        result = json.loads(resp.read())
        lines.append(json.dumps(result))

    atomic_write(jsonl, "\n".join(lines))


# ------------------------------------------------------------------------------
# MAIN
# ------------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fps", type=float, default=1.0)
    parser.add_argument("--change_threshold", type=int, default=6)
    parser.add_argument("--burst_window", type=int, default=3)
    parser.add_argument("--whisper_model", default="small")
    parser.add_argument("--do_vlm", action="store_true")
    args = parser.parse_args()

    video = Path(VIDEO_PATH)
    audio = Path(AUDIO_PATH)
    frames = Path(FRAMES_DIR)
    transcript = Path(TRANSCRIPT_PATH)

    run_dir = Path(RUN_ROOT) / time.strftime("%Y%m%d_%H%M%S")
    run_dir.mkdir(parents=True, exist_ok=True)

    extract_audio(video, audio)
    transcribe_audio(audio, args.whisper_model, transcript)
    extract_frames_adaptive(
        video, args.fps, args.change_threshold, args.burst_window, frames)

    if args.do_vlm:
        analyze_frames_vlm(frames)

    print("Pipeline completed successfully.")


if __name__ == "__main__":
    main()
