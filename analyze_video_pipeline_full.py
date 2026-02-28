#!/usr/bin/env python3
"""
Multimodal AI Debugging Pipeline (FINAL)
=======================================

Purpose
-------
Process a screen recording (terminal/UI) and generate a debugging summary by combining:
1) Audio transcript (Whisper)
2) Visual terminal log (VLM via llama.cpp OpenAI-compatible endpoint)
3) Final synthesis (coordinator endpoint)

Hard Requirements Covered
-------------------------
- Binary-content driven (no reliance on filenames for caching decisions)
- Idempotent pipeline EXCEPT synthesize_timeline (always re-runs)
- Run logs with key metrics
- Works with VIDEO_PATH + AUDIO_PATH inputs (pre-extracted audio allowed)
- Optimized and scalable for long UI recordings:
  - pHash/dHash dedupe with adaptive spike + heartbeat
  - Global Frame CAS (across videos) to avoid redundant VLM calls
  - Coordinator/Worker true dynamic load balancing across cluster slots (URL:slots)
  - Network-hardened requests (timeouts + exponential backoff retries)
  - Stateful elapsed time tracking (survives Ctrl+C / restarts)

Dependencies
------------
pip install openai-whisper pillow imagehash requests
Optional:
pip install rich

Requires:
- ffmpeg in PATH
- llama.cpp server(s) exposing OpenAI-compatible endpoint:
  e.g. http://127.0.0.1:8033/v1/chat/completions

Example llama-server
--------------------
./llama-server -hf Qwen/Qwen3-VL-8B-Instruct-GGUF:Q8_0 -c 8192 -np 8 -cb --host 127.0.0.1 --port 8033

Key Concepts
------------
1) Content-Addressable Storage (CAS):
   - Video/audio/frame bytes are hashed using SHA-256.
   - Cache keys include parameter fingerprints (fps/threshold/hashmode/model/prompt_version).
   - Renaming files does not invalidate caches.

2) Global Frame CAS (cross-video):
   - Cache key includes:
     SHA256(frame_bytes + model_name + prompt_version + prompt_text_hash)
   - Prevents re-running VLM on identical frames across different videos.

3) Frame Dedup (hash-mode switch):
   - --hash-mode phash (default): robust under compression
   - --hash-mode dhash: faster and often better for UI

4) Frame selection logic:
   - Keep first frame
   - Keep if hash distance > threshold (change)
   - If adaptive spike: keep next burst_window frames after a change
   - Heartbeat: keep one frame every heartbeat_seconds even if static

CLI Usage Examples
------------------
1) Standard run (extract audio from video, VLM on frames, synthesize):
   python analyze_pipeline_final.py --video video.mp4 --do-vlm

2) Use pre-extracted audio file (skip audio extract):
   python analyze_pipeline_final.py --video video.mp4 --audio audio.m4a --do-vlm

3) Distributed load balancing with slots:
   python analyze_pipeline_final.py --video video.mp4 --do-vlm \
     --main-urls http://127.0.0.1:8033/v1/chat/completions:2 \
     --secondary-urls http://192.168.1.50:8033/v1/chat/completions:4 http://192.168.1.51:8033/v1/chat/completions:4

4) pHash tuning diagnostic (prints distances and would-keep stats):
   python analyze_pipeline_final.py --video video.mp4 --tune-hash --hash-mode phash --threshold 6 --fps 1 --tune-limit 120

5) Fast-scrolling terminal:
   python analyze_pipeline_final.py --video video.mp4 --do-vlm --fps 2 --threshold 5 --adaptive-spike --burst-window 5

Defaults (User Inputs)
----------------------
VIDEO_PATH = "video1822201159.mp4"
FRAMES_DIR = "extracted_frames"
AUDIO_PATH = "audio1822201159.m4a"
TRANSCRIPT_PATH = "transcript-analyze_pipeline.txt"
"""

from __future__ import annotations

import argparse
import base64
import concurrent.futures
import glob
import hashlib
import json
import logging
import os
import queue
import shutil
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import requests
import whisper
from PIL import Image

# imagehash provides phash/dhash implementations
import imagehash

# Optional Live UI
try:
    from rich.live import Live
    from rich.table import Table

    RICH_AVAILABLE = True
except Exception:
    RICH_AVAILABLE = False


# ------------------------- User defaults -------------------------

VIDEO_PATH = "video1822201159.mp4"
FRAMES_DIR = "extracted_frames"
AUDIO_PATH = "audio1822201159.m4a"
TRANSCRIPT_PATH = "transcript-analyze_pipeline.txt"

BASE_CACHE_DIR = ".pipeline_cache"
LOGS_DIR = "logs"
OUTPUT_DIR = "output"

LLAMA_API_URL = "http://127.0.0.1:8033/v1/chat/completions"

# ------------------------- Prompting -------------------------
# Bump PROMPT_VERSION whenever you change prompts in a meaningful way.
PROMPT_VERSION = "v3.terminal_ui_json"

FRAME_PROMPT = (
    "Analyze this screen capture (terminal/UI). Return STRICT JSON only.\n"
    "Schema:\n"
    "{\n"
    '  "screen_summary": "one sentence",\n'
    '  "key_text": ["important visible text lines"],\n'
    '  "commands_or_actions": ["commands typed or actions taken"],\n'
    '  "errors_warnings": ["errors/warnings if visible"],\n'
    '  "context": ["short bullets describing what is happening"],\n'
    '  "confidence": 0.0\n'
    "}\n"
    "Rules:\n"
    "- Do not invent text that is not visible.\n"
    "- If uncertain, leave lists empty and lower confidence.\n"
)

SYNTHESIS_SYSTEM = "You are an expert engineer and system debugger."

SYNTHESIS_TEMPLATE = """\
I have two timelines extracted from a screen recording of a technical debugging session.

1) AUDIO TRANSCRIPT (timestamped):
{transcript}

2) VISUAL TERMINAL LOG (deduplicated state changes):
{visual_log}

Please:
- Summarize the overall goal of the session.
- Align spoken context with what happened on screen.
- Identify errors/warnings/bottlenecks.
- Produce a detailed step-by-step walkthrough from start to end (replacement for watching the video).
- Add a short 'Next Actions' section at the end.
"""


# ------------------------- Helpers: hashing / fingerprints -------------------------

def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def json_fingerprint(obj: Dict[str, Any]) -> str:
    s = json.dumps(obj, sort_keys=True, separators=(
        ",", ":"), ensure_ascii=False)
    return sha256_bytes(s.encode("utf-8"))


def atomic_write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(content, encoding="utf-8")
    tmp.replace(path)


def atomic_write_json(path: Path, obj: Dict[str, Any]) -> None:
    atomic_write_text(path, json.dumps(
        obj, ensure_ascii=False, indent=2) + "\n")


# ------------------------- Helpers: time formatting -------------------------

def format_timestamp(seconds: float) -> str:
    seconds = max(0.0, float(seconds))
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    if h > 0:
        return f"{h:02d}:{m:02d}:{s:02d}"
    return f"{m:02d}:{s:02d}"


def format_duration(seconds: float) -> str:
    seconds = max(0.0, float(seconds))
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    if h > 0:
        return f"{h}h {m:02d}m {s:02d}s"
    return f"{m:02d}m {s:02d}s"


# ------------------------- Logging -------------------------

def setup_logger(run_id: str) -> logging.Logger:
    Path(LOGS_DIR).mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger(run_id)
    logger.setLevel(logging.DEBUG)
    logger.handlers.clear()
    logger.propagate = False

    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")

    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    ch.setFormatter(fmt)
    logger.addHandler(ch)

    fh = logging.FileHandler(
        Path(LOGS_DIR) / f"{run_id}.log", encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    return logger


# ------------------------- ffmpeg helpers -------------------------

def ensure_ffmpeg(logger: logging.Logger) -> None:
    try:
        subprocess.run(["ffmpeg", "-version"], stdout=subprocess.DEVNULL,
                       stderr=subprocess.DEVNULL, check=True)
    except Exception:
        logger.error("ffmpeg not found in PATH.")
        raise


def run_cmd(cmd: List[str], logger: logging.Logger, quiet: bool = True) -> None:
    p = subprocess.run(
        cmd,
        stdout=subprocess.DEVNULL if quiet else subprocess.PIPE,
        stderr=subprocess.DEVNULL if quiet else subprocess.STDOUT,
        text=True,
    )
    if p.returncode != 0:
        raise RuntimeError(f"Command failed ({p.returncode}): {' '.join(cmd)}")


# ------------------------- Cluster slots parsing -------------------------

def parse_url_slots(url_list: List[str], default_slots: int = 1) -> List[Tuple[str, int]]:
    """
    Accepts items like:
      http://host:8033/v1/chat/completions:4   (slots=4)
      http://host:8033/v1/chat/completions     (slots=default_slots)

    NOTE: This parser avoids breaking URLs by checking if suffix is all digits.
    """
    parsed: List[Tuple[str, int]] = []
    for item in url_list:
        parts = item.rsplit(":", 1)
        if len(parts) == 2 and parts[1].isdigit():
            parsed.append((parts[0], int(parts[1])))
        else:
            parsed.append((item, default_slots))
    return parsed


# ------------------------- CAS layout -------------------------

def cas_dir(cache_root: Path, step: str, input_hash: str, params_fp: str) -> Path:
    # Partition by first two chars for filesystem scaling
    return cache_root / step / input_hash[:2] / input_hash / params_fp


def cas_done_flag(dir_path: Path) -> Path:
    return dir_path / "DONE.json"


def cas_is_done(dir_path: Path) -> bool:
    return cas_done_flag(dir_path).exists()


def cas_mark_done(dir_path: Path, meta: Dict[str, Any]) -> None:
    atomic_write_json(cas_done_flag(dir_path), meta)


# ------------------------- Step 1: Audio extraction (idempotent) -------------------------

def extract_audio_idempotent(
    video_path: Path,
    audio_out: Path,
    cache_root: Path,
    logger: logging.Logger,
) -> Path:
    """
    Extract mono 16kHz audio into CAS.
    Materializes result to audio_out.
    """
    video_hash = sha256_file(video_path)
    params_fp = json_fingerprint(
        {"ar": 16000, "ac": 1, "codec": "copy?no", "format": audio_out.suffix.lstrip(".") or "m4a"})
    cas = cas_dir(cache_root, "audio_extract", video_hash, params_fp)
    cas.mkdir(parents=True, exist_ok=True)

    cached_audio = cas / f"audio{audio_out.suffix}"
    if cas_is_done(cas) and cached_audio.exists():
        logger.info("[audio] cache hit (video=%s)", video_hash[:12])
    else:
        logger.info("[audio] cache miss -> extracting with ffmpeg")
        run_cmd(
            ["ffmpeg", "-y", "-i", str(video_path), "-vn",
             "-ar", "16000", "-ac", "1", str(cached_audio)],
            logger,
            quiet=True,
        )
        cas_mark_done(cas, {"video_hash": video_hash,
                      "audio_hash": sha256_file(cached_audio)})

    audio_out.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(cached_audio, audio_out)
    return audio_out


# ------------------------- Step 2: Whisper transcription (idempotent) -------------------------

def transcribe_whisper_idempotent(
    audio_path: Path,
    transcript_out: Path,
    cache_root: Path,
    logger: logging.Logger,
    whisper_model: str,
) -> str:
    """
    Cache key includes:
      - SHA(audio bytes)
      - whisper model
      - anti-hallucination knobs
    Writes:
      transcript_out (timestamped lines)
      transcript_out.json (raw whisper output)
    """
    audio_hash = sha256_file(audio_path)
    whisper_params = {
        "whisper_model": whisper_model,
        "fp16": False,
        "condition_on_previous_text": False,  # anti-loop on silence
        # Extra knobs (safe defaults):
        "no_speech_threshold": 0.6,
        "logprob_threshold": -1.0,
    }
    params_fp = json_fingerprint(whisper_params)
    cas = cas_dir(cache_root, "whisper", audio_hash, params_fp)
    cas.mkdir(parents=True, exist_ok=True)

    cached_txt = cas / "transcript.txt"
    cached_json = cas / "transcript.json"

    if cas_is_done(cas) and cached_txt.exists() and cached_json.exists():
        logger.info("[asr] cache hit (audio=%s model=%s)",
                    audio_hash[:12], whisper_model)
        transcript = cached_txt.read_text(encoding="utf-8")
    else:
        logger.info(
            "[asr] cache miss -> transcribing (model=%s)", whisper_model)
        model = whisper.load_model(whisper_model)
        result = model.transcribe(
            str(audio_path),
            fp16=False,
            condition_on_previous_text=False,
            no_speech_threshold=whisper_params["no_speech_threshold"],
            logprob_threshold=whisper_params["logprob_threshold"],
        )

        lines: List[str] = []
        for seg in (result.get("segments") or []):
            text = (seg.get("text") or "").strip()
            if not text:
                continue
            st = float(seg.get("start", 0.0))
            en = float(seg.get("end", 0.0))
            lines.append(
                f"[{format_timestamp(st)} - {format_timestamp(en)}] {text}")

        transcript = ("\n".join(lines) + "\n") if lines else ""
        atomic_write_text(cached_txt, transcript)
        atomic_write_text(cached_json, json.dumps(
            result, ensure_ascii=False, indent=2) + "\n")
        cas_mark_done(cas, {"audio_hash": audio_hash, "segments": len(
            result.get("segments") or []), "params": whisper_params})

    transcript_out.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(cached_txt, transcript_out)
    shutil.copy2(cached_json, transcript_out.with_suffix(
        transcript_out.suffix + ".json"))
    return transcript


# ------------------------- Hash mode selection (pHash/dHash) -------------------------

def compute_imagehash(img: Image.Image, hash_mode: str) -> imagehash.ImageHash:
    if hash_mode == "phash":
        return imagehash.phash(img)
    if hash_mode == "dhash":
        return imagehash.dhash(img)
    raise ValueError(f"Unsupported hash_mode: {hash_mode}")


# ------------------------- Step 3: Frame extraction + selection (idempotent) -------------------------

def extract_frames_tmp(video_path: Path, tmp_dir: Path, fps: float, logger: logging.Logger) -> List[Path]:
    shutil.rmtree(tmp_dir, ignore_errors=True)
    tmp_dir.mkdir(parents=True, exist_ok=True)
    run_cmd(
        ["ffmpeg", "-y", "-i", str(video_path), "-vf",
         f"fps={fps}", str(tmp_dir / "frame_%06d.jpg")],
        logger,
        quiet=True,
    )
    return sorted(tmp_dir.glob("frame_*.jpg"))


def select_frames_by_hash(
    frames: List[Path],
    fps: float,
    hash_mode: str,
    threshold: int,
    adaptive_spike: bool,
    burst_window: int,
    heartbeat_seconds: int,
    logger: logging.Logger,
) -> Tuple[List[Path], List[Dict[str, Any]]]:
    selected: List[Path] = []
    trace: List[Dict[str, Any]] = []

    last_hash: Optional[imagehash.ImageHash] = None
    spike_counter = 0
    frames_since_last_sent = 0
    heartbeat_frames = max(1, int(round(heartbeat_seconds * fps)))

    for idx, frame_path in enumerate(frames, start=1):
        try:
            with Image.open(frame_path) as img:
                cur = compute_imagehash(img, hash_mode)
        except Exception as e:
            logger.debug("hash read error %s: %s", frame_path.name, e)
            continue

        dist = None if last_hash is None else int(cur - last_hash)

        keep = False
        reason = ""

        if last_hash is None:
            keep, reason = True, "first"
            last_hash = cur
            frames_since_last_sent = 0
        else:
            if dist is not None and dist > threshold:
                keep, reason = True, "change_spike"
                last_hash = cur
                frames_since_last_sent = 0
                if adaptive_spike:
                    spike_counter = burst_window
            elif adaptive_spike and spike_counter > 0:
                keep, reason = True, "spike_falloff"
                last_hash = cur
                spike_counter -= 1
                frames_since_last_sent = 0
            elif frames_since_last_sent >= heartbeat_frames:
                keep, reason = True, "heartbeat"
                last_hash = cur
                frames_since_last_sent = 0
            else:
                keep, reason = False, "too_similar"
                frames_since_last_sent += 1

        if keep:
            selected.append(frame_path)

        trace.append({
            "i": idx,
            "file": frame_path.name,
            "time_sec": (idx - 1) / fps,
            "distance": dist,
            "selected": keep,
            "reason": reason,
        })

    return selected, trace


def build_selected_frames_idempotent(
    video_path: Path,
    frames_dir_out: Path,
    cache_root: Path,
    logger: logging.Logger,
    fps: float,
    hash_mode: str,
    threshold: int,
    adaptive_spike: bool,
    burst_window: int,
    heartbeat_seconds: int,
) -> Tuple[List[Path], Path]:
    """
    Idempotent caching of:
      - selected frame bytes (stored by SHA)
      - selection trace JSON
    Materializes frames into frames_dir_out as frame_XXXXXX.jpg using original indices.
    """
    video_hash = sha256_file(video_path)
    params_fp = json_fingerprint({
        "fps": fps,
        "hash_mode": hash_mode,
        "threshold": threshold,
        "adaptive_spike": adaptive_spike,
        "burst_window": burst_window,
        "heartbeat_seconds": heartbeat_seconds,
    })

    cas = cas_dir(cache_root, "frames_select", video_hash, params_fp)
    cas.mkdir(parents=True, exist_ok=True)

    cached_frames_dir = cas / "frames_by_sha"
    cached_selection = cas / "frame_selection.json"

    if cas_is_done(cas) and cached_frames_dir.exists() and cached_selection.exists():
        logger.info("[frames] cache hit (video=%s)", video_hash[:12])
    else:
        logger.info(
            "[frames] cache miss -> extracting and selecting (%s)", hash_mode)
        tmp = cas / "_tmp_raw_frames"
        raw = extract_frames_tmp(video_path, tmp, fps, logger)
        selected_raw, trace = select_frames_by_hash(
            raw,
            fps=fps,
            hash_mode=hash_mode,
            threshold=threshold,
            adaptive_spike=adaptive_spike,
            burst_window=burst_window,
            heartbeat_seconds=heartbeat_seconds,
            logger=logger,
        )

        shutil.rmtree(cached_frames_dir, ignore_errors=True)
        cached_frames_dir.mkdir(parents=True, exist_ok=True)

        # Attach SHA256 to selected frames (binary-content anchored)
        for e in trace:
            if not e["selected"]:
                continue
            src = tmp / e["file"]
            b = src.read_bytes()
            sha = sha256_bytes(b)
            e["frame_sha256"] = sha
            dst = cached_frames_dir / f"{sha}.jpg"
            if not dst.exists():
                dst.write_bytes(b)

        selection_doc = {
            "video_hash": video_hash,
            "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "params": {
                "fps": fps,
                "hash_mode": hash_mode,
                "threshold": threshold,
                "adaptive_spike": adaptive_spike,
                "burst_window": burst_window,
                "heartbeat_seconds": heartbeat_seconds,
            },
            "raw_frames": len(raw),
            "selected_frames": len(selected_raw),
            "trace": trace,
        }
        atomic_write_json(cached_selection, selection_doc)
        cas_mark_done(cas, {"video_hash": video_hash, "raw_frames": len(
            raw), "selected_frames": len(selected_raw)})

        shutil.rmtree(tmp, ignore_errors=True)

    # Materialize into frames_dir_out
    frames_dir_out.mkdir(parents=True, exist_ok=True)
    for p in frames_dir_out.glob("frame_*.jpg"):
        p.unlink(missing_ok=True)

    sel = json.loads(cached_selection.read_text(encoding="utf-8"))

    for e in sel["trace"]:
        if not e.get("selected"):
            continue
        i = int(e["i"])
        sha = e["frame_sha256"]
        src = cached_frames_dir / f"{sha}.jpg"
        dst = frames_dir_out / f"frame_{i:06d}.jpg"
        shutil.copy2(src, dst)

    # Persist selection metadata for inspection
    shutil.copy2(cached_selection, frames_dir_out / "frame_selection.json")

    selected_paths = sorted(frames_dir_out.glob("frame_*.jpg"))
    return selected_paths, frames_dir_out / "frame_selection.json"


# ------------------------- Tuning diagnostic -------------------------

def tune_hash_diagnostic(
    video_path: Path,
    fps: float,
    hash_mode: str,
    threshold: int,
    adaptive_spike: bool,
    burst_window: int,
    heartbeat_seconds: int,
    limit: int,
    logger: logging.Logger,
) -> None:
    tmp = Path(".hash_tune_tmp")
    raw = extract_frames_tmp(video_path, tmp, fps, logger)
    if limit > 0:
        raw = raw[:limit]

    selected, trace = select_frames_by_hash(
        raw,
        fps=fps,
        hash_mode=hash_mode,
        threshold=threshold,
        adaptive_spike=adaptive_spike,
        burst_window=burst_window,
        heartbeat_seconds=heartbeat_seconds,
        logger=logger,
    )

    print("=" * 72)
    print(f"{hash_mode} tuning diagnostic | fps={fps} threshold={threshold} adaptive_spike={adaptive_spike} burst={burst_window} heartbeat_s={heartbeat_seconds}")
    print("-" * 72)
    print(f"{'frame':<14} {'dist':<6} {'selected':<9} reason")
    print("-" * 72)
    for e in trace:
        dist = "N/A" if e["distance"] is None else str(e["distance"])
        print(f"{e['file']:<14} {dist:<6} {str(e['selected']):<9} {e['reason']}")
    print("-" * 72)
    kept = len(selected)
    total = len(raw)
    comp = (1 - (kept / max(1, total))) * 100
    print(f"kept={kept}/{total} compression={comp:.1f}%")
    print("=" * 72)

    shutil.rmtree(tmp, ignore_errors=True)


# ------------------------- Robust API call with telemetry -------------------------

def robust_api_call(
    url: str,
    payload: Dict[str, Any],
    timeout_s: int,
    max_retries: int,
    backoff_base_s: float,
) -> Tuple[str, Dict[str, Any]]:
    last_err: Optional[Exception] = None
    for attempt in range(max(1, max_retries)):
        try:
            start = time.time()
            r = requests.post(url, json=payload, headers={
                              "Content-Type": "application/json"}, timeout=timeout_s)
            r.raise_for_status()
            duration = max(1e-6, time.time() - start)

            data = r.json()
            content = data["choices"][0]["message"]["content"]

            usage = data.get("usage", {}) or {}
            prompt_tokens = int(usage.get("prompt_tokens", 0) or 0)
            completion_tokens = int(usage.get("completion_tokens", 0) or 0)

            tps = (completion_tokens / duration) if duration > 0 else 0.0
            stats = {
                "duration_s": duration,
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "tps": tps,
            }
            return content, stats
        except Exception as e:
            last_err = e
            if attempt == max_retries - 1:
                raise
            sleep_s = backoff_base_s * (2 ** attempt)
            time.sleep(sleep_s)
    raise RuntimeError(str(last_err))


# ------------------------- Global Frame CAS (cross-video) -------------------------

def global_vlm_cache_dir(cache_root: Path) -> Path:
    d = cache_root / "global_vlm_cache"
    d.mkdir(parents=True, exist_ok=True)
    return d


def global_frame_key(frame_bytes: bytes, model_name: str) -> str:
    # Include prompt version + prompt hash to avoid stale reuse when prompt changes
    prompt_hash = sha256_bytes(FRAME_PROMPT.encode("utf-8"))
    key_material = frame_bytes + \
        model_name.encode(
            "utf-8") + PROMPT_VERSION.encode("utf-8") + prompt_hash.encode("utf-8")
    return sha256_bytes(key_material)


# ------------------------- Distributed VLM processing with slot queue -------------------------

@dataclass(frozen=True)
class Slot:
    url: str
    slot_id: int  # 1..N


def build_slots(cluster_nodes: List[Tuple[str, int]]) -> List[Slot]:
    slots: List[Slot] = []
    for url, n in cluster_nodes:
        for i in range(1, max(1, n) + 1):
            slots.append(Slot(url=url, slot_id=i))
    return slots


def make_live_table(active_status: Dict[str, str], completed_log: List[str], total_rows: int, lock: threading.Lock) -> "Table":
    table = Table(show_header=True, header_style="bold cyan")
    table.add_column("✅ Completed", style="dim green", width=60)
    table.add_column("⏳ Active", style="yellow", width=60)

    with lock:
        comp = completed_log[-total_rows:]
        comp = [""] * (total_rows - len(comp)) + comp
        act = list(active_status.values())
        act = act + [""] * (total_rows - len(act))

    for c, a in zip(comp, act):
        table.add_row(c, a)
    return table


def analyze_frames_vlm_distributed(
    selected_frames: List[Path],
    fps: float,
    cache_root: Path,
    logger: logging.Logger,
    model_name: str,
    cluster_nodes: List[Tuple[str, int]],
    max_tokens: int,
    timeout_s: int,
    retries: int,
    backoff_base_s: float,
    use_rich_ui: bool,
    elapsed_time_path: Path,
) -> Tuple[List[str], Dict[str, Any]]:
    """
    Returns:
      visual_log lines: "[MM:SS] Visual State: ..."
      perf summary dict (per server, global cache hits, elapsed)
    """
    slots = build_slots(cluster_nodes)
    total_slots = len(slots)
    if total_slots <= 0:
        raise ValueError("No cluster slots provided.")

    url_pool: "queue.Queue[Slot]" = queue.Queue()
    for s in slots:
        url_pool.put(s)

    # Active status keyed by slot label
    active_status: Dict[str, str] = {
        f"{s.url} (Slot {s.slot_id})": "Idle" for s in slots}
    completed_log: List[str] = []
    lock = threading.Lock()

    # Stats
    cluster_stats: Dict[str, Dict[str, Any]] = {}
    global_cache_hits = 0
    completed = 0

    # Stateful elapsed time recovery
    prev_elapsed = 0.0
    if elapsed_time_path.exists():
        try:
            prev_elapsed = float(elapsed_time_path.read_text(
                encoding="utf-8").strip() or "0")
        except Exception:
            prev_elapsed = 0.0
    session_start = time.time()

    gcache = global_vlm_cache_dir(cache_root)

    def update_elapsed() -> float:
        total_elapsed = prev_elapsed + (time.time() - session_start)
        atomic_write_text(elapsed_time_path, str(total_elapsed))
        return total_elapsed

    def frame_timestamp_from_name(frame_path: Path) -> str:
        # frame_000123.jpg -> index 123
        try:
            n = int(frame_path.stem.split("_")[1])
        except Exception:
            n = 1
        tsec = (n - 1) / fps
        return format_timestamp(tsec)

    def process_one(task: Tuple[int, Path]) -> str:
        nonlocal global_cache_hits, completed

        idx, frame_path = task
        fname = frame_path.name
        ts = frame_timestamp_from_name(frame_path)

        frame_bytes = frame_path.read_bytes()
        gkey = global_frame_key(frame_bytes, model_name)
        gpath = gcache / f"{gkey}.txt"

        # Global cache hit
        if gpath.exists():
            analysis = gpath.read_text(encoding="utf-8").strip()
            with lock:
                global_cache_hits += 1
                completed += 1
                elapsed = update_elapsed()
                completed_log.append(
                    f"{fname} -> GLOBAL CACHE HIT | Elapsed: {format_duration(elapsed)}")
            logger.info("♻️  [%03d/%03d] %s | GLOBAL CACHE HIT | Elapsed: %s",
                        completed, len(selected_frames), fname, format_duration(elapsed))
            return f"[{ts}] Visual State: {analysis}"

        # Acquire slot (dynamic load balancing)
        slot = url_pool.get()
        slot_label = f"{slot.url} (Slot {slot.slot_id})"
        server_short = slot.url.split("//")[-1].split("/")[0]

        with lock:
            active_status[slot_label] = f"{fname} ({idx}/{len(selected_frames)})"

        payload = {
            "model": model_name,
            "messages": [{
                "role": "user",
                "content": [
                    {"type": "text", "text": FRAME_PROMPT},
                    {"type": "image_url", "image_url": {
                        "url": "data:image/jpeg;base64," + base64.b64encode(frame_bytes).decode("utf-8")}},
                ],
            }],
            "temperature": 0.1,
            "max_tokens": max_tokens,
        }

        try:
            analysis, stats = robust_api_call(
                slot.url, payload, timeout_s=timeout_s, max_retries=retries, backoff_base_s=backoff_base_s
            )
            analysis = (analysis or "").strip()
            atomic_write_text(gpath, analysis + "\n")

            with lock:
                completed += 1
                elapsed = update_elapsed()

                if server_short not in cluster_stats:
                    cluster_stats[server_short] = {
                        "frames": 0, "completion_tokens": 0, "time_s": 0.0}
                cluster_stats[server_short]["frames"] += 1
                cluster_stats[server_short]["completion_tokens"] += int(
                    stats.get("completion_tokens", 0))
                cluster_stats[server_short]["time_s"] += float(
                    stats.get("duration_s", 0.0))

                active_status[slot_label] = "Idle"
                completed_log.append(
                    f"{fname} -> {server_short} ({stats.get('tps', 0.0):.1f} t/s) | Elapsed: {format_duration(elapsed)}")

            logger.info("✅ [%03d/%03d] %s | Node: %-18s | Speed: %5.1f t/s | Time: %4.1fs | Elapsed: %s",
                        completed, len(selected_frames), fname, server_short, stats.get(
                            "tps", 0.0),
                        stats.get("duration_s", 0.0), format_duration(elapsed))
            return f"[{ts}] Visual State: {analysis}"

        except Exception as e:
            with lock:
                active_status[slot_label] = "Idle"
                completed_log.append(f"{fname} -> {server_short} (FAILED)")
            logger.error("Frame failed: %s | %s", fname, str(e))
            return f"[{ts}] Visual State: Error processing frame."

        finally:
            url_pool.put(slot)

    tasks = [(i, f) for i, f in enumerate(selected_frames, start=1)]
    visual_log: List[str] = []

    # Rich UI
    use_live = use_rich_ui and RICH_AVAILABLE
    live = None
    if use_live:
        live = Live(make_live_table(active_status, completed_log,
                    total_rows=total_slots, lock=lock), refresh_per_second=10)
        live.__enter__()

    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=total_slots) as ex:
            for res in ex.map(process_one, tasks):
                visual_log.append(res)
                if use_live and live:
                    live.update(make_live_table(
                        active_status, completed_log, total_rows=total_slots, lock=lock))
    finally:
        if use_live and live:
            live.__exit__(None, None, None)

    # Summarize stats
    summary: Dict[str, Any] = {
        "cluster_stats": {},
        "global_cache_hits": global_cache_hits,
        "total_frames": len(selected_frames),
        "elapsed_s": prev_elapsed + (time.time() - session_start),
    }
    for server, data in cluster_stats.items():
        avg_tps = (data["completion_tokens"] / data["time_s"]
                   ) if data["time_s"] > 0 else 0.0
        summary["cluster_stats"][server] = {
            "frames": data["frames"],
            "completion_tokens": data["completion_tokens"],
            "time_s": data["time_s"],
            "avg_tps": avg_tps,
        }

    return visual_log, summary


# ------------------------- Synthesis (NOT idempotent) -------------------------

def synthesize_timeline(
    transcript: str,
    visual_log: List[str],
    synthesis_url: str,
    model_name: str,
    logger: logging.Logger,
    timeout_s: int,
    retries: int,
    backoff_base_s: float,
) -> Tuple[str, Dict[str, Any]]:
    # Cheap text compression by unique states
    compressed: List[str] = []
    last_state: Optional[str] = None
    for line in visual_log:
        state = line
        if state != last_state:
            compressed.append(line)
            last_state = state

    prompt = SYNTHESIS_TEMPLATE.format(
        transcript=transcript if transcript else "None",
        visual_log="\n".join(compressed) if compressed else "None",
    )

    payload = {
        "model": model_name,
        "messages": [
            {"role": "system", "content": [
                {"type": "text", "text": SYNTHESIS_SYSTEM}]},
            {"role": "user", "content": [{"type": "text", "text": prompt}]},
        ],
        "temperature": 0.3,
        "max_tokens": 2048,
    }

    logger.info(
        "[synth] coordinator=%s | sending compressed timelines", synthesis_url)
    analysis, stats = robust_api_call(
        synthesis_url, payload, timeout_s=timeout_s, max_retries=retries, backoff_base_s=backoff_base_s
    )
    return (analysis or "").strip(), stats


# ------------------------- Main -------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Multimodal Debugging Pipeline (FINAL)")

    parser.add_argument("--video", default=VIDEO_PATH, help="Input video path")
    parser.add_argument("--audio", default=AUDIO_PATH,
                        help="Audio path. If file exists, it will be used; else extracted from video.")
    parser.add_argument("--frames-dir", default=FRAMES_DIR,
                        help="Materialized selected frames directory")
    parser.add_argument("--transcript-out",
                        default=TRANSCRIPT_PATH, help="Transcript output path")

    parser.add_argument("--cache-root", default=BASE_CACHE_DIR,
                        help="Cache root directory")
    parser.add_argument("--output-dir", default=OUTPUT_DIR,
                        help="Output directory for summary + artifacts")

    # Cluster endpoints with slot support
    parser.add_argument("--main-urls", nargs="+", default=[LLAMA_API_URL],
                        help="Primary endpoint(s). FIRST is used for synthesis. Supports URL:slots.")
    parser.add_argument("--secondary-urls", nargs="*", default=[],
                        help="Worker endpoint(s) for frames. Supports URL:slots.")

    parser.add_argument("--model", default="qwen3-vl",
                        help="Model name expected by the endpoint")
    parser.add_argument("--whisper-model", default="base",
                        help="Whisper model: tiny|base|small|medium|large")

    # Frame selection tuning
    parser.add_argument("--fps", type=float, default=1.0)
    parser.add_argument(
        "--hash-mode", choices=["phash", "dhash"], default="phash")
    parser.add_argument("--threshold", type=int, default=5)
    parser.add_argument("--adaptive-spike", action="store_true")
    parser.add_argument("--burst-window", type=int, default=3)
    parser.add_argument("--heartbeat-seconds", type=int, default=30)

    # Modes
    parser.add_argument("--tune-hash", action="store_true",
                        help="Run hash tuning diagnostic and exit")
    parser.add_argument("--tune-limit", type=int, default=120)
    parser.add_argument("--audio-only", action="store_true")
    parser.add_argument("--do-vlm", action="store_true",
                        help="Enable VLM calls for frames")
    parser.add_argument("--clear-cache", action="store_true")

    # Network / inference controls
    parser.add_argument("--timeout", type=int, default=180)
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--backoff-base", type=float, default=1.0)
    parser.add_argument("--max-tokens-frame", type=int, default=150)

    # UI
    parser.add_argument("--no-rich", action="store_true")

    args = parser.parse_args()

    cache_root = Path(args.cache_root).resolve()
    out_root = Path(args.output_dir).resolve()
    frames_dir_out = Path(args.frames_dir).resolve()

    video_path = Path(args.video).resolve()
    audio_path = Path(args.audio).resolve()
    transcript_out = Path(args.transcript_out).resolve()

    out_root.mkdir(parents=True, exist_ok=True)
    cache_root.mkdir(parents=True, exist_ok=True)

    run_video_hash = sha256_file(
        video_path) if video_path.exists() else "no_video"
    run_id = f"pipeline_{video_path.stem}_{run_video_hash[:8]}_{time.strftime('%Y%m%d_%H%M%S')}"
    logger = setup_logger(run_id)

    ensure_ffmpeg(logger)

    if not video_path.exists():
        logger.error("Video not found: %s", str(video_path))
        sys.exit(1)

    logger.info("START | video=%s | hash=%s", str(video_path), run_video_hash)
    logger.info("Config | fps=%.2f hash=%s threshold=%d spike=%s burst=%d heartbeat=%ds",
                args.fps, args.hash_mode, args.threshold, args.adaptive_spike, args.burst_window, args.heartbeat_seconds)

    if args.clear_cache:
        logger.info("Clearing cache root: %s", str(cache_root))
        shutil.rmtree(cache_root, ignore_errors=True)
        cache_root.mkdir(parents=True, exist_ok=True)

    # Cluster parse
    main_nodes = parse_url_slots(args.main_urls, default_slots=1)
    sec_nodes = parse_url_slots(args.secondary_urls, default_slots=1)
    cluster_nodes = main_nodes + sec_nodes
    synthesis_url = main_nodes[0][0]

    total_slots = sum(s for _, s in cluster_nodes)
    logger.info("Cluster | nodes=%d total_slots=%d | synthesis=%s",
                len(cluster_nodes), total_slots, synthesis_url)

    # Tune-only mode
    if args.tune_hash:
        tune_hash_diagnostic(
            video_path=video_path,
            fps=args.fps,
            hash_mode=args.hash_mode,
            threshold=args.threshold,
            adaptive_spike=args.adaptive_spike,
            burst_window=args.burst_window,
            heartbeat_seconds=args.heartbeat_seconds,
            limit=args.tune_limit,
            logger=logger,
        )
        return

    # Audio: if provided audio exists use it; else extract from video (idempotent)
    if audio_path.exists() and audio_path.stat().st_size > 0:
        logger.info("[audio] using existing audio=%s", str(audio_path))
    else:
        logger.info("[audio] audio missing -> extracting to %s",
                    str(audio_path))
        extract_audio_idempotent(video_path, audio_path, cache_root, logger)

    # Transcript (idempotent)
    transcript = transcribe_whisper_idempotent(
        audio_path, transcript_out, cache_root, logger, whisper_model=args.whisper_model)

    # Selected frames (idempotent)
    selected_frames, selection_json = build_selected_frames_idempotent(
        video_path=video_path,
        frames_dir_out=frames_dir_out,
        cache_root=cache_root,
        logger=logger,
        fps=args.fps,
        hash_mode=args.hash_mode,
        threshold=args.threshold,
        adaptive_spike=args.adaptive_spike,
        burst_window=args.burst_window,
        heartbeat_seconds=args.heartbeat_seconds,
    )

    logger.info("[frames] selected=%d (materialized at %s)",
                len(selected_frames), str(frames_dir_out))

    # VLM visual log
    visual_log: List[str] = []
    perf_summary: Dict[str, Any] = {}

    # Stateful elapsed time file path is parameter-specific to avoid collisions
    elapsed_fp = json_fingerprint({
        "fps": args.fps,
        "hash_mode": args.hash_mode,
        "threshold": args.threshold,
        "adaptive_spike": args.adaptive_spike,
        "burst_window": args.burst_window,
        "heartbeat_seconds": args.heartbeat_seconds,
        "model": args.model,
        "prompt_version": PROMPT_VERSION,
    })
    elapsed_time_path = cas_dir(
        cache_root, "elapsed_time", run_video_hash, elapsed_fp) / "elapsed_time.txt"
    elapsed_time_path.parent.mkdir(parents=True, exist_ok=True)

    if args.audio_only:
        logger.info("MODE: audio-only (skipping VLM frames)")
    elif args.do_vlm:
        visual_log, perf_summary = analyze_frames_vlm_distributed(
            selected_frames=selected_frames,
            fps=args.fps,
            cache_root=cache_root,
            logger=logger,
            model_name=args.model,
            cluster_nodes=cluster_nodes,
            max_tokens=args.max_tokens_frame,
            timeout_s=args.timeout,
            retries=args.retries,
            backoff_base_s=args.backoff_base,
            use_rich_ui=(not args.no_rich),
            elapsed_time_path=elapsed_time_path,
        )
        atomic_write_text(frames_dir_out / "visual_log.txt",
                          "\n".join(visual_log) + ("\n" if visual_log else ""))
        atomic_write_json(
            frames_dir_out / "cluster_perf_summary.json", perf_summary)
        logger.info("[vlm] done | global_cache_hits=%d | elapsed=%s",
                    perf_summary.get("global_cache_hits", 0), format_duration(perf_summary.get("elapsed_s", 0.0)))
    else:
        logger.info("VLM disabled. Use --do-vlm to enable visual analysis.")

    # Synthesis (NOT idempotent)
    final_analysis, synth_stats = synthesize_timeline(
        transcript=transcript,
        visual_log=visual_log,
        synthesis_url=synthesis_url,
        model_name=args.model,
        logger=logger,
        timeout_s=args.timeout,
        retries=args.retries,
        backoff_base_s=args.backoff_base,
    )

    # Output organization
    video_basename = video_path.stem
    video_hash8 = run_video_hash[:8]
    out_dir = out_root / video_basename
    out_dir.mkdir(parents=True, exist_ok=True)

    # Save final artifacts
    summary_name = f"debug_summary_{video_hash8}_fps{args.fps}_hm{args.hash_mode}_t{args.threshold}_sp{int(args.adaptive_spike)}_{args.model}.md"
    summary_path = out_dir / summary_name
    atomic_write_text(summary_path, final_analysis + "\n")

    # Copy important artifacts for tracking
    if transcript_out.exists():
        shutil.copy2(transcript_out, out_dir / transcript_out.name)
    tj = transcript_out.with_suffix(transcript_out.suffix + ".json")
    if tj.exists():
        shutil.copy2(tj, out_dir / tj.name)

    if selection_json.exists():
        shutil.copy2(selection_json, out_dir / "frame_selection.json")

    if (frames_dir_out / "visual_log.txt").exists():
        shutil.copy2(frames_dir_out / "visual_log.txt",
                     out_dir / "visual_log.txt")

    if (frames_dir_out / "cluster_perf_summary.json").exists():
        shutil.copy2(frames_dir_out / "cluster_perf_summary.json",
                     out_dir / "cluster_perf_summary.json")

    # Record run metadata
    run_meta = {
        "run_id": run_id,
        "video": str(video_path),
        "video_hash": run_video_hash,
        "audio": str(audio_path),
        "model": args.model,
        "whisper_model": args.whisper_model,
        "frame_params": {
            "fps": args.fps,
            "hash_mode": args.hash_mode,
            "threshold": args.threshold,
            "adaptive_spike": args.adaptive_spike,
            "burst_window": args.burst_window,
            "heartbeat_seconds": args.heartbeat_seconds,
        },
        "cluster": {
            "main": main_nodes,
            "secondary": sec_nodes,
            "synthesis_url": synthesis_url,
            "total_slots": total_slots,
        },
        "prompt_version": PROMPT_VERSION,
        "synthesis_stats": synth_stats,
        "perf_summary": perf_summary,
        "outputs": {
            "summary": str(summary_path),
            "output_dir": str(out_dir),
        },
    }
    atomic_write_json(out_dir / "run_metadata.json", run_meta)

    logger.info("DONE | summary=%s", str(summary_path))
    logger.info("Synthesis | speed=%.1f t/s | prompt=%d | gen=%d | time=%.1fs",
                synth_stats.get("tps", 0.0), synth_stats.get(
                    "prompt_tokens", 0),
                synth_stats.get("completion_tokens", 0), synth_stats.get("duration_s", 0.0))

    print("\n" + "=" * 72)
    print("FINAL MULTIMODAL DEBUGGING SUMMARY")
    print("=" * 72)
    print(final_analysis)
    print("=" * 72)


if __name__ == "__main__":
    main()
