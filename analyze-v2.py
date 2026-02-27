#!/usr/bin/env python3
"""
analyze_video_pipeline_idempotent.py

Defaults (your inputs):
  VIDEO_PATH      = "video1822201159.mp4"
  FRAMES_DIR      = "extracted_frames"
  AUDIO_PATH      = "audio1822201159.m4a"
  TRANSCRIPT_PATH = "transcript-analyze_pipeline.txt"

Key properties:
- Idempotent steps based on BINARY content hashes + parameter fingerprints.
- Caches are keyed by SHA-256(video bytes), SHA-256(audio bytes), SHA-256(frame bytes),
  and fingerprints of parameters (fps, whisper model, prompt, etc.).
- synthesize_timeline is NOT idempotent by design (always writes fresh output).

Requirements:
- ffmpeg in PATH
- pip install openai-whisper
- (optional) llama-server running at http://127.0.0.1:8033

Run:
  python3 analyze_video_pipeline_idempotent.py --fps 1 --whisper_model small
  python3 analyze_video_pipeline_idempotent.py --fps 1 --do_vlm --frame_limit 50
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import logging
import os
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import urllib.request

import whisper  # pip openai-whisper

# ------------------------- Your defaults -------------------------

VIDEO_PATH = "video1822201159.mp4"
FRAMES_DIR = "extracted_frames"
AUDIO_PATH = "audio1822201159.m4a"
TRANSCRIPT_PATH = "transcript-analyze_pipeline.txt"

LLAMA_URL_DEFAULT = "http://127.0.0.1:8033"

# Cache and logs
CACHE_ROOT_DEFAULT = ".pipeline_cache"     # caches are based on binary content
RUNS_ROOT_DEFAULT = ".pipeline_runs"       # per-run outputs and logs

# ------------------------- Prompts -------------------------

FRAME_PROMPT = """You are analyzing a screen-recording frame (a UI screenshot).
Extract ONLY high-signal information. Return strict JSON with this schema:

{
  "screen_summary": "one sentence",
  "key_text": ["important text strings you can read"],
  "entities": {
    "urls": [],
    "job_ids": [],
    "pipeline_ids": [],
    "commit_ids": [],
    "cluster_or_env": [],
    "errors": []
  },
  "observations": ["bullets of what is happening"],
  "suggested_next_questions": ["what to look for next in subsequent frames or logs"],
  "confidence": 0.0
}

Rules:
- Keep key_text short and relevant (titles, errors, ids, durations).
- Put anything error-like in entities.errors.
- If you cannot read a field, omit it or leave empty; do not guess.
- confidence: 0.0 to 1.0.
"""

# ------------------------- Utilities -------------------------

def now_ts() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


def run_cmd(cmd: List[str], logger: logging.Logger) -> None:
    p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    if p.returncode != 0:
        logger.error("Command failed (%s): %s", p.returncode, " ".join(cmd))
        logger.error("Output:\n%s", p.stdout)
        raise RuntimeError(f"Command failed ({p.returncode}): {' '.join(cmd)}\n\n{p.stdout}")
    logger.debug("Command ok: %s", " ".join(cmd))


def ensure_ffmpeg(logger: logging.Logger) -> None:
    try:
        run_cmd(["ffmpeg", "-version"], logger)
    except Exception as e:
        raise RuntimeError("ffmpeg not found in PATH.") from e


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            b = f.read(chunk_size)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def fingerprint_dict(d: Dict[str, Any]) -> str:
    # Stable fingerprint: JSON canonicalization
    s = json.dumps(d, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def atomic_write_text(path: Path, text: str, encoding: str = "utf-8") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text, encoding=encoding)
    tmp.replace(path)


def atomic_write_json(path: Path, obj: Any) -> None:
    atomic_write_text(path, json.dumps(obj, ensure_ascii=False, indent=2) + "\n")


def http_post_json(url: str, payload: Dict[str, Any], timeout_s: int = 180) -> Dict[str, Any]:
    req = urllib.request.Request(
        url=url,
        method="POST",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=timeout_s) as resp:
        return json.loads(resp.read().decode("utf-8"))


def b64_image(path: Path) -> str:
    return base64.b64encode(path.read_bytes()).decode("utf-8")


# ------------------------- Logging setup -------------------------

def setup_logger(run_dir: Path) -> logging.Logger:
    run_dir.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger(f"pipeline_{run_dir.name}")
    logger.setLevel(logging.DEBUG)
    logger.handlers.clear()
    logger.propagate = False

    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")

    # Console handler (INFO)
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    ch.setFormatter(fmt)
    logger.addHandler(ch)

    # File handler (DEBUG)
    fh = logging.FileHandler(run_dir / "run.log", encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    return logger


# ------------------------- Cache model -------------------------

@dataclass(frozen=True)
class StepKey:
    step: str
    input_hash: str
    params_fp: str

    def path(self, cache_root: Path) -> Path:
        # Partition by step and first 2 chars of hash for filesystem scalability
        return cache_root / self.step / self.input_hash[:2] / self.input_hash / self.params_fp


def cache_hit(step_key: StepKey, cache_root: Path) -> bool:
    return (step_key.path(cache_root) / "DONE.json").exists()


def cache_mark_done(step_key: StepKey, cache_root: Path, meta: Dict[str, Any]) -> None:
    p = step_key.path(cache_root)
    p.mkdir(parents=True, exist_ok=True)
    atomic_write_json(p / "DONE.json", meta)


def cache_read_meta(step_key: StepKey, cache_root: Path) -> Dict[str, Any]:
    p = step_key.path(cache_root) / "DONE.json"
    return json.loads(p.read_text(encoding="utf-8"))


# ------------------------- Steps (idempotent) -------------------------

def step_extract_audio_m4a(
    video: Path,
    audio_out: Path,
    cache_root: Path,
    logger: logging.Logger,
) -> Tuple[Path, str]:
    """
    Idempotent based on video bytes hash.
    If audio_out exists but is not tied to the same video hash, this step will still
    prefer cache output (binary-based), not filename-based.
    """
    t0 = time.time()
    video_hash = sha256_file(video)
    params_fp = fingerprint_dict({"codec": "aac", "bitrate": "128k", "container": "m4a"})
    key = StepKey("extract_audio_m4a", video_hash, params_fp)
    out_dir = key.path(cache_root)

    cached_audio = out_dir / "audio.m4a"
    meta_path = out_dir / "meta.json"

    if cache_hit(key, cache_root) and cached_audio.exists():
        meta = cache_read_meta(key, cache_root)
        logger.info("[audio] cache hit | video=%s | audio=%s", video_hash[:12], meta.get("audio_hash", "")[:12])
    else:
        logger.info("[audio] extracting from video (idempotent) ...")
        out_dir.mkdir(parents=True, exist_ok=True)
        # Write directly into cache location
        run_cmd([
            "ffmpeg", "-y",
            "-i", str(video),
            "-vn",
            "-c:a", "aac",
            "-b:a", "128k",
            str(cached_audio),
        ], logger)
        audio_hash = sha256_file(cached_audio)
        meta = {
            "step": "extract_audio_m4a",
            "created_at": now_ts(),
            "video_hash": video_hash,
            "audio_hash": audio_hash,
            "params": {"codec": "aac", "bitrate": "128k", "container": "m4a"},
        }
        atomic_write_json(meta_path, meta)
        cache_mark_done(key, cache_root, meta)
        logger.info("[audio] extracted | audio_hash=%s | took=%.2fs", audio_hash[:12], time.time() - t0)

    # Materialize to requested output path (copy is fast and keeps idempotence)
    audio_out.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(cached_audio, audio_out)

    # Return audio path + audio hash
    audio_hash = sha256_file(audio_out)
    return audio_out, audio_hash


def step_transcribe_whisper(
    audio: Path,
    whisper_model: str,
    transcript_txt_out: Path,
    cache_root: Path,
    logger: logging.Logger,
) -> Tuple[Path, Path, str]:
    """
    Idempotent based on audio bytes hash + model name.
    Writes:
      transcript_txt_out
      transcript_txt_out.json (full whisper JSON)
      transcript_txt_out.segments.txt (timestamped lines)
    """
    t0 = time.time()
    audio_hash = sha256_file(audio)
    params_fp = fingerprint_dict({"whisper_model": whisper_model, "fp16": False})
    key = StepKey("transcribe_whisper", audio_hash, params_fp)
    out_dir = key.path(cache_root)

    cached_txt = out_dir / "transcript.txt"
    cached_json = out_dir / "transcript.json"
    cached_segments = out_dir / "transcript.segments.txt"

    if cache_hit(key, cache_root) and cached_txt.exists() and cached_json.exists():
        logger.info("[asr] cache hit | audio=%s | model=%s", audio_hash[:12], whisper_model)
    else:
        logger.info("[asr] transcribing (idempotent) | model=%s ...", whisper_model)
        out_dir.mkdir(parents=True, exist_ok=True)

        model = whisper.load_model(whisper_model)
        result = model.transcribe(str(audio), fp16=False)

        # Save cached outputs
        atomic_write_text(cached_txt, (result.get("text") or "").strip() + "\n")
        atomic_write_json(cached_json, result)

        # Timestamped segments text
        seg_lines = []
        for s in result.get("segments", []) or []:
            st = float(s.get("start", 0.0))
            en = float(s.get("end", 0.0))
            tx = (s.get("text") or "").strip()
            if tx:
                seg_lines.append(f"{st:.2f} - {en:.2f}  {tx}")
        atomic_write_text(cached_segments, "\n".join(seg_lines) + ("\n" if seg_lines else ""))

        meta = {
            "step": "transcribe_whisper",
            "created_at": now_ts(),
            "audio_hash": audio_hash,
            "whisper_model": whisper_model,
            "text_chars": len((result.get("text") or "")),
            "segments": len(result.get("segments", []) or []),
        }
        cache_mark_done(key, cache_root, meta)
        logger.info("[asr] done | segments=%d | took=%.2fs", meta["segments"], time.time() - t0)

    # Materialize to requested paths
    transcript_txt_out.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(cached_txt, transcript_txt_out)

    transcript_json_out = transcript_txt_out.with_suffix(transcript_txt_out.suffix + ".json")
    shutil.copy2(cached_json, transcript_json_out)

    transcript_segments_out = transcript_txt_out.with_suffix(transcript_txt_out.suffix + ".segments.txt")
    shutil.copy2(cached_segments, transcript_segments_out)

    return transcript_txt_out, transcript_json_out, audio_hash


def step_extract_frames(
    video: Path,
    fps: float,
    frames_dir_out: Path,
    cache_root: Path,
    logger: logging.Logger,
) -> Tuple[Path, str, int]:
    """
    Idempotent based on video bytes hash + fps.
    Additionally dedupes frames by binary content hash (skips identical frames).
    Outputs are materialized into frames_dir_out, but caching is binary-hash based.
    """
    t0 = time.time()
    video_hash = sha256_file(video)
    params_fp = fingerprint_dict({"fps": fps, "format": "jpg"})
    key = StepKey("extract_frames", video_hash, params_fp)
    out_dir = key.path(cache_root)

    cached_frames_dir = out_dir / "frames"
    cached_index = out_dir / "frames_index.json"

    if cache_hit(key, cache_root) and cached_frames_dir.exists() and cached_index.exists():
        idx = json.loads(cached_index.read_text(encoding="utf-8"))
        logger.info("[frames] cache hit | video=%s | fps=%.3f | unique=%d", video_hash[:12], fps, idx["unique_frames"])
    else:
        logger.info("[frames] extracting (idempotent) | fps=%.3f ...", fps)
        out_dir.mkdir(parents=True, exist_ok=True)

        tmp_dir = out_dir / "_tmp_raw_frames"
        if tmp_dir.exists():
            shutil.rmtree(tmp_dir)
        tmp_dir.mkdir(parents=True, exist_ok=True)

        # Extract raw frames
        run_cmd([
            "ffmpeg", "-y",
            "-i", str(video),
            "-vf", f"fps={fps}",
            str(tmp_dir / "frame_%06d.jpg"),
        ], logger)

        # Deduplicate by content hash
        cached_frames_dir.mkdir(parents=True, exist_ok=True)
        seen: Dict[str, str] = {}  # hash -> canonical filename
        mapping: List[Dict[str, Any]] = []

        raw = sorted(tmp_dir.glob("frame_*.jpg"))
        for i, f in enumerate(raw, start=1):
            fb = f.read_bytes()
            fh = sha256_bytes(fb)
            ts = (i - 1) / fps

            if fh in seen:
                mapping.append({"i": i, "time": ts, "hash": fh, "dup_of": seen[fh]})
                continue

            # store unique by its hash, not original name
            uniq_name = f"{fh}.jpg"
            (cached_frames_dir / uniq_name).write_bytes(fb)
            seen[fh] = uniq_name
            mapping.append({"i": i, "time": ts, "hash": fh, "file": uniq_name})

        idx = {
            "step": "extract_frames",
            "created_at": now_ts(),
            "video_hash": video_hash,
            "fps": fps,
            "raw_frames": len(raw),
            "unique_frames": len(seen),
            "mapping": mapping,  # preserves time alignment even with dedupe
        }
        atomic_write_json(cached_index, idx)
        cache_mark_done(key, cache_root, {"unique_frames": len(seen), "raw_frames": len(raw)})

        shutil.rmtree(tmp_dir, ignore_errors=True)
        logger.info("[frames] done | raw=%d unique=%d | took=%.2fs", len(raw), len(seen), time.time() - t0)

    # Materialize: recreate requested frames_dir_out as "time-ordered symlinks/copies"
    # Using copies for portability (symlinks can be restricted in some envs).
    frames_dir_out.mkdir(parents=True, exist_ok=True)

    idx = json.loads(cached_index.read_text(encoding="utf-8"))
    mapping = idx["mapping"]

    # Clear only files we manage (jpg + index)
    for p in frames_dir_out.glob("*.jpg"):
        p.unlink(missing_ok=True)
    (frames_dir_out / "frames_index.json").unlink(missing_ok=True)

    # Write index and materialize time-ordered frames named by sequence, but content sourced by hash
    atomic_write_json(frames_dir_out / "frames_index.json", idx)

    # For each original extracted frame position, write a copy of the canonical unique frame content.
    # This preserves 1:1 frame index semantics while still being binary-cache-based.
    for entry in mapping:
        i = entry["i"]
        out_name = f"frame_{i:06d}.jpg"
        if "file" in entry:
            src = cached_frames_dir / entry["file"]
        else:
            src = cached_frames_dir / entry["dup_of"]
        shutil.copy2(src, frames_dir_out / out_name)

    return frames_dir_out, video_hash, int(idx["unique_frames"])


def step_vlm_analyze_frames(
    frames_dir: Path,
    fps: float,
    llama_url: str,
    prompt: str,
    cache_root: Path,
    logger: logging.Logger,
    frame_limit: Optional[int] = None,
) -> Path:
    """
    Idempotent per-frame based on FRAME BYTES hash + prompt fingerprint.
    Produces a JSONL in cache; materializes to runs directory.
    """
    t0 = time.time()
    # frames_dir content is derived from video; we still key each frame by its own bytes
    prompt_fp = fingerprint_dict({"prompt": prompt, "max_tokens": 512, "temperature": 0.1})

    frames = sorted(frames_dir.glob("frame_*.jpg"))
    if frame_limit is not None:
        frames = frames[:frame_limit]

    out_dir = cache_root / "vlm_frames" / prompt_fp
    out_dir.mkdir(parents=True, exist_ok=True)

    jsonl_path = out_dir / "frames_analysis.jsonl"

    # We build JSONL fresh each run, but each frame analysis is cached.
    # (This step is still idempotent because cached per-frame results are reused.)
    lines: List[str] = []
    analyzed = 0
    cached = 0

    logger.info("[vlm] analyzing frames | count=%d | prompt_fp=%s", len(frames), prompt_fp[:12])

    for idx, frame_path in enumerate(frames, start=1):
        fb = frame_path.read_bytes()
        fh = sha256_bytes(fb)

        # Per-frame cache key
        key = StepKey("vlm_frame", fh, prompt_fp)
        p = key.path(cache_root)
        cached_resp = p / "response.json"

        if cache_hit(key, cache_root) and cached_resp.exists():
            resp = json.loads(cached_resp.read_text(encoding="utf-8"))
            cached += 1
        else:
            # Call llama-server
            payload = {
                "model": "default",
                "temperature": 0.1,
                "max_tokens": 512,
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64.b64encode(fb).decode('utf-8')}"}},
                        ],
                    }
                ],
            }
            try:
                resp_raw = http_post_json(f"{llama_url.rstrip('/')}/v1/chat/completions", payload, timeout_s=180)
                content = resp_raw["choices"][0]["message"]["content"]
                if not isinstance(content, str):
                    parts = []
                    for c in content:
                        if isinstance(c, dict) and c.get("type") == "text":
                            parts.append(c.get("text", ""))
                    content = "\n".join(parts).strip()
                try:
                    resp = json.loads(content.strip())
                except json.JSONDecodeError:
                    resp = {"_raw_model_output": content.strip()}

                p.mkdir(parents=True, exist_ok=True)
                atomic_write_json(p / "response.json", resp)
                cache_mark_done(key, cache_root, {"created_at": now_ts()})
                analyzed += 1
            except Exception as e:
                resp = {"error": str(e)}
                analyzed += 1

        ts = frame_index_to_timestamp_sec(idx, fps)
        event = {"type": "frame", "time": ts, "frame_hash": fh, "frame_file": frame_path.name, "analysis": resp}
        lines.append(json.dumps(event, ensure_ascii=False))

        if idx % 50 == 0:
            logger.info("[vlm] progress | %d/%d (cached=%d analyzed=%d)", idx, len(frames), cached, analyzed)

    # Write JSONL atomically
    atomic_write_text(jsonl_path, "\n".join(lines) + ("\n" if lines else ""))
    logger.info("[vlm] done | cached=%d analyzed=%d | took=%.2fs", cached, analyzed, time.time() - t0)
    return jsonl_path


def frame_index_to_timestamp_sec(frame_idx_1based: int, fps: float) -> float:
    return (frame_idx_1based - 1) / fps


# ------------------------- synthesize_timeline (NOT idempotent) -------------------------

def synthesize_timeline(
    transcript_json: Path,
    frames_analysis_jsonl: Optional[Path],
    out_path: Path,
    logger: logging.Logger,
) -> Path:
    """
    Not idempotent by design: always regenerates output.
    Merges:
      - audio segments from transcript_json
      - frame analysis events from frames_analysis_jsonl (if provided)
    """
    t0 = time.time()
    logger.info("[timeline] synthesizing (non-idempotent) ...")

    tj = json.loads(transcript_json.read_text(encoding="utf-8"))
    segments = tj.get("segments", []) or []

    merged: List[Dict[str, Any]] = []
    for s in segments:
        text = (s.get("text") or "").strip()
        if not text:
            continue
        merged.append({
            "type": "audio",
            "time": float(s.get("start", 0.0)),
            "start": float(s.get("start", 0.0)),
            "end": float(s.get("end", 0.0)),
            "text": text,
        })

    if frames_analysis_jsonl and frames_analysis_jsonl.exists():
        for line in frames_analysis_jsonl.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                merged.append(json.loads(line))
            except json.JSONDecodeError:
                continue

    merged.sort(key=lambda x: float(x.get("time", 0.0)))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_json(out_path, {"generated_at": now_ts(), "events": merged})

    logger.info("[timeline] wrote %s | events=%d | took=%.2fs", out_path, len(merged), time.time() - t0)
    return out_path


# ------------------------- Main -------------------------

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", default=VIDEO_PATH)
    ap.add_argument("--frames_dir", default=FRAMES_DIR)
    ap.add_argument("--audio", default=AUDIO_PATH)
    ap.add_argument("--transcript", default=TRANSCRIPT_PATH)

    ap.add_argument("--fps", type=float, default=1.0)
    ap.add_argument("--whisper_model", default="small")

    ap.add_argument("--do_vlm", action="store_true")
    ap.add_argument("--llama_url", default=LLAMA_URL_DEFAULT)
    ap.add_argument("--frame_limit", type=int, default=0)

    ap.add_argument("--cache_root", default=CACHE_ROOT_DEFAULT)
    ap.add_argument("--runs_root", default=RUNS_ROOT_DEFAULT)

    args = ap.parse_args()

    video = Path(args.video).expanduser().resolve()
    frames_dir_out = Path(args.frames_dir).expanduser().resolve()
    audio_out = Path(args.audio).expanduser().resolve()
    transcript_txt_out = Path(args.transcript).expanduser().resolve()

    cache_root = Path(args.cache_root).expanduser().resolve()
    runs_root = Path(args.runs_root).expanduser().resolve()

    # Basic checks
    if not video.exists():
        print(f"Video not found: {video}", file=sys.stderr)
        return 2

    # Run directory named by video hash (binary), plus timestamp for log separation
    video_hash = sha256_file(video)
    run_dir = runs_root / f"run_{video_hash[:12]}_{time.strftime('%Y%m%d_%H%M%S')}"
    logger = setup_logger(run_dir)

    logger.info("RUN START | video=%s | fps=%.3f | whisper=%s | do_vlm=%s",
                video_hash[:12], args.fps, args.whisper_model, args.do_vlm)

    ensure_ffmpeg(logger)
    cache_root.mkdir(parents=True, exist_ok=True)

    # 1) Audio (idempotent)
    step1_t0 = time.time()
    audio_path, audio_hash = step_extract_audio_m4a(video, audio_out, cache_root, logger)
    logger.info("STEP audio done | audio=%s | took=%.2fs", audio_hash[:12], time.time() - step1_t0)

    # 2) Transcribe (idempotent)
    step2_t0 = time.time()
    transcript_txt, transcript_json, _ = step_transcribe_whisper(audio_path, args.whisper_model, transcript_txt_out, cache_root, logger)
    logger.info("STEP asr done | transcript=%s | took=%.2fs", transcript_txt.name, time.time() - step2_t0)

    # 3) Frames (idempotent)
    step3_t0 = time.time()
    frames_dir, _, uniq = step_extract_frames(video, args.fps, frames_dir_out, cache_root, logger)
    logger.info("STEP frames done | unique_frames=%d | took=%.2fs", uniq, time.time() - step3_t0)

    # 4) VLM (idempotent per frame)
    frames_analysis_jsonl: Optional[Path] = None
    if args.do_vlm:
        step4_t0 = time.time()
        limit = None if args.frame_limit == 0 else args.frame_limit
        frames_analysis_jsonl = step_vlm_analyze_frames(
            frames_dir=frames_dir,
            fps=args.fps,
            llama_url=args.llama_url,
            prompt=FRAME_PROMPT,
            cache_root=cache_root,
            logger=logger,
            frame_limit=limit,
        )
        # Materialize into run_dir for traceability
        shutil.copy2(frames_analysis_jsonl, run_dir / "frames_analysis.jsonl")
        logger.info("STEP vlm done | jsonl=%s | took=%.2fs", (run_dir / "frames_analysis.jsonl").name, time.time() - step4_t0)
    else:
        logger.info("STEP vlm skipped")

    # 5) Timeline synthesis (NOT idempotent)
    step5_t0 = time.time()
    timeline_out = run_dir / "timeline.json"
    synthesize_timeline(
        transcript_json=transcript_json,
        frames_analysis_jsonl=(run_dir / "frames_analysis.jsonl") if args.do_vlm else None,
        out_path=timeline_out,
        logger=logger,
    )
    logger.info("STEP timeline done | took=%.2fs", time.time() - step5_t0)

    # Key run summary
    summary = {
        "run_dir": str(run_dir),
        "video_path": str(video),
        "video_hash": video_hash,
        "audio_path": str(audio_out),
        "audio_hash": audio_hash,
        "frames_dir": str(frames_dir_out),
        "transcript_txt": str(transcript_txt_out),
        "transcript_json": str(transcript_json),
        "timeline_json": str(timeline_out),
        "do_vlm": bool(args.do_vlm),
        "fps": args.fps,
        "whisper_model": args.whisper_model,
        "created_at": now_ts(),
    }
    atomic_write_json(run_dir / "run_summary.json", summary)

    logger.info("RUN END | run_summary=%s", (run_dir / "run_summary.json").name)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
