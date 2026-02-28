"""
Multimodal AI Debugging Pipeline
================================

This script processes screen recordings of terminal sessions and automatically generates
a debugging summary by combining an audio transcript with a visual terminal log.

Key Features:
- Content-Addressable Cache (CAS): Prevents re-processing identical videos and parameters.
- Global Frame CAS: Caches raw image bytes + exact prompt across all videos.
- True Binary Frames: Images are stored by their SHA256 hash instead of chronological filenames.
- Immutable Metadata: Outputs structured frame_selection.json and cluster_stats.json.
- Stateful Time Tracking: Recovers execution time accurately even after force-quits (Ctrl+C).
- Audio Hallucination Fix: Aggressive silence suppression for terminal recordings.
- Intelligent pHash Deduplication: Uses Perceptual Hashing to drop identical video frames.
- Fuzzy Text Compression: Slides a similarity window over VLM outputs to drop repetitive states.
- Coordinator/Worker Inference: True Dynamic Load Balancing across cluster machines.
- Per-Server Slot Configurations: Dynamically allocate GPU slots per machine (e.g., URL:4).
- Network Hardened: Automatic retries with exponential backoff and request timeouts.
- Optional Live Terminal UI: 2-column active processing dashboard (graceful fallback if missing).
"""

import os
import subprocess
import base64
import requests
import glob
import time
import whisper
import hashlib
import logging
import argparse
import shutil
import concurrent.futures
import sys
import threading
import queue
import json  # Added for manifest generation
import difflib  # Added for Fuzzy Text Compression

# --- Image Processing Imports ---
from PIL import Image
import imagehash

# --- Optional Live UI Imports ---
try:
    from rich.live import Live
    from rich.table import Table
    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False

# --- Global Defaults & Prompts ---
VIDEO_PATH = "video1822201159.mp4"
LLAMA_API_URL = "http://127.0.0.1:8033/v1/chat/completions"
BASE_CACHE_DIR = ".pipeline_cache"
GLOBAL_VLM_CACHE_DIR = os.path.join(BASE_CACHE_DIR, "global_vlm_cache")

# Defined globally so it can be securely injected into the CAS hash key
VISUAL_PROMPT = "Analyze this screen capture. What are the steps covered from start to end, and are there any errors or warnings visible?"

# --- Setup Base Logging (Console Only Initially) ---
os.makedirs("logs", exist_ok=True)
os.makedirs(GLOBAL_VLM_CACHE_DIR, exist_ok=True)

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s [%(levelname)s] %(message)s')

console_handler = logging.StreamHandler()
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)


def parse_url_args(url_list, default_slots=1):
    """Parses URLs and extracts custom slot counts if provided (e.g., url:4)."""
    parsed = []
    for item in url_list:
        parts = item.rsplit(':', 1)
        if len(parts) == 2 and parts[1].isdigit():
            parsed.append((parts[0], int(parts[1])))
        else:
            parsed.append((item, default_slots))
    return parsed


def get_file_hash(filepath, chunk_size=8 * 1024 * 1024):
    """Generates a SHA-256 hash of a file's binary content for CAS caching."""
    hasher = hashlib.sha256()
    try:
        with open(filepath, 'rb') as f:
            for chunk in iter(lambda: f.read(chunk_size), b''):
                hasher.update(chunk)
        return hasher.hexdigest()
    except FileNotFoundError:
        return None


def setup_cache(video_hash, audio_type, fps, threshold, adaptive_spike, model_name):
    """Creates cache directories and dynamically names files based on run parameters."""
    cache_dir = os.path.join(BASE_CACHE_DIR, video_hash)

    frames_dir = os.path.join(cache_dir, f"frames_{fps}fps_cas")
    os.makedirs(frames_dir, exist_ok=True)

    param_fingerprint = f"fps{fps}_t{threshold}_s{adaptive_spike}_{model_name}"

    return {
        "dir": cache_dir,
        "frames_dir": frames_dir,
        "audio": os.path.join(cache_dir, f"audio.{audio_type}"),
        "transcript": os.path.join(cache_dir, "transcript.txt"),
        "visual_log": os.path.join(cache_dir, f"visual_log_{param_fingerprint}.txt"),
        "final_analysis": os.path.join(cache_dir, f"final_analysis_{param_fingerprint}.md"),
        "elapsed_time": os.path.join(cache_dir, f"elapsed_time_{param_fingerprint}.txt"),
        "frame_manifest": os.path.join(cache_dir, f"frame_selection_{param_fingerprint}.json"),
        "cluster_stats": os.path.join(cache_dir, f"cluster_stats_{param_fingerprint}.json")
    }


def format_timestamp(seconds):
    """Formats raw seconds into a readable MM:SS or HH:MM:SS string."""
    seconds = int(seconds)
    h, remainder = divmod(seconds, 3600)
    m, s = divmod(remainder, 60)
    if h > 0:
        return f"{h:02d}:{m:02d}:{s:02d}"
    return f"{m:02d}:{s:02d}"


def format_duration(seconds):
    """Formats overall elapsed time for the logs."""
    h, remainder = divmod(int(seconds), 3600)
    m, s = divmod(remainder, 60)
    if h > 0:
        return f"{h}h {m:02d}m {s:02d}s"
    return f"{m:02d}m {s:02d}s"


def robust_api_call(url, payload, max_retries=3, timeout=180):
    """Network hardened API call with performance telemetry (Tokens/Second)."""
    for attempt in range(max_retries):
        try:
            start_time = time.time()
            response = requests.post(
                url, headers={"Content-Type": "application/json"}, json=payload, timeout=timeout)

            # If the server throws a 400 or 500 error, this triggers the exception
            response.raise_for_status()

            duration = time.time() - start_time
            data = response.json()
            content = data["choices"][0]["message"]["content"]

            usage = data.get("usage", {})
            prompt_tokens = usage.get("prompt_tokens", 0)
            completion_tokens = usage.get("completion_tokens", 0)

            tps = completion_tokens / duration if duration > 0 else 0

            stats = {
                "duration": duration,
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "tps": tps
            }

            return content, stats

        except requests.exceptions.RequestException as e:
            # --- THE FIX: Extract the actual server rejection details ---
            error_details = ""
            if hasattr(e, 'response') and e.response is not None:
                error_details = f" | Server Details: {e.response.text}"

            if attempt == max_retries - 1:
                logger.error(
                    f"API Failed after {max_retries} attempts on {url}: {e}{error_details}")
                # Raise a custom exception that includes the server details so synthesize_timeline catches it
                raise Exception(f"{e}{error_details}")

            logger.warning(
                f"Network error on attempt {attempt+1}/{max_retries}. Retrying... ({e})")
            time.sleep(2 ** attempt)


def tune_phash_diagnostic(frames_dir, threshold, limit=50, output_file=None, adaptive_spike=False):
    """Diagnostic Tool: Scans a directory of sequential frames and prints the pHash distance."""
    frames = sorted(glob.glob(os.path.join(frames_dir, "*.jpg")))

    if not frames:
        logger.error(f"No frames found in {frames_dir} to tune.")
        return

    if limit > 0:
        frames = frames[:limit]

    report_lines = []

    def log_both(message):
        print(message)
        report_lines.append(message)

    log_both(f"\n{'='*60}")
    log_both(f"🔍 PERCEPTUAL HASH (pHash) TUNING DIAGNOSTIC")
    log_both(f"{'='*60}")
    log_both(f"Analyzing {len(frames)} frames...")
    log_both(
        f"Target Threshold: {threshold} | Adaptive Spike: {adaptive_spike}\n")
    log_both(f"{'Frame':<15} | {'Dist':<4} | {'Action'}")
    log_both("-" * 60)

    last_hash = None
    kept_count = 0
    SPIKE_DURATION = 3
    MAX_STATIC_SECONDS = 30
    spike_counter = 0
    frames_since_last_sent = 0

    for frame_path in frames:
        frame_name = os.path.basename(frame_path)
        try:
            with Image.open(frame_path) as img:
                current_hash = imagehash.phash(img)
        except Exception as e:
            log_both(f"Error reading {frame_name}: {e}")
            continue

        if last_hash is None:
            log_both(f"{frame_name:<15} | {'N/A':<4} | 🟢 KEPT (First Frame)")
            last_hash = current_hash
            kept_count += 1
            continue

        distance = current_hash - last_hash

        if distance > threshold:
            action = "🟢 KEPT (Change Spike)"
            last_hash = current_hash
            if adaptive_spike:
                spike_counter = SPIKE_DURATION
            frames_since_last_sent = 0
            kept_count += 1
        elif adaptive_spike and spike_counter > 0:
            action = "🟡 KEPT (Spike Falloff)"
            last_hash = current_hash
            spike_counter -= 1
            frames_since_last_sent = 0
            kept_count += 1
        elif frames_since_last_sent >= MAX_STATIC_SECONDS:
            action = "🔵 KEPT (Heartbeat)"
            last_hash = current_hash
            frames_since_last_sent = 0
            kept_count += 1
        else:
            action = "🔴 DROPPED (Too Similar)"
            frames_since_last_sent += 1

        bar = "█" * distance
        log_both(f"{frame_name:<15} | {distance:<4} | {action} {bar}")

    log_both("-" * 60)
    log_both(
        f"Result: Would keep {kept_count}/{len(frames)} frames ({(1 - kept_count/len(frames))*100:.1f}% compression)")
    log_both(f"{'='*60}\n")

    if output_file:
        with open(output_file, "w", encoding="utf-8") as f:
            f.write("\n".join(report_lines) + "\n")
        logger.info(f"📄 Full diagnostic report saved to: {output_file}")


def get_transcript(paths, video_path):
    """Extracts audio using FFmpeg and transcribes it using Whisper."""
    if os.path.exists(paths["transcript"]) and os.path.getsize(paths["transcript"]) > 0:
        logger.info("Transcript cache hit. Skipping Whisper transcription.")
        with open(paths["transcript"], "r") as f:
            return f.read()

    logger.info("Transcript cache miss. Initiating audio pipeline...")
    logger.info(f"Extracting audio to {paths['audio']}...")
    command = [
        "ffmpeg", "-y", "-i", video_path,
        "-vn", "-ar", "16000", "-ac", "1", paths["audio"]
    ]
    subprocess.run(command, stdout=subprocess.DEVNULL,
                   stderr=subprocess.DEVNULL)

    logger.info("Running Whisper model transcription...")
    model = whisper.load_model("base")

    # --- AGGRESSIVE SILENCE SUPPRESSION ENABLED ---
    result = model.transcribe(
        paths["audio"],
        condition_on_previous_text=False,
        no_speech_threshold=0.6,
        logprob_threshold=-1.0
    )

    transcript = ""
    for segment in result["segments"]:
        text = segment['text'].strip()
        if not text:
            continue
        start = format_timestamp(segment["start"])
        end = format_timestamp(segment["end"])
        transcript += f"[{start} - {end}] {text}\n"

    with open(paths["transcript"], "w") as f:
        f.write(transcript)

    logger.info("Audio pipeline completed and cached.")
    return transcript


def get_visual_log(paths, video_path, cluster_nodes, threshold=5, fps=1, adaptive_spike=False, model_name="qwen3-vl"):
    """Extracts frames, hashes to true CAS, builds JSON manifest, and distributes via Queue."""

    # Check for complete Cache + Manifest combo
    if os.path.exists(paths["visual_log"]) and os.path.exists(paths["frame_manifest"]):
        logger.info(
            "Visual log and JSON Manifest cache hit. Skipping FFmpeg and Vision analysis.")
        with open(paths["visual_log"], "r") as f:
            return [line.strip() for line in f.readlines() if line.strip()]

    logger.info("Visual log cache miss. Initiating visual pipeline...")

    # --- THE NEW BINARY CAS & JSON MANIFEST GENERATION ---
    manifest_path = paths["frame_manifest"]

    if not os.path.exists(manifest_path):
        temp_dir = os.path.join(paths['frames_dir'], "temp_raw")
        os.makedirs(temp_dir, exist_ok=True)

        logger.info(f"Extracting raw frames at {fps} fps to temp directory...")
        command = [
            "ffmpeg", "-y", "-i", video_path,
            "-vf", f"fps={fps}", f"{temp_dir}/frame_%04d.jpg"
        ]
        subprocess.run(command, stdout=subprocess.DEVNULL,
                       stderr=subprocess.DEVNULL)

        temp_frames = sorted(glob.glob(os.path.join(temp_dir, "*.jpg")))
        logger.info(
            f"Loaded {len(temp_frames)} raw frames. Computing binary CAS and pHash...")

        manifest = []
        last_hash = None
        SPIKE_DURATION = 3
        MAX_STATIC_SECONDS = 30
        spike_counter = 0
        frames_since_last_sent = 0

        for i, frame_path in enumerate(temp_frames, 1):
            # 1. Compute exact raw binary hash
            with open(frame_path, 'rb') as f:
                img_bytes = f.read()
            sha256_hash = hashlib.sha256(img_bytes).hexdigest()
            cas_filename = f"{sha256_hash}.jpg"
            cas_filepath = os.path.join(paths['frames_dir'], cas_filename)

            # 2. Convert to true CAS storage
            if not os.path.exists(cas_filepath):
                shutil.move(frame_path, cas_filepath)
            else:
                os.remove(frame_path)

            time_sec = (i - 1) / fps
            timestamp = format_timestamp(time_sec)
            kept = False
            action = ""

            # 3. pHash Logic
            try:
                with Image.open(cas_filepath) as img:
                    current_hash = imagehash.phash(img)
            except Exception as e:
                logger.error(
                    f"Failed to read image for hashing: {cas_filepath}. Error: {e}")
                continue

            if last_hash is None:
                kept = True
                action = "First Frame"
                last_hash = current_hash
            else:
                distance = current_hash - last_hash
                if distance > threshold:
                    kept = True
                    action = "Change Spike"
                    last_hash = current_hash
                    if adaptive_spike:
                        spike_counter = SPIKE_DURATION
                    frames_since_last_sent = 0
                elif adaptive_spike and spike_counter > 0:
                    kept = True
                    action = "Spike Falloff"
                    last_hash = current_hash
                    spike_counter -= 1
                    frames_since_last_sent = 0
                elif frames_since_last_sent >= MAX_STATIC_SECONDS:
                    kept = True
                    action = "Heartbeat"
                    last_hash = current_hash
                    frames_since_last_sent = 0
                else:
                    action = "Too Similar"
                    frames_since_last_sent += 1

            # 4. Append structured logic to manifest
            manifest.append({
                "index": i,
                "time_sec": time_sec,
                "timestamp": timestamp,
                "sha256": sha256_hash,
                "filename": cas_filename,
                "kept": kept,
                "reason": action
            })

        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)

        with open(manifest_path, "w") as f:
            json.dump(manifest, f, indent=4)
    else:
        logger.info("JSON Manifest found. Loading existing frame metadata...")
        with open(manifest_path, "r") as f:
            manifest = json.load(f)

    # Filter strictly by frames marked as "kept" in the JSON ledger
    target_frames_meta = [m for m in manifest if m["kept"]]
    total_targets = len(target_frames_meta)

    compression_ratio = round(
        (1 - (total_targets / max(1, len(manifest)))) * 100, 1)
    logger.info(
        f"pHash Dedupe complete. Dropped {compression_ratio}% of redundant frames.")

    # --- TRUE DYNAMIC LOAD BALANCER W/ CUSTOM SLOTS ---
    url_pool = queue.Queue()
    active_status = {}
    total_slots = 0

    for url, slots in cluster_nodes:
        for i in range(slots):
            slot_key = f"{url} (Slot {i+1})"
            url_pool.put(f"{url}|{i+1}")
            active_status[slot_key] = "Idle"
            total_slots += 1

    logger.info(
        f"Dynamically distributing {total_targets} frames across {total_slots} global compute slots...")

    # --- STATEFUL TIME & PROGRESS TRACKING INITIALIZATION ---
    completed_log = []
    status_lock = threading.Lock()

    cluster_stats = {}
    global_cache_hit_count = 0
    global_completed_count = 0

    previous_elapsed_time = 0.0
    if os.path.exists(paths["elapsed_time"]):
        try:
            with open(paths["elapsed_time"], "r") as f:
                previous_elapsed_time = float(f.read().strip())
        except Exception:
            previous_elapsed_time = 0.0

    session_start_time = time.time()

    def generate_table():
        table = Table(show_header=True, header_style="bold cyan")
        table.add_column("✅ Completed Frames", style="dim green", width=55)
        table.add_column("⏳ Active Processing", style="yellow", width=55)

        num_rows = total_slots

        with status_lock:
            recent_completed = completed_log[-num_rows:]
            while len(recent_completed) < num_rows:
                recent_completed.insert(0, "")

            active_list = [
                f"[{slot_key.split('//')[-1]}] {task}" for slot_key, task in active_status.items()]

            for comp, act in zip(recent_completed, active_list):
                table.add_row(comp, act)

        return table

    def process_frame(task_data):
        nonlocal global_cache_hit_count, global_completed_count
        index, meta = task_data

        # Read true properties securely from the JSON Manifest
        fname = meta["filename"]
        sha256_hash = meta["sha256"]
        timestamp = meta["timestamp"]
        frame_path = os.path.join(paths['frames_dir'], fname)

        with open(frame_path, "rb") as image_file:
            img_bytes = image_file.read()
            base64_image = base64.b64encode(img_bytes).decode('utf-8')

            # --- PROMPT INJECTION: Securing the cache key against prompt changes ---
            cache_hasher = hashlib.sha256(img_bytes)
            cache_hasher.update(model_name.encode('utf-8'))
            cache_hasher.update(VISUAL_PROMPT.encode('utf-8'))
            global_cache_key = cache_hasher.hexdigest()
            global_cache_path = os.path.join(
                GLOBAL_VLM_CACHE_DIR, f"{global_cache_key}.txt")

        # -----------------------------------------
        # SCENARIO 1: GLOBAL CACHE HIT
        # -----------------------------------------
        if os.path.exists(global_cache_path):
            with open(global_cache_path, "r") as f:
                analysis = f.read().strip()

            with status_lock:
                global_cache_hit_count += 1
                global_completed_count += 1
                current_total_time = previous_elapsed_time + \
                    (time.time() - session_start_time)

                with open(paths["elapsed_time"], "w") as f_time:
                    f_time.write(str(current_total_time))

                elapsed_str = format_duration(current_total_time)
                # Keep UI clean by showing the short hash
                completed_log.append(f"{sha256_hash[:8]} -> GLOBAL CACHE HIT")

            logger.info(
                f"♻️  [{global_completed_count:03d}/{total_targets:03d}] {sha256_hash[:8]}.jpg "
                f"| GLOBAL CACHE HIT        "
                f"| 100% API Compute Saved                      "
                f"| Elapsed: {elapsed_str:>8}"
            )
            return f"[{timestamp}] Visual State: {analysis}"

        # -----------------------------------------
        # SCENARIO 2: LIVE NETWORK API CALL
        # -----------------------------------------
        assigned_slot = url_pool.get()
        target_url, slot_num = assigned_slot.split("|")
        slot_key = f"{target_url} (Slot {slot_num})"
        server_short = target_url.split('//')[-1].split('/')[0]

        with status_lock:
            active_status[slot_key] = f"{sha256_hash[:8]} ({index}/{total_targets})"

        payload = {
            "model": model_name,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": VISUAL_PROMPT},
                        {"type": "image_url", "image_url": {
                            "url": f"data:image/jpeg;base64,{base64_image}"}}
                    ]
                }
            ],
            "temperature": 0.1, "max_tokens": 150
        }

        try:
            analysis, stats = robust_api_call(target_url, payload)

            with open(global_cache_path, "w") as f:
                f.write(analysis.strip())

            with status_lock:
                global_completed_count += 1

                if server_short not in cluster_stats:
                    cluster_stats[server_short] = {
                        'frames': 0, 'tokens': 0, 'time': 0.0}
                cluster_stats[server_short]['frames'] += 1
                cluster_stats[server_short]['tokens'] += stats['completion_tokens']
                cluster_stats[server_short]['time'] += stats['duration']

                machine_total_frames = cluster_stats[server_short]['frames']
                current_total_time = previous_elapsed_time + \
                    (time.time() - session_start_time)

                with open(paths["elapsed_time"], "w") as f_time:
                    f_time.write(str(current_total_time))

                elapsed_str = format_duration(current_total_time)
                active_status[slot_key] = "Idle"
                completed_log.append(
                    f"{sha256_hash[:8]} -> {server_short} ({stats['tps']:.1f} t/s)")

            logger.info(
                f"✅ [{global_completed_count:03d}/{total_targets:03d}] {sha256_hash[:8]}.jpg "
                f"| Node: {server_short:<18} "
                f"| Node Total: {machine_total_frames:<3} "
                f"| Speed: {stats['tps']:>4.1f} t/s "
                f"| Time: {stats['duration']:>4.1f}s "
                f"| Elapsed: {elapsed_str:>8}"
            )

            return f"[{timestamp}] Visual State: {analysis.strip()}"

        except Exception:
            with status_lock:
                active_status[slot_key] = "Idle"
                completed_log.append(
                    f"{sha256_hash[:8]} -> {server_short} (FAILED)")
            return f"[{timestamp}] Visual State: Error processing frame."
        finally:
            url_pool.put(assigned_slot)

    # Pass the JSON dictionary items to the threaded worker instead of file paths
    tasks = [(i, meta) for i, meta in enumerate(target_frames_meta, 1)]
    visual_log = []
    max_concurrent = total_slots

    if console_handler in logger.handlers:
        logger.removeHandler(console_handler)

    print("\n")

    if RICH_AVAILABLE:
        with Live(get_renderable=generate_table, refresh_per_second=10):
            with concurrent.futures.ThreadPoolExecutor(max_workers=max_concurrent) as executor:
                results = executor.map(process_frame, tasks)
                for res in results:
                    visual_log.append(res)
    else:
        logger.info(
            "Rich library not detected. Running standard parallel processing (No Live UI).")
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_concurrent) as executor:
            results = executor.map(process_frame, tasks)
            for res in results:
                visual_log.append(res)
                logger.info(res)

    if console_handler not in logger.handlers:
        logger.addHandler(console_handler)

    logger.info("Distributed visual pipeline completed.")

    if cluster_stats or global_cache_hit_count > 0:
        logger.info("\n" + "="*70)
        logger.info("📊 CLUSTER VISION PERFORMANCE SUMMARY")
        logger.info("="*70)
        for server, data in cluster_stats.items():
            avg_tps = (data['tokens'] / data['time']
                       ) if data['time'] > 0 else 0
            logger.info(
                f"🖥️  {server:<18} | Frames: {data['frames']:<4} | Tokens: {data['tokens']:<6} | Avg Speed: {avg_tps:.1f} t/s")
        if global_cache_hit_count > 0:
            logger.info(
                f"♻️  Global Cache Hits  | Frames: {global_cache_hit_count:<4} | 100% API Compute Saved")

        final_elapsed = previous_elapsed_time + \
            (time.time() - session_start_time)
        logger.info(
            f"⏱️  Total Processing Time: {format_duration(final_elapsed)}")
        logger.info("="*70 + "\n")

        # --- JSON STATS LEDGER GENERATION ---
        with open(paths["cluster_stats"], "w") as f:
            json.dump({
                "final_elapsed_seconds": final_elapsed,
                "global_cache_hits": global_cache_hit_count,
                "server_performance": cluster_stats
            }, f, indent=4)

    with open(paths["visual_log"], "w") as f:
        for log in visual_log:
            f.write(log + "\n")

    return visual_log


def synthesize_timeline(transcript, visual_log, synthesis_api_url, model_name="qwen3-vl", timeout=1800):
    """Synthesizes the audio transcript and visual logs into a final cohesive summary."""
    logger.info(
        f"Executing Synthesis Phase on Coordinator Server ({synthesis_api_url})...")

    compressed_log = []
    last_state = ""

    if visual_log:
        for line in visual_log:
            try:
                state_only = line.split("] Visual State: ")[1]

                # --- FUZZY COMPRESSION ENABLED ---
                # Compares the new string to the last string. If they are >= 85% similar, we drop the new one.
                similarity = difflib.SequenceMatcher(
                    None, state_only, last_state).ratio()

                if similarity < 0.85:
                    compressed_log.append(line)
                    last_state = state_only

            except IndexError:
                compressed_log.append(line)

        logger.info(
            f"Fuzzy Text Compression dropped visual log from {len(visual_log)} down to {len(compressed_log)} critical state changes.")

    visual_text = "\n".join(compressed_log) if compressed_log else "None"

    master_prompt = f"""
    You are an expert engineer and system debugger. 
    I have provided two timelines extracted from a screen recording of a technical session.
    
    1. AUDIO TRANSCRIPT:
    {transcript if transcript else "None"}
    
    2. VISUAL TERMINAL LOG (Deduplicated for unique state changes):
    {visual_text}
    
    Please analyze both timelines together. 
    - Summarize the overall goal of the session.
    - Match spoken context with technical execution seen on screen.
    - Identify if there were any errors, warnings, or bottlenecks.
    - Create a detailed steps covered from start to end which can be referred instead of watching the complete video.
    """

    payload = {
        "model": model_name,
        "messages": [{"role": "user", "content": [{"type": "text", "text": master_prompt}]}],
        "temperature": 0.3,
        "max_tokens": 2048
    }

    try:
        logger.info(
            "Sending compressed timelines to model for final synthesis. This may take a minute or two...")

        # --- DYNAMIC TIMEOUT INJECTION ---
        analysis, stats = robust_api_call(
            synthesis_api_url, payload, timeout=timeout)

        logger.info(f"✅ Synthesis complete on {synthesis_api_url}")
        logger.info(
            f"📊 Synthesis Stats | Speed: {stats['tps']:.1f} t/s | Prompt: {stats['prompt_tokens']} tkns | Gen: {stats['completion_tokens']} tkns | Time: {stats['duration']:.1f}s")

        return analysis
    except Exception as e:
        logger.error(f"Unexpected error during final synthesis: {e}")
        return f"Error during final synthesis: {e}"


def main():
    parser = argparse.ArgumentParser(
        description="Multimodal Video Debugging Analysis")
    parser.add_argument("--video", type=str, default=VIDEO_PATH,
                        help="Path to the video or audio file")

    parser.add_argument("--main-urls", type=str, nargs='+', default=[LLAMA_API_URL],
                        help="Primary LLM servers. The FIRST one handles final heavy text synthesis. Optional slot count: e.g. URL:4")
    parser.add_argument("--secondary-urls", type=str, nargs='*', default=[],
                        help="Worker LLM servers used ONLY for parallel image processing. Optional slot count: e.g. URL:2")

    parser.add_argument("--model", type=str, default="qwen3-vl",
                        help="The model name expected by the API endpoint (default: qwen3-vl)")

    # --- NEW: CONFIGURABLE TIMEOUT ---
    parser.add_argument("--synthesis-timeout", type=int, default=1800,
                        help="Timeout in seconds for the final synthesis API call (default: 1800).")

    parser.add_argument("--audio-only", action="store_true",
                        help="Only process audio, skipping visual pipeline")
    parser.add_argument("--audio-type", type=str,
                        help="Output audio format (e.g., m4a, wav, mp3). Defaults to input file extension.")
    parser.add_argument("--clear-cache", action="store_true",
                        help="Delete existing cache for this file and run fresh")

    parser.add_argument("--phash-threshold", type=int, default=5,
                        help="The perceptual hash distance required to trigger a frame capture (default: 5)")
    parser.add_argument("--tune-phash", action="store_true",
                        help="Dry-run diagnostic mode. Prints a table of frame distances to help tune the threshold.")
    parser.add_argument("--tune-limit", type=int, default=50,
                        help="Number of frames to analyze during --tune-phash. Set to 0 to analyze all frames (default: 50)")
    parser.add_argument("--adaptive-spike", action="store_true",
                        help="Enable spike memory to capture sequential frames after a change (Best for scrolling terminals)")

    args = parser.parse_args()

    main_parsed = parse_url_args(args.main_urls, default_slots=1)
    secondary_parsed = parse_url_args(args.secondary_urls, default_slots=1)
    cluster_nodes = main_parsed + secondary_parsed

    total_slots = sum(slots for url, slots in cluster_nodes)
    synthesis_url = main_parsed[0][0]  # Pure URL, stripped of :slots

    FPS = 1

    video_basename = os.path.splitext(os.path.basename(args.video))[0]
    input_ext = os.path.splitext(args.video)[1].lower().strip('.')
    if not input_ext:
        input_ext = "m4a"
    target_audio_type = args.audio_type if args.audio_type else input_ext

    video_hash = get_file_hash(args.video)
    if not video_hash:
        logger.error(f"❌ File not found: {args.video}")
        sys.exit(1)

    log_filename = os.path.join(
        "logs", f"pipeline_{video_basename}_{video_hash[:8]}.log")
    file_handler = logging.FileHandler(log_filename)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    logger.info("=== Starting Multimodal Pipeline Execution ===")
    logger.info(f"Target Media: {args.video}")
    logger.info(f"Media signature (Hash): {video_hash}")
    logger.info(f"Target LLM Model: {args.model}")

    if not args.tune_phash:
        logger.info(
            f"Cluster Size: {len(cluster_nodes)} total instances ({total_slots} global compute slots)")
        logger.info(f"Synthesis Coordinator: {synthesis_url}")
    if args.audio_only:
        logger.info("MODE: AUDIO-ONLY")

    output_dir = os.path.join("output", video_basename)
    os.makedirs(output_dir, exist_ok=True)

    if args.clear_cache:
        cache_dir_to_clear = os.path.join(BASE_CACHE_DIR, video_hash)
        if os.path.exists(cache_dir_to_clear):
            logger.info(f"Wiping existing cache at {cache_dir_to_clear}...")
            shutil.rmtree(cache_dir_to_clear)
        else:
            logger.info("No existing cache found to clear.")

    paths = setup_cache(video_hash, target_audio_type, FPS,
                        args.phash_threshold, args.adaptive_spike, args.model)

    if args.tune_phash:
        logger.info("MODE: pHash Tuning Diagnostic")

        if not glob.glob(os.path.join(paths['frames_dir'], "*.jpg")):
            logger.info("Extracting frames for diagnostic run...")
            command = ["ffmpeg", "-y", "-i", args.video, "-vf",
                       f"fps={FPS}", f"{paths['frames_dir']}/frame_%04d.jpg"]
            subprocess.run(command, stdout=subprocess.DEVNULL,
                           stderr=subprocess.DEVNULL)

        tuning_report_file = os.path.join(
            "logs", f"phash_tuning_report_{video_basename}_{video_hash[:8]}.log")
        tune_phash_diagnostic(
            paths['frames_dir'],
            args.phash_threshold,
            args.tune_limit,
            output_file=tuning_report_file,
            adaptive_spike=args.adaptive_spike
        )
        sys.exit(0)

    transcript = get_transcript(paths, args.video)

    visual_log = []
    if not args.audio_only:
        visual_log = get_visual_log(
            paths,
            args.video,
            cluster_nodes,
            threshold=args.phash_threshold,
            fps=FPS,
            adaptive_spike=args.adaptive_spike,
            model_name=args.model
        )

    # --- DYNAMIC TIMEOUT INJECTION ---
    final_analysis = synthesize_timeline(
        transcript, visual_log, synthesis_url, model_name=args.model, timeout=args.synthesis_timeout)

    with open(paths["final_analysis"], "w") as f:
        f.write(final_analysis)

    prefix = "audio_only_" if args.audio_only else ""
    param_fingerprint = f"fps{FPS}_t{args.phash_threshold}_s{args.adaptive_spike}_{args.model}"
    human_readable_file = os.path.join(
        output_dir, f"debug_summary_{prefix}{video_hash[:8]}_{param_fingerprint}.md")

    with open(human_readable_file, "w") as f:
        f.write(final_analysis)

    print("\n" + "="*50)
    print("FINAL MULTIMODAL AI DEBUGGING SUMMARY")
    print("="*50)
    print(final_analysis)
    print("="*50)
    logger.info(
        f"=== Pipeline Execution Finished. Output saved to {human_readable_file} ===")


if __name__ == "__main__":
    main()
