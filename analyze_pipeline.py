"""
Multimodal AI Debugging Pipeline
================================

This script processes screen recordings of terminal sessions and automatically generates
a debugging summary by combining an audio transcript with a visual terminal log.

Key Features:
- Content-Addressable Cache (CAS): Prevents re-processing identical videos.
- Audio Hallucination Fix: Prevents Whisper from looping during silence.
- Intelligent pHash Deduplication: Uses Perceptual Hashing to drop identical video frames.
- Distributed Inference: Supports load-balancing across multiple LLM server instances.
- pHash Tuner: Built-in diagnostic tool to calibrate frame filtering sensitivity.

==============================================================================
CLI USAGE EXAMPLES
==============================================================================
1. Standard Analysis (Strict Deduplication - Best for UI, Jira, Webcams)
   python analyze_pipeline.py --video bug_report.mp4

2. Adaptive Spike Mode (Best for fast-scrolling terminal logs)
   python analyze_pipeline.py --video server_crash.mp4 --adaptive-spike

3. Tune the Perceptual Hash (Dry-run diagnostic)
   python analyze_pipeline.py --video bug_report.mp4 --tune-phash --phash-threshold 5 --tune-limit 100
==============================================================================
"""

import subprocess
import base64
import requests
import os
import glob
import time
import whisper
import hashlib
import logging
import argparse
import shutil
import concurrent.futures
import sys

# --- Image Processing Imports ---
from PIL import Image
import imagehash

# --- Global Defaults ---
VIDEO_PATH = "video1822201159.mp4"
LLAMA_API_URL = "http://127.0.0.1:8033/v1/chat/completions"
BASE_CACHE_DIR = ".pipeline_cache"

# --- Setup Logging ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler("pipeline_execution.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


def get_file_hash(filepath, chunk_size=8 * 1024 * 1024):
    """Generates a SHA-256 hash of a file's binary content for CAS caching."""
    logger.info(f"Calculating SHA-256 hash for {filepath}...")
    hasher = hashlib.sha256()
    try:
        with open(filepath, 'rb') as f:
            for chunk in iter(lambda: f.read(chunk_size), b''):
                hasher.update(chunk)
        file_hash = hasher.hexdigest()
        logger.info(f"Media signature (Hash): {file_hash}")
        return file_hash
    except FileNotFoundError:
        logger.error(f"File not found: {filepath}")
        sys.exit(1)


def setup_cache(video_hash, audio_type):
    """Creates a unique hidden cache directory based on the file's hash."""
    cache_dir = os.path.join(BASE_CACHE_DIR, video_hash)
    frames_dir = os.path.join(cache_dir, "frames")
    os.makedirs(frames_dir, exist_ok=True)

    return {
        "dir": cache_dir,
        "frames_dir": frames_dir,
        "audio": os.path.join(cache_dir, f"audio.{audio_type}"),
        "transcript": os.path.join(cache_dir, "transcript.txt"),
        "visual_log": os.path.join(cache_dir, "visual_log.txt"),
        "final_analysis": os.path.join(cache_dir, "final_analysis.md")
    }


def tune_phash_diagnostic(frames_dir, threshold, limit=50, output_file=None, adaptive_spike=False):
    """
    Diagnostic Tool: Scans a directory of sequential frames and prints the pHash 
    distance to help tune the threshold. Mirrors the exact logic of the visual pipeline.
    """
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

        # UNIFIED LOGIC BLOCK
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

    result = model.transcribe(paths["audio"], condition_on_previous_text=False)

    def format_time(seconds):
        h = int(seconds // 3600)
        m = int((seconds % 3600) // 60)
        s = int(seconds % 60)
        if h > 0:
            return f"{h:02d}:{m:02d}:{s:02d}"
        return f"{m:02d}:{s:02d}"

    transcript = ""
    for segment in result["segments"]:
        text = segment['text'].strip()
        if not text:
            continue
        start = format_time(segment["start"])
        end = format_time(segment["end"])
        transcript += f"[{start} - {end}] {text}\n"

    with open(paths["transcript"], "w") as f:
        f.write(transcript)

    logger.info("Audio pipeline completed and cached.")
    return transcript


def get_visual_log(paths, video_path, api_urls, threshold=5, fps=1, adaptive_spike=False):
    """Extracts frames, filters via pHash, and distributes to LLM instances."""
    if os.path.exists(paths["visual_log"]) and os.path.getsize(paths["visual_log"]) > 0:
        logger.info(
            "Visual log cache hit. Skipping FFmpeg and Qwen3-VL analysis.")
        with open(paths["visual_log"], "r") as f:
            return [line.strip() for line in f.readlines() if line.strip()]

    logger.info("Visual log cache miss. Initiating visual pipeline...")
    logger.info(f"Extracting frames at {fps} fps to {paths['frames_dir']}...")
    command = [
        "ffmpeg", "-y", "-i", video_path,
        "-vf", f"fps={fps}", f"{paths['frames_dir']}/frame_%04d.jpg"
    ]
    subprocess.run(command, stdout=subprocess.DEVNULL,
                   stderr=subprocess.DEVNULL)

    frames = sorted(glob.glob(f"{paths['frames_dir']}/*.jpg"))
    logger.info(
        f"Extracted {len(frames)} raw frames. Analyzing perceptual hashes...")

    target_frames = []
    SPIKE_DURATION = 3
    MAX_STATIC_SECONDS = 30
    last_hash = None
    spike_counter = 0
    frames_since_last_sent = 0

    for frame_path in frames:
        try:
            with Image.open(frame_path) as img:
                current_hash = imagehash.phash(img)
        except Exception as e:
            logger.error(
                f"Failed to read image for hashing: {frame_path}. Error: {e}")
            continue

        if last_hash is None:
            target_frames.append(frame_path)
            last_hash = current_hash
            continue

        distance = current_hash - last_hash

        # UNIFIED LOGIC BLOCK
        if distance > threshold:
            target_frames.append(frame_path)
            last_hash = current_hash
            if adaptive_spike:
                spike_counter = SPIKE_DURATION
            frames_since_last_sent = 0
        elif adaptive_spike and spike_counter > 0:
            target_frames.append(frame_path)
            last_hash = current_hash
            spike_counter -= 1
            frames_since_last_sent = 0
        elif frames_since_last_sent >= MAX_STATIC_SECONDS:
            target_frames.append(frame_path)
            last_hash = current_hash
            frames_since_last_sent = 0
        else:
            frames_since_last_sent += 1

    total_targets = len(target_frames)
    compression_ratio = round(
        (1 - (total_targets / max(1, len(frames)))) * 100, 1)
    logger.info(
        f"pHash Dedupe complete. Dropped {compression_ratio}% of redundant frames.")
    logger.info(
        f"Distributing {total_targets} critical frames across {len(api_urls)} servers...")

    def process_frame(task_data):
        index, frame_path, target_url = task_data
        server_short = target_url.split('//')[-1].split('/')[0]

        logger.info(
            f"[{index}/{total_targets}] Routing {os.path.basename(frame_path)} to {server_short}...")

        with open(frame_path, "rb") as image_file:
            base64_image = base64.b64encode(image_file.read()).decode('utf-8')

        frame_num = int(os.path.basename(frame_path).replace(
            "frame_", "").replace(".jpg", ""))
        timestamp = time.strftime('%M:%S', time.gmtime(frame_num))

        payload = {
            "model": "qwen3-vl",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Analyze this terminal output. What is the current build step, and are there any errors or warnings visible?"},
                        {"type": "image_url", "image_url": {
                            "url": f"data:image/jpeg;base64,{base64_image}"}}
                    ]
                }
            ],
            "temperature": 0.1, "max_tokens": 150
        }

        try:
            response = requests.post(target_url, headers={
                                     "Content-Type": "application/json"}, json=payload)
            response.raise_for_status()
            analysis = response.json()["choices"][0]["message"]["content"]
            return f"[{timestamp}] Visual State: {analysis.strip()}"
        except Exception as e:
            logger.error(
                f"Error processing {frame_path} on {server_short}: {e}")
            return f"[{timestamp}] Visual State: Error processing frame."

    tasks = []
    for i, frame in enumerate(target_frames, 1):
        target_url = api_urls[i % len(api_urls)]
        tasks.append((i, frame, target_url))

    visual_log = []
    max_concurrent = len(api_urls) * 2

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_concurrent) as executor:
        results = executor.map(process_frame, tasks)
        for res in results:
            visual_log.append(res)

    logger.info("Distributed visual pipeline completed.")

    with open(paths["visual_log"], "w") as f:
        for log in visual_log:
            f.write(log + "\n")

    return visual_log


def synthesize_timeline(transcript, visual_log, api_url):
    """Synthesizes the audio transcript and visual logs into a final cohesive summary."""
    logger.info("Executing Synthesis Phase (Non-Idempotent)...")

    compressed_log = []
    last_state = ""

    if visual_log:
        for line in visual_log:
            try:
                state_only = line.split("] Visual State: ")[1]
                if state_only != last_state:
                    compressed_log.append(line)
                    last_state = state_only
            except IndexError:
                compressed_log.append(line)

        logger.info(
            f"Text compression dropped visual log from {len(visual_log)} down to {len(compressed_log)} unique states.")

    visual_text = "\n".join(compressed_log) if compressed_log else "None"

    master_prompt = f"""
    You are an expert DevOps engineer and system debugger. 
    I have provided two timelines extracted from a screen recording of a terminal session.
    
    1. AUDIO TRANSCRIPT:
    {transcript if transcript else "None"}
    
    2. VISUAL TERMINAL LOG (Deduplicated for unique state changes):
    {visual_text}
    
    Please analyze both timelines together. 
    - Summarize the overall goal of the session.
    - Match spoken context with technical execution seen on screen.
    - Identify if there were any errors, warnings, or bottlenecks.
    """

    payload = {
        "model": "qwen3-vl",
        "messages": [{"role": "user", "content": [{"type": "text", "text": master_prompt}]}],
        "temperature": 0.3,
        "max_tokens": 2048
    }

    try:
        logger.info(
            "Sending compressed timelines to model for final synthesis. This may take a minute or two...")
        response = requests.post(
            api_url, headers={"Content-Type": "application/json"}, json=payload)
        response.raise_for_status()
        logger.info("Synthesis complete.")
        return response.json()["choices"][0]["message"]["content"]
    except requests.exceptions.HTTPError as e:
        error_details = e.response.text
        logger.error(f"Server rejected the payload! Details: {error_details}")
        return f"Synthesis Failed. Server said: {error_details}"
    except Exception as e:
        logger.error(f"Unexpected error during final synthesis: {e}")
        return f"Error during final synthesis: {e}"


def main():
    parser = argparse.ArgumentParser(
        description="Multimodal Video Debugging Analysis")
    parser.add_argument("--video", type=str, default=VIDEO_PATH,
                        help="Path to the video or audio file")
    parser.add_argument("--api-urls", type=str, nargs='+', default=[LLAMA_API_URL],
                        help="One or more LLM API Endpoint URLs separated by spaces for load balancing")
    parser.add_argument("--audio-only", action="store_true",
                        help="Only process audio, skipping visual pipeline")
    parser.add_argument("--audio-type", type=str,
                        help="Output audio format (e.g., m4a, wav, mp3). Defaults to input file extension.")
    parser.add_argument("--clear-cache", action="store_true",
                        help="Delete existing cache for this file and run fresh")

    # --- pHash Tuning Arguments ---
    parser.add_argument("--phash-threshold", type=int, default=5,
                        help="The perceptual hash distance required to trigger a frame capture (default: 5)")
    parser.add_argument("--tune-phash", action="store_true",
                        help="Dry-run diagnostic mode. Prints a table of frame distances to help tune the threshold.")
    parser.add_argument("--tune-limit", type=int, default=50,
                        help="Number of frames to analyze during --tune-phash. Set to 0 to analyze all frames (default: 50)")

    # --- NEW: Adaptive Spike Toggle ---
    parser.add_argument("--adaptive-spike", action="store_true",
                        help="Enable spike memory to capture sequential frames after a change (Best for scrolling terminals)")

    args = parser.parse_args()

    logger.info("=== Starting Multimodal Pipeline Execution ===")
    logger.info(f"Target Media: {args.video}")
    if not args.tune_phash:
        logger.info(f"Active LLM Servers: {len(args.api_urls)} instance(s)")
    if args.audio_only:
        logger.info("MODE: AUDIO-ONLY")

    input_ext = os.path.splitext(args.video)[1].lower().strip('.')
    if not input_ext:
        input_ext = "m4a"

    target_audio_type = args.audio_type if args.audio_type else input_ext
    video_hash = get_file_hash(args.video)

    if args.clear_cache:
        cache_dir_to_clear = os.path.join(BASE_CACHE_DIR, video_hash)
        if os.path.exists(cache_dir_to_clear):
            logger.info(f"Wiping existing cache at {cache_dir_to_clear}...")
            shutil.rmtree(cache_dir_to_clear)
        else:
            logger.info("No existing cache found to clear.")

    paths = setup_cache(video_hash, target_audio_type)

    # --- Run the Diagnostic Tuner if requested ---
    if args.tune_phash:
        logger.info("MODE: pHash Tuning Diagnostic")

        if not glob.glob(os.path.join(paths['frames_dir'], "*.jpg")):
            logger.info("Extracting frames for diagnostic run...")
            command = ["ffmpeg", "-y", "-i", args.video, "-vf",
                       "fps=1", f"{paths['frames_dir']}/frame_%04d.jpg"]
            subprocess.run(command, stdout=subprocess.DEVNULL,
                           stderr=subprocess.DEVNULL)

        tuning_report_file = f"phash_tuning_report_{video_hash[:8]}.log"
        tune_phash_diagnostic(
            paths['frames_dir'],
            args.phash_threshold,
            args.tune_limit,
            output_file=tuning_report_file,
            adaptive_spike=args.adaptive_spike  # <-- Pass the toggle
        )
        sys.exit(0)

    transcript = get_transcript(paths, args.video)

    visual_log = []
    if not args.audio_only:
        visual_log = get_visual_log(
            paths,
            args.video,
            args.api_urls,
            threshold=args.phash_threshold,
            fps=1,
            adaptive_spike=args.adaptive_spike  # <-- Pass the toggle
        )

    final_analysis = synthesize_timeline(
        transcript, visual_log, args.api_urls[0])

    with open(paths["final_analysis"], "w") as f:
        f.write(final_analysis)

    prefix = "audio_only_" if args.audio_only else ""
    human_readable_file = f"debug_summary_{prefix}{video_hash[:8]}.md"
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
