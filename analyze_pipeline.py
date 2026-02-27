"""
Multimodal AI Debugging Pipeline
================================

This script processes screen recordings of terminal sessions and automatically generates
a debugging summary by combining an audio transcript with a visual terminal log.

Key Features:
- Content-Addressable Cache (CAS): Prevents re-processing identical videos.
- Audio Hallucination Fix: Prevents Whisper from looping during silence.
- Intelligent pHash Deduplication: Uses Perceptual Hashing to drop identical video frames,
  saving massive amounts of API compute time.
- Distributed Inference: Supports load-balancing across multiple LLM server instances.
"""

# ==============================================================================
# USE CASE 1: Standard Multimodal Analysis (The Default)
# python analyze_pipeline.py
#
# USE CASE 2: Audio-Only Analysis
# python analyze_pipeline.py --video path/to/my_recording.mp4 --audio-only
#
# USE CASE 3: Processing a Pure Audio File
# python analyze_pipeline.py --video path/to/meeting.mp3 --audio-only
#
# USE CASE 4: Overriding the Audio Extraction Format
# python analyze_pipeline.py --video tutorial.mkv --audio-type wav
#
# USE CASE 5: Distributed Parallel Inference (Multiple LLM Servers)
# python analyze_pipeline.py --video bug_report.mp4 --api-urls http://ip1:8080/v1/chat/completions http://ip2:8080/v1/chat/completions
# ==============================================================================

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

# --- Image Processing Imports ---
from PIL import Image
import imagehash

# --- Global Defaults ---
VIDEO_PATH = "video1822201159.mp4"
LLAMA_API_URL = "http://127.0.0.1:8033/v1/chat/completions"
BASE_CACHE_DIR = ".pipeline_cache"

# --- Setup Logging ---
# Configured to stream to the terminal and save persistently to a log file.
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
    """
    Generates a SHA-256 hash of a file's binary content.

    This acts as the unique signature for the Content-Addressable Storage (CAS) cache.
    By reading the file in chunks, we prevent out-of-memory (OOM) errors when 
    processing massive video files.

    Args:
        filepath (str): Path to the media file.
        chunk_size (int): Bytes to read into memory at once (Default: 8MB).

    Returns:
        str: The SHA-256 hexadecimal hash string.
    """
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
        exit(1)


def setup_cache(video_hash, audio_type):
    """
    Creates a unique hidden cache directory based on the file's hash.

    Args:
        video_hash (str): The SHA-256 hash of the input file.
        audio_type (str): The target audio extension (e.g., 'm4a', 'wav').

    Returns:
        dict: A dictionary containing all the absolute paths used by the pipeline.
    """
    cache_dir = os.path.join(BASE_CACHE_DIR, video_hash)
    frames_dir = os.path.join(cache_dir, "frames")

    os.makedirs(frames_dir, exist_ok=True)

    paths = {
        "dir": cache_dir,
        "frames_dir": frames_dir,
        "audio": os.path.join(cache_dir, f"audio.{audio_type}"),
        "transcript": os.path.join(cache_dir, "transcript.txt"),
        "visual_log": os.path.join(cache_dir, "visual_log.txt"),
        "final_analysis": os.path.join(cache_dir, "final_analysis.md")
    }
    return paths


def get_transcript(paths, video_path):
    """
    Extracts audio using FFmpeg and transcribes it using OpenAI's Whisper.

    Features:
    - Caching: Instantly skips transcription if a valid cache file exists.
    - Anti-Hallucination: Prevents Whisper from repeating words during long silences.
    - Long-form timestamps: Safely handles audio files longer than 1 hour.

    Args:
        paths (dict): The dictionary of cache paths.
        video_path (str): Path to the input media.

    Returns:
        str: The fully formatted textual transcript with timestamps.
    """
    if os.path.exists(paths["transcript"]) and os.path.getsize(paths["transcript"]) > 0:
        logger.info("Transcript cache hit. Skipping Whisper transcription.")
        with open(paths["transcript"], "r") as f:
            return f.read()

    logger.info("Transcript cache miss. Initiating audio pipeline...")

    # Extract raw audio
    logger.info(f"Extracting audio to {paths['audio']}...")
    command = [
        "ffmpeg", "-y", "-i", video_path,
        "-vn", "-ar", "16000", "-ac", "1", paths["audio"]
    ]
    subprocess.run(command, stdout=subprocess.DEVNULL,
                   stderr=subprocess.DEVNULL)

    logger.info("Running Whisper model transcription...")
    model = whisper.load_model("base")

    # condition_on_previous_text=False prevents the "Yeah Yeah Yeah" hallucination loops
    result = model.transcribe(paths["audio"], condition_on_previous_text=False)

    # Custom time formatter to prevent rolling back to 00:00 after 59 minutes
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


def get_visual_log(paths, video_path, api_urls, fps=1):
    """
    Extracts video frames, filters them using Perceptual Hashing (pHash), 
    and sends them to LLM instances for analysis in parallel.

    Args:
        paths (dict): The dictionary of cache paths.
        video_path (str): Path to the input media.
        api_urls (list): A list of LLM endpoint URLs for load balancing.
        fps (int): Frames to extract per second of video.

    Returns:
        list: An array of timestamped text descriptions of the terminal state.
    """
    if os.path.exists(paths["visual_log"]) and os.path.getsize(paths["visual_log"]) > 0:
        logger.info(
            "Visual log cache hit. Skipping FFmpeg and Qwen3-VL analysis.")
        with open(paths["visual_log"], "r") as f:
            return [line.strip() for line in f.readlines() if line.strip()]

    logger.info("Visual log cache miss. Initiating visual pipeline...")

    # 1. Extract Raw Frames
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

    # =========================================================================
    # --- HOW TO TUNE THE pHash LOGIC ---
    # PHASH_THRESHOLD determines how sensitive the filter is to changes.
    #   - 0: Exactly identical pixels required (Too strict, captures compression artifacts).
    #   - 5 (Default): Perfect for terminal windows. Catches new lines of text.
    #   - 10+: Use if your video has heavy artifacts/glitches to avoid false triggers.
    #   - 1 to 3: Use if you are missing tiny cursor changes or single-character edits.
    # =========================================================================
    target_frames = []
    PHASH_THRESHOLD = 5         # Hash distance required to trigger a change event
    SPIKE_DURATION = 3          # Capture this many frames after a change is detected
    MAX_STATIC_SECONDS = 30     # Force a "heartbeat" frame if the screen is frozen

    last_hash = None
    spike_counter = 0
    frames_since_last_sent = 0

    # 2. Filter Frames (Intelligent Downsampling)
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

        if distance > PHASH_THRESHOLD:
            # Change detected: Trigger an Adaptive Spike
            target_frames.append(frame_path)
            last_hash = current_hash
            spike_counter = SPIKE_DURATION
            frames_since_last_sent = 0

        elif spike_counter > 0:
            # We are inside an active spike window
            target_frames.append(frame_path)
            last_hash = current_hash
            spike_counter -= 1
            frames_since_last_sent = 0

        elif frames_since_last_sent >= MAX_STATIC_SECONDS:
            # Heartbeat fallback for static screens
            target_frames.append(frame_path)
            last_hash = current_hash
            frames_since_last_sent = 0

        else:
            # Frame is visually identical. Skip to save API cost.
            frames_since_last_sent += 1

    total_targets = len(target_frames)
    compression_ratio = round(
        (1 - (total_targets / max(1, len(frames)))) * 100, 1)
    logger.info(
        f"pHash Dedupe complete. Dropped {compression_ratio}% of redundant frames.")
    logger.info(
        f"Distributing {total_targets} critical frames across {len(api_urls)} servers...")

    # 3. Analyze Frames via Parallel Inference
    def process_frame(task_data):
        """Worker function executed by the thread pool for a single frame."""
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

    # Map target frames to servers in a Round-Robin pattern
    tasks = []
    for i, frame in enumerate(target_frames, 1):
        target_url = api_urls[i % len(api_urls)]
        tasks.append((i, frame, target_url))

    visual_log = []
    # Ensure pipelines stay full. Adjust max_workers if your hardware allows.
    max_concurrent = len(api_urls) * 2

    # Using executor.map automatically ensures chronological order of results,
    # even if Server B finishes its task faster than Server A.
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
    """
    Synthesizes the audio transcript and visual logs into a final cohesive summary.

    Args:
        transcript (str): The audio timeline text.
        visual_log (list): The array of visual state descriptions.
        api_url (str): The primary LLM endpoint used for final generation.

    Returns:
        str: The final markdown debugging summary.
    """
    logger.info("Executing Synthesis Phase (Non-Idempotent)...")

    # Final Text Compression: If the LLM outputted the exact same description
    # for two adjacent frames (despite pHash passing it), deduplicate the text
    # to save massive amounts of context tokens.
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
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": master_prompt
                    }
                ]
            }
        ],
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
    """Entry point and CLI handler."""
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

    args = parser.parse_args()

    logger.info("=== Starting Multimodal Pipeline Execution ===")
    logger.info(f"Target Media: {args.video}")
    logger.info(f"Active LLM Servers: {len(args.api_urls)} instance(s)")
    if args.audio_only:
        logger.info("MODE: AUDIO-ONLY")

    # Automatically map the output audio format to the input file format if not provided
    input_ext = os.path.splitext(args.video)[1].lower().strip('.')
    if not input_ext:
        input_ext = "m4a"

    target_audio_type = args.audio_type if args.audio_type else input_ext
    video_hash = get_file_hash(args.video)

    # Allow the user to force-wipe a corrupted or stale cache
    if args.clear_cache:
        cache_dir_to_clear = os.path.join(BASE_CACHE_DIR, video_hash)
        if os.path.exists(cache_dir_to_clear):
            logger.info(f"Wiping existing cache at {cache_dir_to_clear}...")
            shutil.rmtree(cache_dir_to_clear)
        else:
            logger.info("No existing cache found to clear.")

    paths = setup_cache(video_hash, target_audio_type)
    transcript = get_transcript(paths, args.video)

    visual_log = []
    if not args.audio_only:
        visual_log = get_visual_log(paths, args.video, args.api_urls, fps=1)

    # Route the final text synthesis request to the first server in the pool
    final_analysis = synthesize_timeline(
        transcript, visual_log, args.api_urls[0])

    # Save to the hidden cache for reference
    with open(paths["final_analysis"], "w") as f:
        f.write(final_analysis)

    # Dump a human-readable copy to the root directory
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
