# ==============================================================================
# USE CASE 1: Standard Multimodal Analysis (The Default)
# Processes both video frames and audio, merging them into a final summary.
# Falls back to the hardcoded video if no --video flag is passed.
# ==============================================================================
# python analyze_pipeline.py

# Or, specifying a video explicitly:
# python analyze_pipeline.py --video path/to/my_recording.mp4


# ==============================================================================
# USE CASE 2: Audio-Only Analysis from a Video File
# Skips the heavy frame extraction/vision processing completely.
# Extracts the audio from the video and summarizes the transcript.
# ==============================================================================
# python analyze_pipeline.py --video path/to/my_recording.mp4 --audio-only


# ==============================================================================
# USE CASE 3: Processing a Pure Audio File (Meeting recording, voice memo)
# The script automatically detects the .mp3 (or .wav, .m4a) extension,
# ignores the visual pipeline (via --audio-only), and analyzes the speech.
# ==============================================================================
# python analyze_pipeline.py --video path/to/meeting.mp3 --audio-only


# ==============================================================================
# USE CASE 4: Overriding the Audio Extraction Format
# FFmpeg usually matches the input file extension (e.g., extracting .mkv audio
# into an .mkv container). This forces FFmpeg to extract it as a clean .wav file.
# ==============================================================================
# python analyze_pipeline.py --video tutorial.mkv --audio-type wav


# ==============================================================================
# USE CASE 5: Using a Custom LLM API Endpoint
# If you are running llama-server on a different port or a remote machine,
# you can point the script to it without editing the Python code.
# ==============================================================================
# python analyze_pipeline.py --video bug_report.mp4 --api-url http://192.168.1.100:8080/v1/chat/completions

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
import shutil  # <-- NEW: Required for deleting cache directories

# --- Configuration ---
VIDEO_PATH = "video1822201159.mp4"
LLAMA_API_URL = "http://127.0.0.1:8033/v1/chat/completions"
BASE_CACHE_DIR = ".pipeline_cache"

# --- Setup Logging ---
# Configured to log to both the console and a persistent file
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
    Generate a SHA-256 hash of the file's binary content.
    Reads in 8MB chunks to optimize memory and disk I/O for large videos.
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
    """Create a unique cache directory based on the video's binary hash."""
    cache_dir = os.path.join(BASE_CACHE_DIR, video_hash)
    frames_dir = os.path.join(cache_dir, "frames")

    os.makedirs(frames_dir, exist_ok=True)

    paths = {
        "dir": cache_dir,
        "frames_dir": frames_dir,
        # Dynamically set the audio extension based on user input or file type
        "audio": os.path.join(cache_dir, f"audio.{audio_type}"),
        "transcript": os.path.join(cache_dir, "transcript.txt"),
        "visual_log": os.path.join(cache_dir, "visual_log.txt"),
        "final_analysis": os.path.join(cache_dir, "final_analysis.md")
    }
    return paths


def get_transcript(paths, video_path):
    """Idempotent audio extraction and transcription."""
    if os.path.exists(paths["transcript"]) and os.path.getsize(paths["transcript"]) > 0:
        logger.info("Transcript cache hit. Skipping Whisper transcription.")
        with open(paths["transcript"], "r") as f:
            return f.read()

    logger.info("Transcript cache miss. Initiating audio pipeline...")

    # 1. Extract Audio
    logger.info(f"Extracting audio to {paths['audio']}...")
    command = [
        "ffmpeg", "-y", "-i", video_path,
        "-vn", "-ar", "16000", "-ac", "1", paths["audio"]
    ]
    subprocess.run(command, stdout=subprocess.DEVNULL,
                   stderr=subprocess.DEVNULL)

    # 2. Transcribe
    logger.info("Running Whisper model transcription...")
    model = whisper.load_model("base")

    # FIX 1: Prevent Whisper from hallucinating repeating words during silence
    result = model.transcribe(paths["audio"], condition_on_previous_text=False)

    # FIX 2: Correctly format timestamps over 1 hour long
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
        # Skip empty segments
        if not text:
            continue

        start = format_time(segment["start"])
        end = format_time(segment["end"])
        transcript += f"[{start} - {end}] {text}\n"

    with open(paths["transcript"], "w") as f:
        f.write(transcript)

    logger.info("Audio pipeline completed and cached.")
    return transcript


def get_visual_log(paths, video_path, api_url, fps=1):
    """Idempotent frame extraction and Qwen3-VL analysis."""
    if os.path.exists(paths["visual_log"]) and os.path.getsize(paths["visual_log"]) > 0:
        logger.info(
            "Visual log cache hit. Skipping FFmpeg and Qwen3-VL analysis.")
        with open(paths["visual_log"], "r") as f:
            return [line.strip() for line in f.readlines() if line.strip()]

    logger.info("Visual log cache miss. Initiating visual pipeline...")

    # 1. Extract Frames
    logger.info(f"Extracting frames at {fps} fps to {paths['frames_dir']}...")
    command = [
        "ffmpeg", "-y", "-i", video_path,
        "-vf", f"fps={fps}", f"{paths['frames_dir']}/frame_%04d.jpg"
    ]
    subprocess.run(command, stdout=subprocess.DEVNULL,
                   stderr=subprocess.DEVNULL)

    frames = sorted(glob.glob(f"{paths['frames_dir']}/*.jpg"))
    logger.info(
        f"Extracted {len(frames)} frames. Processing every 5th frame...")

    # 2. Analyze Frames
    visual_log = []
    processed_count = 0

    for frame in frames[::5]:
        logger.info(f"Sending {frame} to Qwen3-VL...")

        with open(frame, "rb") as image_file:
            base64_image = base64.b64encode(image_file.read()).decode('utf-8')

        frame_num = int(os.path.basename(frame).replace(
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
            response = requests.post(api_url, headers={
                                     "Content-Type": "application/json"}, json=payload)
            response.raise_for_status()
            analysis = response.json()["choices"][0]["message"]["content"]
            visual_log.append(
                f"[{timestamp}] Visual State: {analysis.strip()}")
            processed_count += 1
        except Exception as e:
            logger.error(f"Error processing {frame}: {e}")
            visual_log.append(
                f"[{timestamp}] Visual State: Error processing frame.")

    logger.info(
        f"Visual pipeline completed. Successfully analyzed {processed_count} frames.")

    with open(paths["visual_log"], "w") as f:
        for log in visual_log:
            f.write(log + "\n")

    return visual_log


def synthesize_timeline(transcript, visual_log, api_url):
    """The final, non-idempotent step: Generates fresh analysis every time."""
    logger.info("Executing Synthesis Phase (Non-Idempotent)...")

    # --- INTELLIGENT COMPRESSION ---
    compressed_log = []
    last_state = ""

    # Only process compression if visual logs exist (handles audio-only gracefully)
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
            f"Compressed visual log from {len(visual_log)} down to {len(compressed_log)} unique states.")

    visual_text = "\n".join(compressed_log) if compressed_log else "None"
    # -------------------------------

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
        response = requests.post(api_url, headers={
                                 "Content-Type": "application/json"}, json=payload)
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
    # --- CLI Argument Parsing ---
    parser = argparse.ArgumentParser(
        description="Multimodal Video Debugging Analysis")
    parser.add_argument("--video", type=str, default=VIDEO_PATH,
                        help="Path to the video or audio file")
    parser.add_argument("--api-url", type=str,
                        default=LLAMA_API_URL, help="LLM API Endpoint URL")
    parser.add_argument("--audio-only", action="store_true",
                        help="Only process audio, skipping visual pipeline")
    parser.add_argument("--audio-type", type=str,
                        help="Output audio format (e.g., m4a, wav, mp3). Defaults to input file extension.")
    # <-- NEW LINE
    parser.add_argument("--clear-cache", action="store_true",
                        help="Delete existing cache for this file and run fresh")

    args = parser.parse_args()

    logger.info("=== Starting Multimodal Pipeline Execution ===")
    logger.info(f"Target Media: {args.video}")
    if args.audio_only:
        logger.info("MODE: AUDIO-ONLY")

    # --- Determine Target Audio Format ---
    # Read the extension from the input file (e.g., '.mp4' -> 'mp4')
    input_ext = os.path.splitext(args.video)[1].lower().strip('.')
    if not input_ext:
        input_ext = "m4a"  # Safe fallback if file has no extension

    # If user provided --audio-type, it overrides the input extension
    target_audio_type = args.audio_type if args.audio_type else input_ext
    logger.info(f"Audio extraction format set to: .{target_audio_type}")

    # 1. Content-Based Idempotency Setup
    video_hash = get_file_hash(args.video)

    # --- NEW: Clear Cache Logic ---
    if args.clear_cache:
        cache_dir_to_clear = os.path.join(BASE_CACHE_DIR, video_hash)
        if os.path.exists(cache_dir_to_clear):
            logger.info(f"Wiping existing cache at {cache_dir_to_clear}...")
            shutil.rmtree(cache_dir_to_clear)
        else:
            logger.info("No existing cache found to clear.")
    # ------------------------------

    paths = setup_cache(video_hash, target_audio_type)

    # 2. Run Pipelines (Idempotent)
    transcript = get_transcript(paths, args.video)

    visual_log = []
    if not args.audio_only:
        visual_log = get_visual_log(paths, args.video, args.api_url, fps=1)

    # 3. Final Synthesis (Dynamic)
    final_analysis = synthesize_timeline(transcript, visual_log, args.api_url)

    # --- NEW FILE DUMP LOGIC ---
    # Save to the cache directory
    with open(paths["final_analysis"], "w") as f:
        f.write(final_analysis)

    # Save a copy to the root directory for easy reading
    prefix = "audio_only_" if args.audio_only else ""
    human_readable_file = f"debug_summary_{prefix}{video_hash[:8]}.md"
    with open(human_readable_file, "w") as f:
        f.write(final_analysis)
    # ---------------------------

    print("\n" + "="*50)
    print("FINAL MULTIMODAL AI DEBUGGING SUMMARY")
    print("="*50)
    print(final_analysis)
    print("="*50)

    logger.info(
        f"=== Pipeline Execution Finished. Output saved to {human_readable_file} ===")


if __name__ == "__main__":
    main()
