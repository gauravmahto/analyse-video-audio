# Video Multimodal Pipeline

Analyze screen-recorded videos by combining audio transcription (Whisper) with visual frame understanding (Qwen3-VL via an OpenAI-compatible API).

**Key Features:**

- **Perceptual Hashing (pHash/dHash)**: Deduplicates visually identical frames, reducing API calls by 60-80% on typical screen recordings.
- **Fuzzy Text Compression**: Intelligently drops repetitive visual states using similarity matching (85% threshold), further reducing synthesis token costs.
- **Global Frame CAS**: Cross-video caching prevents re-processing identical frames across different videos.
- **Content-Addressable Cache**: Never re-process the same video twice; cache keys are based on file SHA-256 hashes.
- **Coordinator/Worker Cluster Inference**: Main servers handle synthesis while secondary servers process frame analysis with per-server slot configurations.
- **JSON Manifests & Metadata**: Comprehensive tracking of frame selection, cluster performance, and run parameters.
- **Stateful Time Tracking**: Accurately recovers elapsed time even after force-quits or restarts.
- **Network Hardened**: Automatic retries with exponential backoff and configurable timeouts for reliable distributed inference.
- **Anti-Hallucination Whisper Fixes**: Prevents the "Yeah Yeah Yeah" loops during silent sections.
- **Long-form Timestamps**: Handles videos longer than 1 hour correctly.
- **Live Terminal Dashboard**: Rich-based 2-column active processing UI for frame routing and completion.
- **Organized Output Routing**: Saves summaries to `output/` with comprehensive metadata and logs to `logs/` automatically.

## What's here

### Two pipelines for different use cases

- **`analyze_pipeline.py`** (Recommended for most users)
  - Fast parallel frame analysis with perceptual hashing (pHash)
  - Fuzzy Text Compression: automatically drops repetitive visual states (85% similarity threshold)
  - Global Frame CAS: cross-video caching of frame analyses
  - JSON manifests for frame selection and cluster performance tracking
  - Distributed inference across multiple LLM servers with per-server slot configurations
  - Stateful elapsed time tracking (survives restarts)
  - Configurable synthesis timeout for long-running final analysis
  - Real-time diagnostic tuning with `--tune-phash`
  - Best for: quick iteration, UI recordings, cost optimization

- **`analyze_video_pipeline_full.py`** (For advanced/backend use)
  - Fully documented, deterministic pipeline with hash-mode switching (pHash/dHash)
  - Per-step idempotent caching with Content-Addressable Storage (CAS) layout
  - Global Frame CAS: cross-video frame caching with prompt versioning
  - Comprehensive JSON metadata: run parameters, cluster stats, frame selection traces
  - Structured output organization with run metadata and artifact tracking
  - Network hardened with exponential backoff and configurable timeouts
  - Best for: production workflows, reproducibility, parameter research, multi-video analysis

- **`phash_visualizer.ipynb`** (Jupyter notebook for analysis & tuning)
  - Interactive visualization of perceptual hash distances
  - Threshold impact analysis and distribution charts
  - Frame-by-frame inspection of kept/dropped frames
  - Best for: understanding pHash behavior, threshold selection, debugging

## Requirements

- Python 3.10+
- `ffmpeg` on PATH
- Python packages:
  - `openai-whisper`
  - `requests`
  - `imagehash`
  - `Pillow`
  - `rich`
  - `pandas` (for notebook)
  - `matplotlib` (for notebook)
  - `seaborn` (for notebook)

## Install

```bash
pip install -U openai-whisper requests imagehash Pillow rich pandas matplotlib seaborn
```

### LLM server (llama.cpp)

The visual pipeline expects an OpenAI-compatible chat endpoint. Start `llama-server` before running the scripts:

```bash
./llama-server \
  -hf Qwen/Qwen3-VL-8B-Instruct-GGUF:Q8_0 \
  --host 127.0.0.1 --port 8033 \
  -ngl 99 \
  -fa on \
  -c 0 \
  -b 1024 \
  -ub 1024
```

### ffmpeg install

macOS (Homebrew):

```bash
brew install ffmpeg
```

Ubuntu/Debian:

```bash
sudo apt-get update && sudo apt-get install -y ffmpeg
```

Windows (winget):

```powershell
winget install Gyan.FFmpeg
```

## Run — `analyze_pipeline.py` (Recommended)

### Basic multimodal analysis

```bash
python analyze_pipeline.py --video path/to/video.mp4
```

### Audio-only mode

```bash
python analyze_pipeline.py --video path/to/video.mp4 --audio-only
```

### Coordinator/Worker cluster inference with per-server slots

```bash
python analyze_pipeline.py --video path/to/video.mp4 \
  --main-urls http://192.168.1.50:8033/v1/chat/completions:2 \
  --secondary-urls http://192.168.1.51:8033/v1/chat/completions:4 http://192.168.1.52:8033/v1/chat/completions:4
```

**Slot configuration**: Append `:N` to any URL to allocate N parallel slots (e.g., `URL:4` = 4 parallel workers on that server).

### Tune pHash threshold

Run a diagnostic to see frame distances:

```bash
python analyze_pipeline.py --video path/to/video.mp4 --tune-phash
```

Tune more frames for better calibration:

```bash
python analyze_pipeline.py --video path/to/video.mp4 --tune-phash --tune-limit 100
```

Then set your threshold:

```bash
python analyze_pipeline.py --video path/to/video.mp4 --phash-threshold 3
```

### Adaptive Spike mode (for scrolling/fast-changing videos)

For terminal recordings with rapid text scrolling, enable adaptive spike to capture burst windows:

```bash
python analyze_pipeline.py --video path/to/crash_log.mp4 --adaptive-spike --phash-threshold 5
```

This captures N additional frames after each detected change (spike window), useful for terminal sessions.

### Custom synthesis timeout (for large videos)

For very long recordings that generate extensive transcripts and visual logs:

```bash
python analyze_pipeline.py --video path/to/long_video.mp4 --synthesis-timeout 3600
```

Default timeout is 1800 seconds (30 minutes). Increase for hour-long+ recordings.

### Clear cache

```bash
python analyze_pipeline.py --video path/to/video.mp4 --clear-cache
```

---

## Run — `analyze_video_pipeline_full.py` (Advanced)

Fully documented, idempotent backend pipeline with comprehensive parameter control.

### Basic run

```bash
python analyze_video_pipeline_full.py --do-vlm
```

### Hash mode selection (pHash vs dHash)

```bash
# Use pHash (better for compressed videos, default)
python analyze_video_pipeline_full.py --do-vlm --hash-mode phash --threshold 5

# Use dHash (faster, good for UI recordings)
python analyze_video_pipeline_full.py --do-vlm --hash-mode dhash --threshold 6
```

### Cluster inference with per-server slots

```bash
python analyze_video_pipeline_full.py --do-vlm \
  --main-urls http://192.168.1.50:8033/v1/chat/completions:2 \
  --secondary-urls http://192.168.1.51:8033/v1/chat/completions:4
```

### Custom tuning with adaptive spike

```bash
python analyze_video_pipeline_full.py --do-vlm \
  --fps 1 \
  --hash-mode phash \
  --threshold 5 \
  --adaptive-spike \
  --burst-window 3 \
  --heartbeat-seconds 30 \
  --whisper-model small
```

### Hash tuning diagnostic

```bash
python analyze_video_pipeline_full.py --tune-hash --hash-mode phash --threshold 5 --tune-limit 100
```

**Threshold tuning:**

- **pHash**: `--threshold 3-4` (sensitive), `5` (balanced), `6-8` (conservative)
- **dHash**: `--threshold 4-5` (sensitive), `6` (balanced), `8+` (conservative)

---

## Analyze & Visualize — `phash_visualizer.ipynb` (Jupyter Notebook)

Interactive notebook for understanding and optimizing pHash threshold selection.

### Launch the notebook

```bash
jupyter notebook phash_visualizer.ipynb
```

### What it does

1. **Load cached frames** from a `.pipeline_cache/<hash>/frames` directory
2. **Calculate pHash** for each frame and compute Hamming distances
3. **Visualize distances over time** with kept/dropped frame coloring
4. **Display frames** that would be sent to the LLM (kept frames)
5. **Show distribution** of hash distances with cumulative curves
6. **Compare thresholds** across a range to see compression impact
7. **Print detailed statistics** on frame retention and compression ratios

### Configuration (top cell)

```python
HASH_FOLDER = "<video_hash>"   # Paste hash from .pipeline_cache/<hash>/
THRESHOLD = 5                    # Current threshold to analyze
ADAPTIVE_SPIKE = False           # Set to True for terminal/scrolling videos
FRAME_LIMIT = 100               # Limit to first N frames (0 = all)
```

### Visual indicators

- **🟢 Green (Kept - Change Spike)**: Major visual changes detected
- **🟠 Orange (Kept - Spike Falloff)**: Frames captured after a change
- **🟢 Green (Kept - First)**: Always kept (first frame)
- **⚫ Gray (Dropped - Similar)**: Visually redundant frames dropped
- **Red dashed line**: Your current threshold

### When to use

- **First time setup**: Run `--tune-phash` from CLI, then verify with notebook
- **Video type analysis**: Understand your specific video's frame patterns
- **Cost optimization**: Visualize compression ratio at different thresholds
- **Debugging**: Inspect which frames are kept and understand why

## Outputs

### `analyze_pipeline.py`

- `.pipeline_cache/<video_hash>/` — cached audio, frames, analysis
  - `frames_<fps>fps_cas/` — Binary content-addressable frame storage (SHA256-named)
  - `frame_selection_<params>.json` — JSON manifest of all frames with selection metadata
  - `cluster_stats_<params>.json` — JSON summary of cluster performance and cache hits
  - `elapsed_time_<params>.txt` — Stateful elapsed time (persists across restarts)
  - `.pipeline_cache/global_vlm_cache/` — Global cross-video frame analysis cache
- `output/<video_basename>/debug_summary_<hash>_<params>.md` — final human-readable summary
- `logs/pipeline_<video_basename>_<hash8>.log` — per-run log file
- `logs/phash_tuning_report_<video_basename>_<hash8>.log` — tuning report (when using `--tune-phash`)

### `analyze_video_pipeline_full.py`

- `.pipeline_cache/` — Content-Addressable Storage (CAS) with step-based organization
  - `audio_extract/` — Idempotent audio extraction cache
  - `whisper/` — Idempotent transcription cache with JSON metadata
  - `frames_select/` — Idempotent frame selection cache with SHA256-named frames
  - `elapsed_time/` — Stateful elapsed time tracking per parameter set
  - `global_vlm_cache/` — Global cross-video frame analysis cache
- `output/<video_basename>/` — Organized output directory per video
  - `debug_summary_<hash>_<params>.md` — Final synthesis summary
  - `frame_selection.json` — Complete frame selection trace with timestamps and distances
  - `visual_log.txt` — Timestamped visual state changes
  - `cluster_perf_summary.json` — Per-server performance stats and global cache hits
  - `run_metadata.json` — Comprehensive run parameters, cluster config, and output paths
  - `transcript-analyze_pipeline.txt` — Timestamped audio transcript
  - `transcript-analyze_pipeline.txt.json` — Raw Whisper output with segment details
- `logs/pipeline_<video_basename>_<hash8>_<timestamp>.log` — Detailed per-run log with DEBUG level

## Troubleshooting

- `ffmpeg` not found: confirm it's on PATH (`ffmpeg -version`). Reopen the terminal after install.
- Whisper downloads are slow: ensure a stable network and try a smaller model (e.g., `base`).
- LLM endpoint errors: verify `--main-urls` / `--secondary-urls` are reachable and model is `qwen3-vl`.
- Missing `imagehash` or `Pillow`: Install them with `pip install imagehash Pillow`.
- Missing Rich UI: install with `pip install rich`.
- **Synthesis timeout errors**: If synthesis fails with timeout, increase with `--synthesis-timeout 3600` (especially for hour+ videos).
- **Empty final summary**: Check logs for VLM failures. Fuzzy compression may drop all states if they're too repetitive—verify frame selection JSON.

## Which pipeline should I use?

| Feature             | `analyze_pipeline.py`                 | `analyze_video_pipeline_full.py`               |
| ------------------- | ------------------------------------- | ---------------------------------------------- |
| Speed               | ⚡ Fast (parallel)                    | 🔧 Flexible                                    |
| Ease of use         | ✅ Recommended                        | Advanced                                       |
| Frame filtering     | pHash                                 | pHash or dHash (switchable)                    |
| Hash mode switching | No                                    | ✅ `--hash-mode phash/dhash`                   |
| Fuzzy text compress | ✅ 85% similarity auto-dedup          | No (manual compression only)                   |
| Tuning              | `--tune-phash`                        | `--tune-hash`                                  |
| Cluster mode        | ✅ `--main-urls` + `--secondary-urls` | ✅ `--main-urls` + `--secondary-urls`          |
| Per-server slots    | ✅ `URL:N` syntax                     | ✅ `URL:N` syntax                              |
| Synthesis timeout   | ✅ `--synthesis-timeout`              | ✅ `--timeout`                                 |
| Caching             | Per-file + Global Frame CAS           | Per-step CAS + Global Frame CAS                |
| JSON metadata       | ✅ Manifests + cluster stats          | ✅ Full run metadata + performance tracking    |
| Stateful time track | ✅ Survives restarts                  | ✅ Survives restarts                           |
| Network hardening   | ✅ Exponential backoff                | ✅ Configurable timeouts + backoff             |
| Output organization | `output/` + `logs/`                   | `output/<video>/` with comprehensive artifacts |
| Best for            | UI/terminal recordings, quick runs    | Production, multi-video, research              |

## Notes

- Both pipelines expect `qwen3-vl` model when using VLM.
- **Content-based caching**: Videos identified by SHA-256 hash, not filename. Prevents re-processing renamed files.
- **Global Frame CAS**: Both pipelines cache frame analyses across videos, preventing redundant API calls for identical frames.
- **Fuzzy Text Compression** (`analyze_pipeline.py` only): Uses `difflib.SequenceMatcher` to compare consecutive VLM outputs. If two states are ≥85% similar, the duplicate is dropped. This can reduce synthesis prompt size by 30-50% on static UI recordings.
- **pHash vs dHash**:
  - pHash: More robust under compression, recommended for screen recordings (both pipelines)
  - dHash: Faster computation, good for UI recordings (`analyze_video_pipeline_full.py` only)
- **Perceptual hashing** can reduce API calls by 60-80% on typical screen recordings.
- **JSON metadata**: Both pipelines now output comprehensive tracking data for frame selection, cluster performance, and run parameters.
- **Adaptive Spike Mode**: Enables burst frame capture after changes, best for terminal logs and scrolling content.
- **Threshold recommendations**:
  - Silent/mostly static videos: `--threshold 6-8` (fewer API calls)
  - UI/terminal with moderate activity: `--threshold 5` (default, balanced)
  - Fast scrolling/coding sessions: `--threshold 3-4` with `--adaptive-spike` (more frames)
- **Synthesis timeout**: Default is 1800s (30 min). For hour-long videos with extensive logs, use `--synthesis-timeout 3600` or higher.
