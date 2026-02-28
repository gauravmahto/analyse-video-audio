# Video Multimodal Pipeline

Analyze screen-recorded videos by combining audio transcription (Whisper) with visual frame understanding (Qwen3-VL via an OpenAI-compatible API).

**Key Features:**

- **Perceptual Hashing (pHash)**: Deduplicates visually identical frames, reducing API calls by 60-80% on typical screen recordings.
- **Content-Addressable Cache (CAS)**: Never re-process the same video twice; cache keys are based on file SHA-256 hashes.
- **Distributed Parallel Inference**: Load-balance frame processing across multiple LLM server instances.
- **Anti-Hallucination Whisper Fixes**: Prevents the "Yeah Yeah Yeah" loops during silent sections.
- **Long-form Timestamps**: Handles videos longer than 1 hour correctly.

## What's here

### Two pipelines for different use cases

- **`analyze_pipeline.py`** (Recommended for most users)
  - Fast parallel frame analysis with perceptual hashing (pHash)
  - Distributed inference across multiple LLM servers
  - Real-time diagnostic tuning with `--tune-phash`
  - Best for: quick iteration, UI recordings, cost optimization

- **`analyze_video_pipeline_full.py`** (For advanced/backend use)
  - Fully documented, deterministic pipeline with adaptive dHash frame selection
  - Per-step idempotent caching with binary content fingerprints
  - Comprehensive logging and frame selection tracking
  - Best for: production workflows, reproducibility, parameter research

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
  - `pandas` (for notebook)
  - `matplotlib` (for notebook)
  - `seaborn` (for notebook)

## Install

```bash
pip install -U openai-whisper requests imagehash Pillow pandas matplotlib seaborn
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

### Distributed parallel inference

```bash
python analyze_pipeline.py --video path/to/video.mp4 --api-urls http://127.0.0.1:8033/v1/chat/completions http://127.0.0.1:8034/v1/chat/completions
```

### Tune pHash threshold

Run a diagnostic to see frame distances:

```bash
python analyze_pipeline.py --video path/to/video.mp4 --tune-phash
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

### Clear cache

```bash
python analyze_pipeline.py --video path/to/video.mp4 --clear-cache
```

---

## Run — `analyze_video_pipeline_full.py` (Advanced)

Fully documented, idempotent backend pipeline with comprehensive parameter control.

### Basic run

```bash
python analyze_video_pipeline_full.py
```

### With VLM analysis

```bash
python analyze_video_pipeline_full.py --do_vlm
```

### Custom tuning

```bash
python analyze_video_pipeline_full.py \
  --fps 1 \
  --change_threshold 6 \
  --burst_window 3 \
  --whisper_model small \
  --do_vlm
```

**dHash threshold tuning:**

- `--change_threshold 4`: More sensitive, more frames
- `--change_threshold 6`: Default, balanced
- `--change_threshold 8+`: Less sensitive, fewer API calls

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
- `pipeline_execution.log` — execution log
- `debug_summary_<hash>.md` — final human-readable summary

### `analyze_video_pipeline_full.py`

- `.pipeline_cache/` — binary content-based cache
- `.pipeline_runs/<timestamp>/` — per-run logs
- `frame_selection.json` — adaptive frame selection report
- `frames_analysis.jsonl` — frame-by-frame VLM results

## Troubleshooting

- `ffmpeg` not found: confirm it's on PATH (`ffmpeg -version`). Reopen the terminal after install.
- Whisper downloads are slow: ensure a stable network and try a smaller model (e.g., `base`).
- LLM endpoint errors: verify the `--api-url` is reachable and the model name is `qwen3-vl`.
- Missing `imagehash` or `Pillow`: Install them with `pip install imagehash Pillow`.

## Which pipeline should I use?

| Feature               | `analyze_pipeline.py`        | `analyze_video_pipeline_full.py`      |
| --------------------- | ---------------------------- | ------------------------------------- |
| Speed                 | ⚡ Fast (parallel)           | 🔧 Flexible                           |
| Ease of use           | ✅ Recommended               | Advanced                              |
| Frame filtering       | pHash                        | dHash                                 |
| Tuning                | `--tune-phash`               | parameters                            |
| Distributed inference | ✅ `--api-urls`              | Single server                         |
| Caching               | Per-file                     | Per-step                              |
| Best for              | UI/terminal recordings, cost | Production, research, reproducibility |

## Notes

- Both pipelines expect `qwen3-vl` model when using VLM.
- Content-based caching: videos identified by SHA-256 hash, not filename.
- pHash (`analyze_pipeline.py`) can reduce API calls by 60-80%.
- dHash (`analyze_video_pipeline_full.py`) provides detailed frame selection logs.
- **Adaptive Spike Mode**: Enables burst frame capture after changes, best for terminal logs and scrolling content
- **Threshold recommendations**:
  - Silent/mostly static videos: `--phash-threshold 6-8` (fewer API calls)
  - UI/terminal with moderate activity: `--phash-threshold 5` (default, balanced)
  - Fast scrolling/coding sessions: `--phash-threshold 3-4` with `--adaptive-spike` (more frames)
