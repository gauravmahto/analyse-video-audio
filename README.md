# Video Multimodal Pipeline

Analyze screen-recorded videos by combining audio transcription (Whisper) with visual frame understanding (Qwen3-VL via an OpenAI-compatible API).

## What's here

- `analyze_pipeline.py`: main multimodal pipeline with content-hash cache and a final synthesis step.
- `analyze-v2.py`: more advanced, fully idempotent pipeline with per-step caching and richer logging.

## Requirements

- Python 3.10+
- `ffmpeg` on PATH
- Python packages:
  - `openai-whisper`
  - `requests`

## Install

```bash
pip install -U openai-whisper requests
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

## Run

### Multimodal analysis (audio + frames)

```bash
python analyze_pipeline.py --video path/to/video.mp4
```

### Audio-only (skip visual frames)

```bash
python analyze_pipeline.py --video path/to/video.mp4 --audio-only
```

### Force audio extraction format

```bash
python analyze_pipeline.py --video path/to/video.mkv --audio-type wav
```

### Custom API endpoint

```bash
python analyze_pipeline.py --video path/to/video.mp4 --api-url http://127.0.0.1:8033/v1/chat/completions
```

### Clear cache for a file

```bash
python analyze_pipeline.py --video path/to/video.mp4 --clear-cache
```

### analyze-v2.py examples

```bash
python analyze-v2.py --video path/to/video.mp4 --fps 1 --whisper_model small
```

```bash
python analyze-v2.py --video path/to/video.mp4 --fps 1 --do_vlm --frame_limit 50
```

## Outputs

- Cached artifacts in `.pipeline_cache/<video_hash>/`
- `pipeline_execution.log`
- Human-readable summary: `debug_summary_<hash>.md`

## Troubleshooting

- `ffmpeg` not found: confirm it's on PATH (`ffmpeg -version`). Reopen the terminal after install.
- Whisper downloads are slow: ensure a stable network and try a smaller model (e.g., `base`).
- LLM endpoint errors: verify the `--api-url` is reachable and the model name is `qwen3-vl`.

## Notes

- The visual pipeline sends frames to a VLM using the OpenAI-compatible endpoint in `--api-url` and expects a model named `qwen3-vl`.
- `analyze-v2.py` defaults to the same `video1822201159.mp4` and uses `.pipeline_cache` plus `.pipeline_runs` for per-run logs.
