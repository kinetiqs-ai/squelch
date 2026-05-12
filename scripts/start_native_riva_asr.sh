#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

exec uv run \
  --no-project \
  --with fastapi \
  --with "uvicorn[standard]" \
  --with nvidia-riva-client \
  --with "audio-edge-agent @ file:///home/ramp-genesis/audio-edge-agent" \
  python -m native_voice.riva_asr_app --host 127.0.0.1 --port 7860
