# Squelch

Squelch is the local voice stack for `ramp-genesis`, an NVIDIA Jetson AGX Thor
Developer Kit. It is a deployment fork of the original
`pipecat-ai/nemotron-january-2026` work, but the validated path on Thor is now a
native audio-edge-agent pipeline rather than the browser/Pipecat reference path.

The goal is a local, high-accuracy, low-latency voice assistant that can run with
open or commercially usable models and without Chinese model dependencies.

## Current Primary Path

The active path is:

```text
audio-edge-agent
  -> WebSocket audio ingress on Thor
  -> WebRTC VAD gate
  -> NVIDIA Riva / Nemotron ASR
  -> llama.cpp Nemotron LLM
  -> NVIDIA Magpie TTS
  -> return audio to audio-edge-agent
```

The primary server entry point is:

```text
native_voice/riva_asr_app.py
```

The turn orchestrator is:

```text
native_voice/orchestrator.py
```

The currently validated external endpoint is:

```text
wss://ramp-genesis.tail314cde.ts.net:7860/ws/audio-ingress
```

Run the edge client from the Mac or another audio edge device with:

```bash
audio-edge-agent run \
  --ws-url "wss://ramp-genesis.tail314cde.ts.net:7860/ws/audio-ingress" \
  --show-transcripts \
  --show-agent-events
```

## Current Runtime Services

The live Thor deployment uses separate services for the heavy model endpoints:

| Service | Local endpoint | Purpose |
| --- | --- | --- |
| Native audio ingress | `127.0.0.1:7860` | Receives edge-agent audio, runs VAD, sends ASR events and return audio |
| Magpie TTS | `127.0.0.1:8101` -> container `8001` | NVIDIA Magpie speech synthesis |
| LLM | `127.0.0.1:8100` -> container `8000` | llama.cpp Nemotron-3-Nano-30B-A3B Q4_K_M |
| Riva ASR | `50051` gRPC | NVIDIA/Riva speech recognition backend |

The Tailscale Serve mapping exposes local `7860` to:

```text
https://ramp-genesis.tail314cde.ts.net:7860
```

## Important Model Decisions

### ASR

We tested multiple ASR paths. The current native path uses NVIDIA/Riva ASR
through `native_voice/audio_ingress.py` and `native_voice/riva_pipeline.py`.

Key lessons:

- Browser capture was not a good enough deployment proxy.
- `audio-edge-agent` gives a deployment-shaped audio ingress path and better
  control over microphone selection, framing, transcripts, and return audio.
- Thor-side VAD is required before ASR. Sending silence/noise into ASR caused
  hallucinated interim and occasional final transcripts.
- The VAD gate now includes preroll and end-of-segment handling so first words
  and final words are preserved.

Voxtral remains in the repo as an experimental ASR backend for the Pipecat path,
but it is not the active native path. Chinese ASR models are out of scope for
this project.

### TTS

We moved away from Orpheus for the Thor-local path after measuring unacceptable
real-time factor on this hardware. Orpheus remains in the repo because it has
important custom-voice capabilities, but it is not the current validated Thor
runtime.

The active TTS target is NVIDIA Magpie.

Key lessons:

- Magpie quality is good enough for the current target when driven correctly.
- The first audio chunk after each assistant turn had an audible artifact when
  released too early.
- The fix is the `startup_quality` streaming preset in
  `src/nemotron_speech/streaming_tts.py`.
- The first segment of each response uses `startup_quality`; later segments use
  `quality`.
- Keeping a persistent Magpie websocket did not fix the artifact and introduced
  second-turn reliability issues, so it is not the default.
- Streaming LLM text into TTS did not materially improve perceived latency and
  reintroduced the first-word artifact, so it is opt-in only.

Relevant defaults:

| Setting | Default | Purpose |
| --- | --- | --- |
| `SQUELCH_TTS_MODE` | `stream` | Use Magpie websocket streaming |
| `SQUELCH_TTS_START_STREAM_PRESET` | `startup_quality` | First segment, artifact avoidance |
| `SQUELCH_TTS_STREAM_PRESET` | `quality` | Follow-on segments |
| `SQUELCH_STREAM_LLM_TO_TTS` | `0` | Keep LLM-to-TTS token streaming disabled by default |
| `SQUELCH_TTS_VOICE` | `aria` | Current Magpie voice |

### LLM

The local LLM endpoint is llama.cpp running Nemotron-3-Nano-30B-A3B Q4_K_M.

The native orchestrator currently waits for the full LLM response before sending
text to TTS. This is deliberate: live testing found little perceived latency
benefit from token-streaming into TTS, and it made the first spoken word less
stable. The streaming code remains available behind:

```bash
SQUELCH_STREAM_LLM_TO_TTS=1
```

Do not enable that by default without retesting first-audio quality.

## Native Audio Ingress

The native ingress implements the `audio-edge-agent` protocol:

- receives 16 kHz mono `s16le` audio frames
- validates protocol headers
- applies WebRTC VAD and RMS gating
- streams only detected speech segments to Riva
- emits ASR interim/final events back to the edge agent
- invokes the local LLM/TTS orchestrator on finalized user utterances
- streams assistant audio frames back over the same websocket

Start it with:

```bash
cd /home/ramp-genesis/squelch
./scripts/start_native_audio_ingress.sh
```

Health check:

```bash
curl -fsS http://127.0.0.1:7860/health
```

## Magpie TTS Service

The Magpie server is implemented in:

```text
src/nemotron_speech/tts_server.py
```

The streaming implementation and presets live in:

```text
src/nemotron_speech/streaming_tts.py
```

Health check:

```bash
curl -fsS http://127.0.0.1:8101/health
```

Basic TTS request:

```bash
curl -sS -D /tmp/magpie.headers \
  -o /tmp/magpie-test.pcm \
  -H 'Content-Type: application/json' \
  -d '{"input":"This is a short Magpie synthesis test.","voice":"aria","language":"en","response_format":"pcm"}' \
  http://127.0.0.1:8101/v1/audio/speech
cat /tmp/magpie.headers
```

## Pipecat Path

The Pipecat browser pipeline still exists, but it is not the primary validated
deployment path on Thor.

Main bot:

```text
pipecat_bots/bot_interleaved_streaming.py
```

Container manager:

```text
scripts/nemotron.sh
```

TTS can be selected through:

```bash
TTS_BACKEND=orpheus_http
TTS_BACKEND=magpie_http
TTS_BACKEND=magpie_ws
```

The unified service script can start either Orpheus or Magpie:

```bash
TTS_ENGINE=orpheus
TTS_ENGINE=magpie
```

Use the Pipecat path for comparison and browser experiments, not as the default
deployment target unless it is revalidated.

## Docker Images

Important image files:

| File | Purpose |
| --- | --- |
| `Dockerfile.thor-nemo-baseline` | Thor NeMo/Magpie baseline image |
| `Dockerfile.unified-ampere` | Legacy A100/Ampere image from earlier deployment work |
| `Dockerfile.voxtral-asr` | Voxtral ASR experiment image |

For Thor work, do not assume x86, A100, CUDA 12.6, or `sm_80`. Thor is Arm64
SBSA with CUDA 13 and Blackwell GPU support.

## Diagnostics

Session diagnostics are written under:

```text
runs/audio-ingress/
```

Each session can include:

- transport events
- ASR events
- agent events
- audio captures
- summary metadata

Useful runtime checks:

```bash
curl -fsS http://127.0.0.1:7860/health
curl -fsS http://127.0.0.1:8101/health
tailscale serve status
```

Useful process checks:

```bash
pgrep -af "native_voice.riva_asr_app|uv run"
sudo docker ps --format 'table {{.Names}}\t{{.Status}}\t{{.Ports}}'
```

## Validation

Use focused checks for this repo. Many tests are model/hardware integration
tests and should not be run casually without confirming the needed services and
model cache state.

Basic validation:

```bash
bash -n scripts/start_native_audio_ingress.sh scripts/start_native_riva_asr.sh scripts/nemotron.sh scripts/start_unified.sh scripts/voxtral.sh scripts/start_voxtral_asr.sh scripts/start_asr_tts.sh
python -m compileall native_voice src/nemotron_speech pipecat_bots scripts/voice_agent_test_client.py scripts/compare_asr.py
curl -fsS http://127.0.0.1:7860/health
curl -fsS http://127.0.0.1:8101/health
```

If `compileall` fails with permissions under `__pycache__`, remove generated
cache directories. Some were previously created by root-owned container runs:

```bash
sudo rm -rf native_voice/__pycache__ src/nemotron_speech/__pycache__ src/nemotron_speech/modal/__pycache__ pipecat_bots/__pycache__ pipecat_bots/modal/__pycache__ scripts/__pycache__
```

## Current Repo State

Canonical local checkout:

```text
/home/ramp-genesis/squelch
```

There should be one working tree for normal development. The previous
`/home/ramp-genesis/squelch-v010-thor` worktree was removed after PR merge to
avoid divergent local repo state.

There is a safety branch preserving old local work before the cleanup:

```text
backup/local-main-pre-sync-20260513-164102
```

Do not base new work on that branch unless intentionally recovering something
from the pre-sync state.

## Secrets And Local State

Do not commit:

- Hugging Face tokens
- Tailscale credentials
- `.env.local`
- model cache paths that are private to a machine
- generated diagnostics
- GGUF model files

The `.gitignore` is set up for the common generated outputs.
