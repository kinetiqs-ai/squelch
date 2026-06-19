# Squelch

Squelch is the local voice stack for `ramp-genesis`, an NVIDIA Jetson AGX Thor
Developer Kit.

The repository now tracks only the final Thor-local product path:

```text
audio-edge-agent
  -> native WebSocket audio ingress on Thor
  -> WebRTC VAD gate
  -> NVIDIA Riva ASR
  -> llama.cpp Nemotron-3-Nano-30B-A3B Q4_K_M
  -> NVIDIA Magpie TTS
  -> return audio to audio-edge-agent
```

## Runtime Services

The live deployment uses separate local services for the model endpoints:

| Service | Local endpoint | Purpose |
| --- | --- | --- |
| Native audio ingress | `http://127.0.0.1:7860/health` | Receives edge-agent audio, runs VAD, coordinates turns, returns assistant audio |
| Magpie TTS | `http://127.0.0.1:8101/health` | NVIDIA Magpie speech synthesis |
| LLM | `http://127.0.0.1:8100` | llama.cpp Nemotron-3-Nano-30B-A3B Q4_K_M |
| Riva ASR | gRPC on `50051` | NVIDIA Riva speech recognition |

The external Tailscale endpoint is:

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

## Source Layout

| Area | Files |
| --- | --- |
| Native ingress | `native_voice/audio_ingress.py`, `native_voice/riva_asr_app.py` |
| Turn orchestration | `native_voice/orchestrator.py` |
| Riva ASR helpers | `native_voice/riva_pipeline.py` |
| Diagnostics | `native_voice/diagnostics.py` |
| Magpie TTS server | `src/nemotron_speech/tts_server.py` |
| Magpie streaming presets | `src/nemotron_speech/streaming_tts.py` |
| Magpie stream state | `src/nemotron_speech/adaptive_stream.py` |
| Native ingress start script | `scripts/start_native_audio_ingress.sh` |
| Voice agent manager | `scripts/voice_agent.sh` |

## Model Decisions

ASR is Riva only. Audio is gated by WebRTC VAD and RMS before it is sent to ASR;
silence and noise sent directly to ASR caused hallucinated transcripts.

The LLM is llama.cpp serving Nemotron-3-Nano-30B-A3B Q4_K_M at
`http://127.0.0.1:8100`.

TTS is Magpie only. The first Magpie segment of each assistant response uses the
`startup_quality` streaming preset, and follow-on segments use `quality`.
`SQUELCH_STREAM_LLM_TO_TTS` defaults to `0` because token streaming back into TTS
did not improve perceived latency and reintroduced the first-word artifact.

## Start Native Ingress

The model services should already be running locally before starting ingress.

```bash
cd /home/ramp-genesis/squelch
./scripts/start_native_audio_ingress.sh
```

For normal operation, use the voice agent manager instead:

```bash
./scripts/voice_agent.sh start
./scripts/voice_agent.sh status
./scripts/voice_agent.sh stop
```

Health check:

```bash
curl -fsS http://127.0.0.1:7860/health
```

## Magpie TTS Check

```bash
curl -fsS http://127.0.0.1:8101/health
curl -sS -D /tmp/magpie.headers \
  -o /tmp/magpie-test.pcm \
  -H 'Content-Type: application/json' \
  -d '{"input":"This is a short Magpie synthesis test.","voice":"aria","language":"en","response_format":"pcm"}' \
  http://127.0.0.1:8101/v1/audio/speech
cat /tmp/magpie.headers
```

## Validation

Focused repository checks:

```bash
bash -n scripts/start_native_audio_ingress.sh
bash -n scripts/voice_agent.sh
python -m compileall native_voice src/nemotron_speech
curl -fsS http://127.0.0.1:7860/health
curl -fsS http://127.0.0.1:8101/health
tailscale serve status
```

Useful runtime process checks:

```bash
pgrep -af "native_voice.riva_asr_app|uv run --no-project --with fastapi"
docker ps --format 'table {{.Names}}\t{{.Status}}\t{{.Ports}}'
```

## Local State

Do not commit:

- `.env.local`
- Hugging Face tokens
- Tailscale credentials
- generated diagnostics
- model files or model cache paths

Canonical local checkout:

```text
/home/ramp-genesis/squelch
```
