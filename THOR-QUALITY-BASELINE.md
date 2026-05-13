# Thor Quality Baseline

This branch starts from `v0.1.0`, the original NVIDIA ASR plus Magpie TTS baseline, and applies only the changes needed to validate it cleanly on `ramp-genesis` / Jetson Thor.

## Goal

Prove the local voice components before optimizing the full conversation loop:

1. Nemotron/Parakeet streaming ASR on port `8080`
2. Magpie batch TTS on port `8001`
3. Browser WebRTC bot on port `7860`
4. LLM only after ASR and TTS are stable

Voxtral and Orpheus are intentionally out of this baseline.

## Key Defaults

- `TTS_BACKEND=magpie_http`
- `VAD_STOP_SECS=0.34`
- `VAD_START_SECS=0.12`
- `USE_SMART_TURN=false`
- `MAGPIE_SAMPLE_RATE=22050`
- WebRTC bind host `0.0.0.0` for the Tailscale path

The `0.34` second VAD stop default comes from `docs/stt-truncation-debugging.md`: the ASR needs enough trailing context to avoid dropping final words.

The `0.12` second VAD start default is a compromise. Pipecat's original `0.20` clipped fast second prompts, while `0.05` caused false starts and split utterances during testing.

## Start Services

From this worktree:

```bash
cd /home/ramp-genesis/squelch-v010-thor
sg docker -c 'docker build -f Dockerfile.thor-nemo-baseline -t squelch-v010-nemo:thor .'
sg docker -c 'SERVICE_TIMEOUT=300 NEMOTRON_IMAGE=squelch-v010-nemo:thor ./scripts/nemotron.sh start --no-llm'
```

For full local LLM testing after ASR and TTS pass:

```bash
cd /home/ramp-genesis/squelch-v010-thor
sg docker -c 'SERVICE_TIMEOUT=300 NEMOTRON_IMAGE=squelch-v010-nemo:thor ./scripts/nemotron.sh start --mode llamacpp-q4'
```

## Start Browser Bot

```bash
cd /home/ramp-genesis/squelch-v010-thor
sg docker -c './scripts/nemotron.sh bot --host 0.0.0.0 --port 7860'
```

External test URL:

```text
https://ramp-genesis.tail314cde.ts.net:7860/client/
```

## Component Checks

TTS health:

```bash
curl -sf http://localhost:8001/health
```

TTS batch latency and RTF:

```bash
curl -sS -D /tmp/magpie.headers \
  -o /tmp/magpie-test.pcm \
  -H 'Content-Type: application/json' \
  -d '{"input":"This is a short Magpie batch synthesis test.","voice":"aria","language":"en","response_format":"pcm"}' \
  http://localhost:8001/v1/audio/speech
cat /tmp/magpie.headers
```

Status:

```bash
sg docker -c './scripts/nemotron.sh status'
```

Logs:

```bash
sg docker -c './scripts/nemotron.sh logs all'
```

## Acceptance Bar

- ASR returns complete short utterances, including final words.
- Magpie TTS is smooth with no boundary pops.
- Magpie batch RTF is under `1.0` when ASR and TTS are loaded without the local LLM.
- If full-stack latency regresses materially with the local LLM loaded, split the LLM off-device instead of changing ASR/TTS quality defaults.
