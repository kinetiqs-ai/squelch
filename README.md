# Squelch — Voice Pipeline for Squad Goals

Fork of [pipecat-ai/nemotron-january-2026](https://github.com/pipecat-ai/nemotron-january-2026) adapted for A100 (Ampere) hardware on ramp-02. This repo is the foundation for the Squad Goals voice sidecar service (Issue #196).

## Current State

**TTS replaced: Magpie → Orpheus.** The original Magpie TTS (NVIDIA, 357M params, NeMo-based) had audible pop/click artifacts from its non-causal HiFi-GAN vocoder. Replaced with Orpheus TTS (Canopy Labs, 3B params, Llama 3.2 derivative) which generates SNAC codec tokens decoded to clean 24kHz audio.

### Architecture

| Port | Service | Model |
|------|---------|-------|
| 8080 | ASR | NeMo Parakeet (streaming WebSocket) |
| 8001 | TTS | Orpheus 3B via vLLM + SNAC decoder |
| 8000 | LLM | Nemotron-3-Nano (llama.cpp or vLLM) |

### What Works

- **Dockerfile.unified-ampere** — CUDA 12.6, pre-built PyTorch/vLLM wheels, sm_80 llama.cpp, SNAC decoder
- **Orpheus TTS** — 8 voices (tara, leah, jess, leo, dan, mia, zac, zoe), emotion tags, 24kHz output, SNAC logit processor for token validity
- **All three model services** healthy simultaneously. Estimated VRAM: ~37 GB / 81 GB
- **Tool calling** via `bot_tools_test.py` using `OpenAILLMService` pointed at llama.cpp
- **Barge-in** via PipeCat core (SileroVAD + SmartTurn)
- **WebRTC** accessible over Tailscale HTTPS

### Known Issues

- **llama-server must be built in-container.** Docker build has no GPU access.
- **LLM startup takes ~90s.** Q4 30B model load requires `SERVICE_TIMEOUT=300`.

## Quick Start (ramp-02)

### Build

```bash
docker build -f Dockerfile.unified-ampere -t nemotron-unified:ampere .
```

### Start

```bash
NEMOTRON_IMAGE=nemotron-unified:ampere SERVICE_TIMEOUT=300 ./scripts/nemotron.sh start
```

### Run the voice bot

```bash
# Inside the container
cd /workspace && uv run pipecat_bots/bot_interleaved_streaming.py -t webrtc --host 0.0.0.0
```

Open `https://ramp-02.tail314cde.ts.net:7860/client` in a browser on the Tailscale network.

### Tool calling test

```bash
cd /workspace && uv run pipecat_bots/bot_tools_test.py --host 0.0.0.0
```

Open `https://ramp-02.tail314cde.ts.net:7861/client` and ask "what time is it?"

## Our Additions

| File | Purpose |
|------|---------|
| `Dockerfile.unified-ampere` | A100 adaptation: CUDA 12.6, pre-built wheels, sm_80 llama.cpp, SNAC decoder |
| `src/nemotron_speech/orpheus_tts_server.py` | Orpheus TTS server: vLLM + SNAC decoder, HTTP streaming API |
| `pipecat_bots/orpheus_http_tts.py` | PipeCat TTS adapter for Orpheus HTTP streaming |
| `pipecat_bots/bot_tools_test.py` | Tool calling validation bot (OpenAILLMService + get_current_time) |

## Next Steps (Issue #196)

This repo becomes the voice sidecar service for Squad Goals. Key decisions for #196:

- **LLM:** Replace Nemotron-3-Nano with the Squad Goals agent LLM (Anthropic API via envelope dispatch)
- **Transport:** WebRTC for browser, possibly Twilio for telephony
- **Integration:** PipeCat pipeline receives audio, dispatches to the agent system, speaks responses

## Upstream

- **origin:** `kinetiqs-ai/squelch`
- **upstream:** `pipecat-ai/nemotron-january-2026`

Pull upstream updates with:
```bash
git fetch upstream
git merge upstream/main
```

## Hardware

| Component | Spec |
|-----------|------|
| GPU | NVIDIA A100 80GB PCIe (Ampere, sm_80) |
| RAM | 216 GB |
| OS | Ubuntu 24.04 |
| CUDA | 12.6 (driver 580.126.09) |
| Docker | 28.2.2 |
| Network | Tailscale |
