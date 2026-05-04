# Squelch — Voice Pipeline for Squad Goals

Fork of [pipecat-ai/nemotron-january-2026](https://github.com/pipecat-ai/nemotron-january-2026) adapted for A100 (Ampere) hardware on ramp-02. This repo is the foundation for the Squad Goals voice sidecar service (Issue #196).

## Current State

This checkout started from the working `v0.1.0` PipeCat reference deployment and replaces Magpie TTS with Orpheus TTS while keeping the ASR, LLM, WebRTC, VAD, and interruption pipeline intact.

### What Works

- **Dockerfile.unified-ampere** — builds on CUDA 12.6 with pre-built PyTorch/vLLM wheels + llama.cpp from source for sm_80. ~60 min build (vs 2-3 hrs for Blackwell source builds).
- **All three model services** run locally: ASR (port 8080), Orpheus TTS (port 8001), LLM (port 8000).
- **Orpheus TTS** outputs 24kHz mono PCM from `canopylabs/orpheus-3b-0.1-ft` via vLLM + SNAC.
- **Tool calling** via `bot_tools_test.py` using `OpenAILLMService` pointed at llama.cpp's OpenAI endpoint. `LlamaCppBufferedLLMService` has no tool calling support.
- **Barge-in** via PipeCat core (SileroVAD + SmartTurn). No custom code in the interrupt path.
- **WebRTC** accessible over Tailscale HTTPS at `https://ramp-02.tail314cde.ts.net:7860/client`

### Known Issues

- **Orpheus requires Hugging Face model access.** The `canopylabs/orpheus-3b-0.1-ft` model requires accepting the Hugging Face conditions for the account used by `HUGGINGFACE_ACCESS_TOKEN`.
- **NeMo HindiCharsTokenizer patch retained.** The pinned NeMo commit (644201898, Dec 2025) predates the `HindiCharsTokenizer` class used by the legacy Magpie path. The Dockerfile patches it during build.
- **llama-server must be built in-container.** Docker build has no GPU access, so the binary must be compiled after `docker run --gpus all`. The Dockerfile verifies with `which llama-server` (hard fail, no `|| echo`).
- **LLM startup takes ~90s.** Q4 30B model load requires `SERVICE_TIMEOUT=300`.

### Build Obstacles Resolved

Three non-trivial issues were hit and fixed during the spike. Full details in the [Squad Goals validation report](https://github.com/magnum6actual/squad-goals/blob/issue-197-pipecat-reference-deploy/docs/spike/issue-197-validation-report.md):

1. **HindiCharsTokenizer vocab mismatch** — 6+ iterations to identify correct tokenizer config (`case="mixed"` + `ascii_lowercase` = 191 tokens matching checkpoint)
2. **llama-server missing from image** — `|| echo` fallback masked the failure; rebuilt in-container with GPU
3. **SERVICE_TIMEOUT too short** — Q4 30B needs ~90s to load; default 60s caused abort

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
./scripts/nemotron.sh bot
```

Open `http://localhost:7860/client` locally, or `https://ramp-02.tail314cde.ts.net:7860/client` from the Tailscale network.

This is Pipecat's default WebRTC reference client served by the runner. It uses the bot's `/api/offer` endpoint and microphone access from the browser.

### Tool calling test

```bash
./scripts/nemotron.sh bot tools
```

Open `http://localhost:7861/client` locally, or `https://ramp-02.tail314cde.ts.net:7861/client` from the Tailscale network. Ask "what time is it?"

### Experimental Voxtral ASR

Voxtral runs as a separate experimental ASR service so the released Nemotron ASR path remains the default.

```bash
./scripts/voxtral.sh build
./scripts/voxtral.sh start
./scripts/nemotron.sh bot --asr voxtral --port 7861
```

Open `https://ramp-02.tail314cde.ts.net:7860/client` when Tailscale Serve is proxying to local port `7861`.

Compare both ASR backends on the same WAV:

```bash
uv run python scripts/compare_asr.py recordings/example.wav \
  --reference "expected transcript"
```

If the host Python environment is not synced, run the comparison from the Voxtral container:

```bash
docker exec voxtral-asr bash -lc \
  'cd /workspace && python3 scripts/compare_asr.py tests/fixtures/harvard_16k.wav --backend voxtral'
```

Defaults:

| Setting | Default |
|---------|---------|
| `ASR_BACKEND` | `nemotron` |
| `NVIDIA_ASR_URL` | `ws://localhost:8080` |
| `VOXTRAL_ASR_URL` | `ws://host.docker.internal:8082/v1/realtime` in bots |
| Voxtral host port | `8082` |
| `VOXTRAL_GPU_MEMORY_UTILIZATION` | `0.35` |
| `VOXTRAL_MAX_MODEL_LEN` | `32768` |

## Our Additions

| File | Purpose |
|------|---------|
| `Dockerfile.unified-ampere` | A100 adaptation: CUDA 12.6, pre-built wheels, sm_80 llama.cpp, NeMo patches |
| `src/nemotron_speech/orpheus_tts_server.py` | Local Orpheus TTS HTTP streaming server |
| `pipecat_bots/orpheus_http_tts.py` | PipeCat TTS adapter preserving LLM/TTS backpressure |
| `pipecat_bots/bot_tools_test.py` | Tool calling validation bot (OpenAILLMService + get_current_time) |
| `Dockerfile.voxtral-asr` | Experimental isolated Voxtral Realtime ASR image |
| `pipecat_bots/voxtral_stt.py` | PipeCat STT adapter for vLLM Realtime Voxtral |
| `scripts/compare_asr.py` | A/B test utility for Nemotron vs Voxtral ASR |

Everything else is from the upstream reference implementation ([pipecat-ai/nemotron-january-2026](https://github.com/pipecat-ai/nemotron-january-2026)).

## Next Steps (Issue #196)

This repo becomes the voice sidecar service for Squad Goals. Key decisions for #196:

- **TTS model:** Orpheus is now the default local TTS engine; tune latency and voice quality next.
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
