# Squelch — Voice Pipeline for Squad Goals

Fork of [pipecat-ai/nemotron-january-2026](https://github.com/pipecat-ai/nemotron-january-2026) adapted for A100 (Ampere) hardware on ramp-02. This repo is the foundation for the Squad Goals voice sidecar service (Issue #196).

## Current State (v0.1.0)

**Spike validation complete.** PipeCat reference implementation runs on ramp-02's A100 80GB with competitive latency. Verdict: GO for PipeCat as voice pipeline framework.

### What Works

- **Dockerfile.unified-ampere** — builds on CUDA 12.6 with pre-built PyTorch/vLLM wheels + llama.cpp from source for sm_80. ~60 min build (vs 2-3 hrs for Blackwell source builds).
- **All three model services** healthy simultaneously: ASR (port 8080), TTS (port 8001), LLM (port 8000). Total VRAM: 29 GB / 81 GB (35.5%).
- **V2V latency** estimated 470-630ms (ASR 160ms + LLM 56ms + TTS 254ms). Well under 1s target.
- **Tool calling** via `bot_tools_test.py` using `OpenAILLMService` pointed at llama.cpp's OpenAI endpoint. `LlamaCppBufferedLLMService` has no tool calling support.
- **Barge-in** via PipeCat core (SileroVAD + SmartTurn). No custom code in the interrupt path.
- **WebRTC** accessible over Tailscale HTTPS at `https://ramp-02.tail314cde.ts.net:7860/client`

### Known Issues

- **TTS audio pops at chunk boundaries.** Magpie TTS uses a non-causal HiFi-GAN vocoder that generates inconsistent waveforms in streaming mode. This is a documented model limitation, not a PipeCat problem. The overlap-add crossfade mitigates but doesn't eliminate it. See [Daily.co blog post](https://www.daily.co/blog/building-voice-agents-with-nvidia-open-models/) for the authors' acknowledgment.
- **NeMo HindiCharsTokenizer patch required.** The pinned NeMo commit (644201898, Dec 2025) predates the `HindiCharsTokenizer` class needed by Magpie TTS. Patched via `docker commit` at runtime. A NeMo commit post-Jan-27-2026 includes it natively.
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
| `Dockerfile.unified-ampere` | A100 adaptation: CUDA 12.6, pre-built wheels, sm_80 llama.cpp, NeMo patches |
| `pipecat_bots/bot_tools_test.py` | Tool calling validation bot (OpenAILLMService + get_current_time) |

Everything else is from the upstream reference implementation ([pipecat-ai/nemotron-january-2026](https://github.com/pipecat-ai/nemotron-january-2026)).

## Next Steps (Issue #196)

This repo becomes the voice sidecar service for Squad Goals. Key decisions for #196:

- **TTS model:** Replace Magpie with Chatterbox (or another model without streaming vocoder artifacts)
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
