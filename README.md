# Squelch - Local Pipecat Voice Pipeline

Squelch is a deployment fork of the
[pipecat-ai/nemotron-january-2026](https://github.com/pipecat-ai/nemotron-january-2026)
reference implementation. It is currently used as the local voice sidecar
prototype for Squad Goals.

This repo intentionally starts from a known-good release lineage rather than
tracking upstream `main`. The newer upstream/main work was not usable for this
deployment when this branch was created.

## Current Implementation

The current implementation is a local, GPU-backed voice pipeline built around
Pipecat. The browser connects through Pipecat's reference WebRTC test client,
the bot receives microphone audio, ASR turns speech into text, the LLM produces
a response, and Orpheus turns the response back into streamed audio.

High-level flow:

```text
Browser WebRTC client
  -> Pipecat bot runner
  -> VAD / turn detection
  -> ASR backend
  -> LLM backend
  -> Orpheus TTS backend
  -> Browser audio playback
```

### Major Components

| Component | Current implementation | Purpose |
|-----------|------------------------|---------|
| Transport | Pipecat WebRTC runner and reference browser client | Browser microphone capture and audio playback |
| Bot pipeline | `pipecat_bots/bot_interleaved_streaming.py` | Main conversational pipeline with VAD, ASR, LLM, TTS, and interruption handling |
| VAD / turns | Silero VAD and Pipecat SmartTurn path | Detects speech boundaries and supports barge-in |
| ASR default | NVIDIA Nemotron streaming ASR on port `8080` | Original reference ASR path; still available |
| ASR alternate | Voxtral Mini 4B Realtime through vLLM on port `8082` | New experimental ASR path; currently preferred after microphone testing |
| LLM | Nemotron 30B through llama.cpp OpenAI-compatible endpoint on port `8000` | Local reasoning / response generation |
| TTS | Orpheus 3B through local HTTP streaming service on port `8001` | Replaces the original Magpie TTS path |
| Diagnostics | `pipecat_bots/asr_audio_diagnostics.py` and `scripts/compare_asr.py` | Captures turn audio and compares ASR backends on the same WAV |
| Container manager | `scripts/nemotron.sh` | Starts/stops the unified ASR/TTS/LLM container and test bots |
| Voxtral manager | `scripts/voxtral.sh` | Builds and runs the separate Voxtral realtime ASR container |

### What Changed From Upstream

The upstream reference implementation used Magpie TTS. This fork replaces that
with Orpheus TTS while keeping the Pipecat browser runner, bot pipeline shape,
LLM endpoint style, VAD, interruption path, and local model-service approach.

Key local additions:

| File | Purpose |
|------|---------|
| `Dockerfile.unified-ampere` | A100-specific unified runtime image using CUDA 12.6 and pre-built PyTorch/vLLM wheels |
| `src/nemotron_speech/orpheus_tts_server.py` | Local Orpheus HTTP streaming TTS server |
| `pipecat_bots/orpheus_http_tts.py` | Pipecat TTS adapter for the local Orpheus service |
| `pipecat_bots/asr_factory.py` | Selects Nemotron or Voxtral ASR via `ASR_BACKEND` |
| `pipecat_bots/voxtral_stt.py` | Pipecat STT adapter for vLLM Voxtral Realtime |
| `Dockerfile.voxtral-asr` | Isolated Voxtral realtime ASR image |
| `scripts/start_voxtral_asr.sh` | Starts vLLM's realtime Voxtral server |
| `scripts/voxtral.sh` | Host-side Voxtral container manager |
| `scripts/compare_asr.py` | Offline ASR comparison utility |
| `pipecat_bots/asr_audio_diagnostics.py` | Optional turn-level WAV/JSON capture for ASR debugging |

## Runtime Services

The normal deployment uses one unified container for the original local services
and a second optional container for Voxtral ASR.

| Service | Container | Port | Notes |
|---------|-----------|------|-------|
| LLM | `nemotron` | `8000` | llama.cpp OpenAI-compatible endpoint |
| Orpheus TTS | `nemotron` | `8001` | HTTP streaming audio endpoint |
| Nemotron ASR | `nemotron` | `8080` | WebSocket ASR endpoint |
| Pipecat browser bot | `nemotron` | `7860` or `7861` | WebRTC runner and reference client |
| Voxtral ASR | `voxtral-asr` | `8082` | vLLM Realtime endpoint |

The browser test client is the default Pipecat reference client served by the
runner at `/client`.

## Quick Start On Current Host

Build the A100 image:

```bash
docker build -f Dockerfile.unified-ampere -t nemotron-unified:ampere .
```

Start the unified container:

```bash
NEMOTRON_IMAGE=nemotron-unified:ampere SERVICE_TIMEOUT=300 ./scripts/nemotron.sh start
```

Start the default bot:

```bash
./scripts/nemotron.sh bot
```

Open locally:

```text
http://localhost:7860/client
```

Open over Tailscale:

```text
https://ramp-02.tail314cde.ts.net:7860/client
```

### Run With Voxtral ASR

Voxtral is isolated from the unified container so the original Nemotron ASR path
remains available.

```bash
./scripts/voxtral.sh build
./scripts/voxtral.sh start
./scripts/nemotron.sh bot --asr voxtral --port 7861
```

When Tailscale Serve is proxying the public test URL to local port `7861`, open:

```text
https://ramp-02.tail314cde.ts.net:7860/client
```

### Compare ASR Backends

Compare Nemotron and Voxtral on the same WAV:

```bash
uv run python scripts/compare_asr.py recordings/example.wav \
  --reference "expected transcript"
```

If the host Python environment is not synced, run from the Voxtral container:

```bash
docker exec voxtral-asr bash -lc \
  'cd /workspace && python3 scripts/compare_asr.py tests/fixtures/harvard_16k.wav --backend voxtral'
```

## Configuration

Local machine configuration should live in an ignored env file such as
`.env.local`. Do not commit secrets, access tokens, model cache paths that are
specific to a private machine, or Tailscale auth material.

Source the file before running the host scripts:

```bash
set -a
source .env.local
set +a
```

Required values that must be supplied outside the repo:

| Variable name | Required for | Notes |
|---------------|--------------|-------|
| `HUGGINGFACE_ACCESS_TOKEN` | Hugging Face gated model downloads | Primary token name used by this repo |
| `HF_TOKEN` | Hugging Face gated model downloads | Should normally match `HUGGINGFACE_ACCESS_TOKEN` |
| `LLAMA_MODEL` | llama.cpp LLM mode, unless auto-detected | Local GGUF path on the target machine |
| `HF_HOME` | Optional model cache override | The scripts mount `~/.cache/huggingface` by default |
| Tailscale auth/config | Remote browser testing | Managed outside this repo by Tailscale |

Important runtime variables:

| Setting | Default | Purpose |
|---------|---------|---------|
| `NEMOTRON_IMAGE` | `nemotron-unified:ampere` | Unified container image |
| `SERVICE_TIMEOUT` | `60` unless overridden | Startup wait time; use `300` for large model loading |
| `ASR_BACKEND` | `nemotron` | `nemotron` or `voxtral` |
| `NVIDIA_ASR_URL` | `ws://localhost:8080` | Nemotron ASR WebSocket URL |
| `VOXTRAL_ASR_URL` | `ws://host.docker.internal:8082/v1/realtime` | Voxtral realtime ASR URL |
| `VOXTRAL_GPU_MEMORY_UTILIZATION` | `0.35` | vLLM memory fraction for Voxtral |
| `VOXTRAL_MAX_MODEL_LEN` | `32768` | vLLM context length for Voxtral |
| `ORPHEUS_MODEL` | `canopylabs/orpheus-3b-0.1-ft` | TTS model |
| `ORPHEUS_VOICE` | `tara` | TTS voice selected by bots |
| `ORPHEUS_GPU_MEMORY_UTILIZATION` | `0.25` | vLLM memory fraction for Orpheus |
| `HUGGINGFACE_ACCESS_TOKEN` | unset | Required for gated Hugging Face models |
| `HF_TOKEN` | unset | Alternate Hugging Face token variable used by some libraries |
| `HF_HUB_DISABLE_XET` | `1` | Avoids Xet-backed HF downloads in this environment |
| `ENABLE_ASR_DIAGNOSTICS` | `false` | Enables turn-level WAV/JSON capture |
| `ASR_DIAGNOSTICS_DIR` | `diagnostics/asr` | Diagnostic output directory |

Generated diagnostic captures, local env files, and test output are ignored by
git.

## Current Hardware: ramp-02 A100 VM

This repo currently targets `ramp-02`, an Ubuntu VM with an attached NVIDIA A100
80GB PCIe GPU. The current deployment image is specifically adapted for this
hardware.

Verified local hardware/runtime:

| Component | Current value |
|-----------|---------------|
| Host architecture | `x86_64` |
| CPU | AMD EPYC 7V13, 24 vCPU exposed to the VM |
| System RAM | 216 GiB |
| GPU | NVIDIA A100 80GB PCIe |
| GPU memory | 81920 MiB |
| CUDA compute capability | `8.0` / `sm_80` |
| NVIDIA driver | `580.126.09` |
| OS | Ubuntu 24.04 |
| Docker | 28.2.2, Linux `x86_64` |
| Network access | Tailscale, currently using `ramp-02.tail314cde.ts.net` |

### A100-Specific Requirements

The active unified runtime image is `Dockerfile.unified-ampere`.

That Dockerfile exists because the upstream/reference CUDA 13 Blackwell path was
not the right fit for the A100:

- It uses `nvidia/cuda:12.6.3-devel-ubuntu24.04`.
- It installs pre-built PyTorch, torchaudio, torchvision, and vLLM wheels instead
  of building large CUDA/PyTorch/vLLM stacks from source.
- It builds llama.cpp for Ampere `sm_80`.
- It keeps the NeMo runtime patch that detects the CUDA device capability at
  runtime.
- It removes Blackwell-specific vLLM patches that are not appropriate for A100.

The current A100 deployment expects:

- NVIDIA Container Toolkit working with Docker.
- `docker run --gpus all` support.
- Hugging Face model cache mounted at `~/.cache/huggingface`.
- A valid `HUGGINGFACE_ACCESS_TOKEN` for gated Orpheus/Voxtral access.
- Enough startup time for the local LLM; use `SERVICE_TIMEOUT=300`.
- Tailscale Serve configured to expose the active Pipecat runner port.

The current model split is conservative on A100 memory:

- Unified container handles LLM, Orpheus TTS, Nemotron ASR, and the bot process.
- Voxtral ASR runs in a separate vLLM container.
- Voxtral defaults to `VOXTRAL_GPU_MEMORY_UTILIZATION=0.35`.
- Orpheus defaults to `ORPHEUS_GPU_MEMORY_UTILIZATION=0.25`.

These values were tuned operationally for this host and should be treated as
deployment parameters, not model requirements.

## NVIDIA AGX Thor Assessment

This section assumes the target board is **NVIDIA Jetson AGX Thor** / Jetson
T5000-class hardware, not DRIVE AGX Thor. The application architecture should
port, but the container/runtime layer should not be assumed portable as-is.

Relevant public NVIDIA platform facts:

- Jetson AGX Thor uses an Arm Neoverse CPU and NVIDIA Blackwell GPU.
- The developer kit is an Arm64/SBSA platform.
- The Jetson T5000 configuration is listed with 128 GB LPDDR5X memory and
  273 GB/s memory bandwidth.
- JetPack 7 targets Jetson Thor, uses Ubuntu 24.04, and aligns Jetson Thor with
  the Arm SBSA software stack.
- JetPack 7 provides CUDA 13 support for Thor.
- NVIDIA documents containerized deployment on Jetson through NVIDIA Container
  Runtime.
- NVIDIA has announced Thor-oriented vLLM container support.

Sources:

- <https://www.nvidia.com/en-us/autonomous-machines/embedded-systems/jetson-thor/>
- <https://developer.nvidia.com/embedded/jetpack>
- <https://developer.nvidia.com/embedded/jetson-cloud-native>
- <https://developer.nvidia.com/blog/unlock-faster-smarter-edge-models-with-7x-gen-ai-performance-on-nvidia-jetson-agx-thor/>
- <https://docs.vllm.ai/en/latest/deployment/docker.html>

### Expected Porting Work

The Pipecat application code should need little or no structural change. The
likely work is below the application layer.

Required or likely changes:

- Create a Thor-specific image path instead of using `Dockerfile.unified-ampere`.
- Use Arm64/SBSA-compatible base images aligned with JetPack 7 and CUDA 13.
- Prefer NVIDIA's Thor-compatible vLLM container for Voxtral and possibly
  Orpheus rather than Docker Hub `vllm/vllm-openai:nightly`.
- Validate Arm64 wheels or source builds for PyTorch, torchaudio, vLLM, SNAC,
  NeMo-related dependencies, `soundfile`, `soxr`, and `librosa`.
- Replace any A100-specific `sm_80` compile settings with runtime detection or
  Thor/Blackwell-specific settings.
- Re-evaluate whether the older `Dockerfile.unified` is a useful starting point.
  It has Arm64/Blackwell intent, but it was not validated for Jetson AGX Thor and
  contains assumptions from a different Blackwell deployment path.
- Test whether Docker on the board prefers `--runtime nvidia` in addition to, or
  instead of, `--gpus all`.
- Consider `--network=host` for local model services to avoid Docker bridge and
  `host.docker.internal` differences on Jetson.
- Retune vLLM memory fractions for Voxtral, Orpheus, and the LLM.
- Set the board's power/performance mode appropriately before benchmarking.
- Pre-stage Hugging Face model weights on the board's NVMe storage.

### Capacity And Performance Expectations

Thor should have enough memory capacity for this pipeline. The 128 GB unified
memory pool is larger than the A100's 80 GB VRAM, and the current deployment
already runs with separate ASR/TTS/LLM services.

The unknown is latency. A100 has much higher GPU memory bandwidth than Jetson
AGX Thor's listed 273 GB/s LPDDR5X bandwidth. Thor has newer Blackwell tensor
features and strong edge inference support, but the pipeline should be
benchmarked rather than assumed faster.

Practical first target on Thor:

1. Bring up Voxtral ASR by itself using a Thor-compatible vLLM image.
2. Verify realtime ASR with `scripts/compare_asr.py` and browser audio.
3. Bring up Orpheus TTS by itself and validate streamed 24 kHz PCM output.
4. Bring up the LLM, starting with llama.cpp if vLLM introduces extra Arm64
   complexity.
5. Start the Pipecat browser bot and test full-duplex voice behavior.
6. Tune memory fractions, context lengths, and power mode based on measured
   latency and stability.

### What Should Not Change Initially

Do not change the Pipecat conversation semantics during the hardware port:

- Keep the same browser WebRTC client.
- Keep the same ASR/TTS/LLM service boundaries.
- Keep Voxtral isolated as an ASR service until Thor-specific stability is
  proven.
- Keep Orpheus as the TTS model.
- Keep the ASR diagnostics path available.

Changing the model pipeline while changing hardware would make debugging much
harder. The first Thor milestone should be functional equivalence with the A100
deployment.

## Known Issues And Constraints

- Orpheus requires accepting the Hugging Face model conditions for the account
  used by `HUGGINGFACE_ACCESS_TOKEN`.
- Voxtral also requires Hugging Face model access and a vLLM build/container with
  realtime/audio support.
- LLM startup can take roughly 90 seconds with the larger local model, so
  `SERVICE_TIMEOUT=300` is recommended.
- Microphone quality materially affects ASR results. Recent testing showed that
  Voxtral accuracy improved substantially with a better microphone.
- Transcript post-processing is intentionally not part of the current pipeline.
  Raw ASR output is preferred for diagnostics and memory ingestion until there is
  a demonstrated reasoning or memory-quality benefit.

## Upstream

- **origin:** `kinetiqs-ai/squelch`
- **upstream:** `pipecat-ai/nemotron-january-2026`

Pull upstream updates with care. This branch intentionally prioritizes the
working release lineage over upstream `main`.

```bash
git fetch upstream
git merge upstream/main
```
