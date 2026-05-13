# Squelch Agent Instructions

## Machine Context

This repository is on `ramp-genesis`, an NVIDIA Jetson AGX Thor Developer Kit.
Treat Thor as the default local execution target unless the user explicitly says
otherwise.

Verified local host facts:

| Resource | Detail |
| --- | --- |
| Host | `ramp-genesis` |
| OS | Ubuntu 24.04.3 LTS |
| Kernel | `6.8.12-tegra` |
| Architecture | `aarch64` / arm64 SBSA |
| GPU | NVIDIA Thor Blackwell, CUDA 13.0 |
| Memory | 128 GB unified CPU/GPU memory |
| Canonical repo path | `/home/ramp-genesis/squelch` |

Important local caveats:

- Do not assume x86_64, A100, CUDA 12.6, or `sm_80`.
- Do not use `/home/ramp-genesis/squelch-v010-thor`; that worktree was removed
  after merge cleanup.
- Do not create extra Git worktrees unless the user explicitly asks.
- The static hostname may not be useful for browser access; prefer the
  configured Tailscale URL after verifying `tailscale serve status`.
- Docker may require `sudo`.
- `~/.docker` may be root-owned from earlier sudo Docker use.
- `jetson_clocks` requires root and should be enabled before benchmark runs.
- Keep Hugging Face tokens and local model paths out of committed files.

## Project Shape

Squelch is a Thor-local voice stack. The validated deployment path is the native
audio-edge-agent flow:

```text
audio-edge-agent -> native_voice ingress -> VAD -> Riva/NVIDIA ASR
  -> llama.cpp Nemotron LLM -> Magpie TTS -> audio-edge-agent playback
```

Primary files:

| Area | Files |
| --- | --- |
| Native ingress | `native_voice/audio_ingress.py`, `native_voice/riva_asr_app.py` |
| Turn orchestration | `native_voice/orchestrator.py` |
| Riva/NVIDIA ASR helpers | `native_voice/riva_pipeline.py` |
| Magpie TTS server | `src/nemotron_speech/tts_server.py` |
| Magpie streaming presets | `src/nemotron_speech/streaming_tts.py` |
| Native start scripts | `scripts/start_native_audio_ingress.sh`, `scripts/start_native_riva_asr.sh` |
| Pipecat comparison path | `pipecat_bots/` |
| Container management | `scripts/nemotron.sh`, `scripts/start_unified.sh` |

## Current Decisions To Preserve

- The native audio-edge-agent path is primary.
- The Pipecat browser path is retained for comparison, not the default Thor
  deployment path.
- Chinese models are out of scope.
- Orpheus is retained in the repo because custom voices matter, but it was not
  the validated Thor-local runtime due real-time-factor constraints.
- Magpie is the active Thor TTS target.
- The first Magpie segment of each assistant response must use
  `startup_quality`.
- Follow-on Magpie segments use `quality`.
- `SQUELCH_STREAM_LLM_TO_TTS` must default to `0`; LLM-to-TTS token streaming
  did not improve perceived latency and brought back the first-word artifact.
- Persistent Magpie websockets are not the default; they did not solve the
  artifact and caused second-turn reliability issues.
- ASR must be gated by VAD before sending audio to Riva/NVIDIA ASR. Sending
  silence/noise into ASR caused hallucinations.
- The VAD gate includes preroll and explicit segment-end finalization to avoid
  dropped first/final words.

## Runtime Expectations

Expected live local endpoints:

| Service | Endpoint |
| --- | --- |
| Native ingress | `http://127.0.0.1:7860/health` |
| Magpie TTS | `http://127.0.0.1:8101/health` |
| LLM | `http://127.0.0.1:8100` |
| Riva ASR | gRPC on `50051` |

External Tailscale endpoint:

```text
wss://ramp-genesis.tail314cde.ts.net:7860/ws/audio-ingress
```

Before handing a test back to the user, verify:

```bash
curl -fsS http://127.0.0.1:7860/health
curl -fsS http://127.0.0.1:8101/health
tailscale serve status
```

Also verify no process or container is still mounted from a deleted worktree:

```bash
for pid in $(pgrep -f 'native_voice.riva_asr_app|uv run --no-project --with fastapi' || true); do
  printf '%s ' "$pid"
  pwdx "$pid" 2>/dev/null || true
done
sudo docker inspect squelch-magpie-native --format '{{json .HostConfig.Binds}}' 2>/dev/null || true
```

## Development Rules

- Keep edits scoped to the active path unless the user asks for broader cleanup.
- Prefer existing local service contracts over new abstractions.
- Use structured parsing/protocol helpers instead of ad hoc byte/string handling.
- Do not change model choices, VAD semantics, TTS presets, or service boundaries
  without calling out the reason.
- Do not silently switch the project back to browser/Pipecat as the main path.
- Do not revert user work. If the worktree is dirty, inspect and preserve before
  cleanup.
- Do not commit `.env.local`, tokens, generated diagnostics, model files, or
  local caches.
- If Docker commands fail due permissions, use `sudo` or explain the host setup
  issue.

## Validation

Default focused checks:

```bash
bash -n scripts/start_native_audio_ingress.sh scripts/start_native_riva_asr.sh scripts/nemotron.sh scripts/start_unified.sh scripts/voxtral.sh scripts/start_voxtral_asr.sh scripts/start_asr_tts.sh
python -m compileall native_voice src/nemotron_speech pipecat_bots scripts/voice_agent_test_client.py scripts/compare_asr.py
curl -fsS http://127.0.0.1:7860/health
curl -fsS http://127.0.0.1:8101/health
```

If Python bytecode generation fails with permission errors, inspect and remove
generated `__pycache__` directories; root-owned caches can be left by container
runs.

## Git Hygiene

- Canonical branch for active work is `main`.
- Canonical local path is `/home/ramp-genesis/squelch`.
- Avoid multiple local worktrees unless explicitly requested.
- If a backup is needed, prefer a named branch over a second checkout.
- Existing safety branch:

```text
backup/local-main-pre-sync-20260513-164102
```

Do not base new work on that branch unless intentionally recovering old local
changes.
