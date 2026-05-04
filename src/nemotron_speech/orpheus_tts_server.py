"""FastAPI server for local Orpheus text-to-speech inference.

Orpheus is a Llama-based speech model that emits SNAC codec tokens. This server
uses vLLM for token generation and decodes complete SNAC frames to 24 kHz PCM.
It intentionally mirrors the existing local TTS surface used by the bots:

    GET  /health
    GET  /v1/audio/config
    POST /v1/audio/speech
"""

import argparse
import os
import time
import uuid
from contextlib import asynccontextmanager
from typing import AsyncGenerator

import numpy as np
import torch
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from loguru import logger
from pydantic import BaseModel

os.environ.setdefault("VLLM_USE_V1", "0")

LLAMA_VOCAB_SIZE = 128256
CUSTOM_TOKEN_BASE = LLAMA_VOCAB_SIZE
SNAC_TOKEN_OFFSET = 10
CODEBOOK_SIZE = 4096
TOKENS_PER_FRAME = 7
SNAC_SAMPLE_RATE = 24000

START_TOKEN = 128259
END_TOKENS = [128009, 128260, 128261, 128257]
STOP_TOKEN = 128258

VOICES = ["tara", "leah", "jess", "leo", "dan", "mia", "zac", "zoe"]
DEFAULT_VOICE = "tara"
DEFAULT_MODEL = "canopylabs/orpheus-3b-0.1-ft"

POSITION_VALID_RANGES: list[tuple[int, int]] = []
for _position in range(TOKENS_PER_FRAME):
    _start = CUSTOM_TOKEN_BASE + SNAC_TOKEN_OFFSET + _position * CODEBOOK_SIZE
    POSITION_VALID_RANGES.append((_start, _start + CODEBOOK_SIZE))


class SNACLogitsProcessor:
    """Mask generation to the valid SNAC codebook range for each frame position."""

    def __init__(self, num_prompt_tokens: int):
        self._num_prompt_tokens = num_prompt_tokens
        self._masks: dict[tuple[int, int, str], torch.Tensor] = {}

    def __call__(self, token_ids: list[int], logits: torch.Tensor) -> torch.Tensor:
        generated_count = len(token_ids) - self._num_prompt_tokens
        if generated_count < 0:
            return logits

        position = generated_count % TOKENS_PER_FRAME
        key = (position, logits.shape[-1], str(logits.device))
        if key not in self._masks:
            mask = torch.full_like(logits, float("-inf"))
            start, end = POSITION_VALID_RANGES[position]
            if start < logits.shape[-1]:
                mask[start : min(end, logits.shape[-1])] = 0.0
            if STOP_TOKEN < logits.shape[-1]:
                mask[STOP_TOKEN] = 0.0
            self._masks[key] = mask

        return logits + self._masks[key]


class SNACDecoder:
    """Decode Orpheus SNAC token IDs into raw pcm_s16le audio."""

    def __init__(self, device: str = "cuda"):
        self._device = device
        self._model = None

    def load(self) -> None:
        from snac import SNAC

        logger.info("Loading SNAC 24kHz decoder")
        self._model = SNAC.from_pretrained("hubertsiuzdak/snac_24khz").eval()
        self._model = self._model.to(self._device)
        logger.info("SNAC decoder loaded")

    def _token_to_code(self, token_id: int, position: int) -> int | None:
        custom_token_number = token_id - CUSTOM_TOKEN_BASE
        code = custom_token_number - SNAC_TOKEN_OFFSET - position * CODEBOOK_SIZE
        if 0 <= code < CODEBOOK_SIZE:
            return code
        return None

    def decode_frames(self, token_ids: list[int]) -> bytes:
        """Decode complete 7-token SNAC frames to PCM bytes."""
        if self._model is None:
            raise RuntimeError("SNAC decoder is not loaded")

        frame_count = len(token_ids) // TOKENS_PER_FRAME
        if frame_count <= 0:
            return b""

        codes_0: list[int] = []
        codes_1: list[int] = []
        codes_2: list[int] = []

        for frame_index in range(frame_count):
            start = frame_index * TOKENS_PER_FRAME
            frame = token_ids[start : start + TOKENS_PER_FRAME]

            c0 = self._token_to_code(frame[0], 0)
            c1a = self._token_to_code(frame[1], 1)
            c2a = self._token_to_code(frame[2], 2)
            c2b = self._token_to_code(frame[3], 3)
            c1b = self._token_to_code(frame[4], 4)
            c2c = self._token_to_code(frame[5], 5)
            c2d = self._token_to_code(frame[6], 6)
            values = [c0, c1a, c2a, c2b, c1b, c2c, c2d]

            if any(value is None for value in values):
                logger.debug(f"Skipping invalid SNAC frame at index {frame_index}")
                continue

            codes_0.append(c0)
            codes_1.extend([c1a, c1b])
            codes_2.extend([c2a, c2b, c2c, c2d])

        if not codes_0:
            return b""

        with torch.inference_mode():
            t0 = torch.tensor([codes_0], dtype=torch.long, device=self._device)
            t1 = torch.tensor([codes_1], dtype=torch.long, device=self._device)
            t2 = torch.tensor([codes_2], dtype=torch.long, device=self._device)
            audio = self._model.decode([t0, t1, t2])

        audio_np = audio.squeeze().detach().cpu().numpy()
        audio_int16 = np.clip(audio_np * 32767, -32768, 32767).astype(np.int16)
        return audio_int16.tobytes()

    def decode_codes(self, codes: list[int]) -> bytes:
        """Decode raw SNAC code values using Orpheus' 4-frame streaming window."""
        if self._model is None:
            raise RuntimeError("SNAC decoder is not loaded")

        if len(codes) < 28:
            return b""

        frame = codes[-28:]
        codes_0: list[int] = []
        codes_1: list[int] = []
        codes_2: list[int] = []

        for frame_index in range(4):
            start = frame_index * TOKENS_PER_FRAME
            codes_0.append(frame[start])
            codes_1.extend([frame[start + 1], frame[start + 4]])
            codes_2.extend([frame[start + 2], frame[start + 3], frame[start + 5], frame[start + 6]])

        if any(code < 0 or code >= CODEBOOK_SIZE for code in codes_0 + codes_1 + codes_2):
            return b""

        with torch.inference_mode():
            t0 = torch.tensor([codes_0], dtype=torch.long, device=self._device)
            t1 = torch.tensor([codes_1], dtype=torch.long, device=self._device)
            t2 = torch.tensor([codes_2], dtype=torch.long, device=self._device)
            audio = self._model.decode([t0, t1, t2])

        # Canopy's reference decoder emits the second 2048-sample slice from each
        # 4-frame window. Taking the last proportional slice causes discontinuous
        # and often unintelligible audio.
        audio_slice = audio[:, :, 2048:4096]
        audio_np = audio_slice.squeeze().detach().cpu().numpy()
        audio_int16 = np.clip(audio_np * 32767, -32768, 32767).astype(np.int16)
        return audio_int16.tobytes()


class SpeechRequest(BaseModel):
    input: str
    voice: str = DEFAULT_VOICE
    language: str = "en"
    response_format: str = "pcm"
    speed: float = 1.0


class TTSConfig(BaseModel):
    sample_rate: int = SNAC_SAMPLE_RATE
    channels: int = 1
    encoding: str = "pcm_s16le"
    voices: list[str] = VOICES
    languages: list[str] = ["en"]


_state: dict = {}


def _env_bool(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.lower() in {"1", "true", "yes", "on"}


def _format_prompt_tokens(text: str, voice: str) -> list[int]:
    tokenizer = _state["tokenizer"]
    prompt_text = f"{voice}: {text}"
    # Match Canopy's OrpheusModel._format_prompt(): tokenizer(...).input_ids
    # keeps the model's normal begin-of-text token inside the Orpheus wrapper.
    text_token_ids = tokenizer(prompt_text, return_tensors="pt").input_ids[0].tolist()
    return [START_TOKEN] + text_token_ids + END_TOKENS


@asynccontextmanager
async def lifespan(app: FastAPI):
    model_name = app.state.model_name
    gpu_memory_utilization = app.state.gpu_memory_utilization

    logger.info(f"Loading Orpheus TTS model: {model_name}")
    logger.info(f"Orpheus vLLM GPU memory utilization: {gpu_memory_utilization}")

    from transformers import AutoTokenizer
    from vllm import AsyncEngineArgs, AsyncLLMEngine

    engine_args = AsyncEngineArgs(
        model=model_name,
        dtype="bfloat16",
        max_model_len=2048,
        gpu_memory_utilization=gpu_memory_utilization,
        enforce_eager=_env_bool("ORPHEUS_ENFORCE_EAGER", False),
        max_num_seqs=1,
        disable_log_requests=True,
    )
    _state["engine"] = AsyncLLMEngine.from_engine_args(engine_args)
    _state["tokenizer"] = AutoTokenizer.from_pretrained(model_name)

    decoder = SNACDecoder(device="cuda" if torch.cuda.is_available() else "cpu")
    decoder.load()
    _state["decoder"] = decoder

    warmup_text = os.getenv("ORPHEUS_WARMUP_TEXT", "Hello, this is a warmup test.")
    try:
        logger.info("Warming up Orpheus TTS")
        t0 = time.time()
        audio = b""
        async for chunk in _generate_speech_streaming(warmup_text, DEFAULT_VOICE):
            audio += chunk
        duration_ms = len(audio) / 2 / SNAC_SAMPLE_RATE * 1000
        logger.info(
            f"Orpheus warmup complete in {time.time() - t0:.1f}s, "
            f"audio={duration_ms:.0f}ms"
        )
    except Exception as e:
        logger.warning(f"Orpheus warmup failed, continuing: {e}")

    torch.cuda.empty_cache()
    _state["ready"] = True
    logger.info("Orpheus TTS server ready")

    yield

    _state.clear()


async def _generate_speech_streaming(text: str, voice: str) -> AsyncGenerator[bytes, None]:
    from vllm import SamplingParams

    engine = _state["engine"]
    decoder: SNACDecoder = _state["decoder"]
    prompt_ids = _format_prompt_tokens(text, voice)

    sampling_params = SamplingParams(
        temperature=float(os.getenv("ORPHEUS_TEMPERATURE", "0.4")),
        top_p=float(os.getenv("ORPHEUS_TOP_P", "0.9")),
        max_tokens=int(os.getenv("ORPHEUS_MAX_TOKENS", "1200")),
        stop_token_ids=[STOP_TOKEN],
        repetition_penalty=float(os.getenv("ORPHEUS_REPETITION_PENALTY", "1.1")),
        logits_processors=(
            [SNACLogitsProcessor(num_prompt_tokens=len(prompt_ids))]
            if os.getenv("ORPHEUS_USE_LOGITS_PROCESSOR", "false").lower()
            in {"1", "true", "yes"}
            else None
        ),
    )

    request_id = str(uuid.uuid4())
    code_buffer: list[int] = []
    previous_output_len = 0
    emitted_frames = 0

    async for result in engine.generate(
        prompt={"prompt_token_ids": prompt_ids},
        sampling_params=sampling_params,
        request_id=request_id,
    ):
        output = result.outputs[0]
        new_token_ids = output.token_ids[previous_output_len:]
        previous_output_len = len(output.token_ids)

        for token_id in new_token_ids:
            if token_id == STOP_TOKEN:
                break
            position = len(code_buffer) % TOKENS_PER_FRAME
            start, end = POSITION_VALID_RANGES[position]
            if start <= token_id < end:
                code_buffer.append(token_id - start)

        complete_frames = len(code_buffer) // TOKENS_PER_FRAME
        while complete_frames > emitted_frames and complete_frames >= 4:
            yield decoder.decode_codes(code_buffer)
            emitted_frames += 1


app = FastAPI(
    title="Orpheus TTS Server",
    description="Local Orpheus TTS inference server",
    version="1.0.0",
    lifespan=lifespan,
)


@app.get("/health")
async def health():
    return {
        "status": "healthy" if _state.get("ready") else "loading",
        "model_loaded": bool(_state.get("ready")),
        "model": getattr(app.state, "model_name", DEFAULT_MODEL),
    }


@app.get("/v1/audio/config")
async def get_config():
    return TTSConfig()


@app.post("/v1/audio/speech")
async def synthesize_speech(request: SpeechRequest):
    if not _state.get("ready"):
        raise HTTPException(status_code=503, detail="Model not loaded")

    text = request.input.strip()
    if not text:
        raise HTTPException(status_code=400, detail="Empty input text")

    voice = request.voice.lower()
    if voice not in VOICES:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown voice '{request.voice}'. Available: {VOICES}",
        )

    logger.info(f"Orpheus TTS request: voice={voice}, text=[{text[:80]}...]")
    start_time = time.time()

    async def audio_stream() -> AsyncGenerator[bytes, None]:
        total_bytes = 0
        async for chunk in _generate_speech_streaming(text, voice):
            if not chunk:
                continue
            total_bytes += len(chunk)
            yield chunk

        duration_s = total_bytes / 2 / SNAC_SAMPLE_RATE
        elapsed_s = time.time() - start_time
        rtf = elapsed_s / duration_s if duration_s > 0 else 0
        logger.info(
            f"Orpheus TTS complete: {duration_s * 1000:.0f}ms audio, "
            f"latency={elapsed_s * 1000:.0f}ms, RTF={rtf:.2f}x"
        )

    return StreamingResponse(
        audio_stream(),
        media_type="audio/pcm",
        headers={
            "X-Sample-Rate": str(SNAC_SAMPLE_RATE),
            "X-Channels": "1",
            "X-Encoding": "pcm_s16le",
        },
    )


def main():
    parser = argparse.ArgumentParser(description="Orpheus TTS server")
    parser.add_argument("--host", default=os.getenv("ORPHEUS_HOST", "0.0.0.0"))
    parser.add_argument("--port", type=int, default=int(os.getenv("ORPHEUS_PORT", "8001")))
    parser.add_argument("--model", default=os.getenv("ORPHEUS_MODEL", DEFAULT_MODEL))
    parser.add_argument(
        "--gpu-memory-utilization",
        type=float,
        default=float(os.getenv("ORPHEUS_GPU_MEMORY_UTILIZATION", "0.25")),
    )
    args = parser.parse_args()

    app.state.model_name = args.model
    app.state.gpu_memory_utilization = args.gpu_memory_utilization

    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
