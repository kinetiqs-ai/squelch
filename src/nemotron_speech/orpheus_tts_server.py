"""Orpheus TTS server — FastAPI HTTP server for Orpheus text-to-speech.

Replaces the Magpie TTS server. Uses vLLM AsyncLLMEngine to run the Orpheus 3B
model (a Llama 3.2 finetune that generates SNAC codec tokens) and the SNAC 24kHz
decoder to convert tokens to audio.

Usage:
    python -m nemotron_speech.orpheus_tts_server --port 8001
"""

import argparse
import asyncio
import struct
import time
import uuid
from contextlib import asynccontextmanager

import numpy as np
import torch
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import Response, StreamingResponse
from loguru import logger
from pydantic import BaseModel

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

LLAMA_VOCAB_SIZE = 128256
CUSTOM_TOKEN_BASE = LLAMA_VOCAB_SIZE  # custom_token_0 = 128256
SNAC_TOKEN_OFFSET = 10  # first 10 custom tokens are special
CODEBOOK_SIZE = 4096
TOKENS_PER_FRAME = 7
SNAC_SAMPLE_RATE = 24000

# Special token IDs
START_TOKEN = 128259
END_TOKENS = [128009, 128260, 128261, 128257]
STOP_TOKEN = 128258

# Codebook mapping: position in 7-token frame -> codebook index
POSITION_TO_CODEBOOK = [0, 1, 2, 2, 1, 2, 2]

# Valid token ID ranges per position in the 7-token frame
# Position p: custom_token_{10 + p*4096} through custom_token_{10 + p*4096 + 4095}
# Token ID = CUSTOM_TOKEN_BASE + custom_token_number
POSITION_VALID_RANGES = []
for _p in range(TOKENS_PER_FRAME):
    _start = CUSTOM_TOKEN_BASE + SNAC_TOKEN_OFFSET + _p * CODEBOOK_SIZE
    _end = _start + CODEBOOK_SIZE  # exclusive
    POSITION_VALID_RANGES.append((_start, _end))

VOICES = ["tara", "leah", "jess", "leo", "dan", "mia", "zac", "zoe"]
DEFAULT_VOICE = "tara"
DEFAULT_MODEL = "canopylabs/orpheus-3b-0.1-ft"

# ---------------------------------------------------------------------------
# SNAC logit processor for vLLM
# ---------------------------------------------------------------------------


class SNACLogitsProcessor:
    """Constrains vLLM output logits to valid SNAC token ranges.

    Each position in the 7-token frame cycle has a specific valid token range
    (one codebook slice of 4096 tokens). This processor masks all other tokens
    to -inf, preventing out-of-range tokens that cause dropped audio frames.
    The stop token is always allowed.
    """

    def __init__(self, num_prompt_tokens: int):
        self._num_prompt = num_prompt_tokens
        self._masks: dict[int, torch.Tensor] = {}

    def __call__(
        self,
        token_ids: list[int],
        logits: torch.Tensor,
    ) -> torch.Tensor:
        num_generated = len(token_ids) - self._num_prompt
        if num_generated < 0:
            return logits

        pos = num_generated % TOKENS_PER_FRAME
        key = (pos, logits.shape[-1], str(logits.device))

        if key not in self._masks:
            mask = torch.full_like(logits, float("-inf"))
            start, end = POSITION_VALID_RANGES[pos]
            end = min(end, logits.shape[-1])
            mask[start:end] = 0.0
            mask[STOP_TOKEN] = 0.0
            self._masks[key] = mask

        return logits + self._masks[key]


# ---------------------------------------------------------------------------
# SNAC decoder
# ---------------------------------------------------------------------------


class SNACDecoder:
    """Converts SNAC token sequences to 24kHz PCM audio."""

    def __init__(self, device: str = "cuda"):
        self._device = device
        self._model = None

    def load(self):
        from snac import SNAC

        logger.info("Loading SNAC 24kHz decoder...")
        self._model = SNAC.from_pretrained("hubertsiuzdak/snac_24khz").eval()
        self._model = self._model.to(self._device)
        logger.info("SNAC decoder loaded")

    def _token_to_codebook_index(self, token_id: int, position: int) -> int | None:
        """Convert a token ID to a codebook index for the given frame position."""
        custom_num = token_id - CUSTOM_TOKEN_BASE
        idx = custom_num - SNAC_TOKEN_OFFSET - position * CODEBOOK_SIZE
        if 0 <= idx < CODEBOOK_SIZE:
            return idx
        return None

    def decode_frames(self, token_ids: list[int]) -> bytes:
        """Decode a list of SNAC token IDs into PCM int16 audio bytes.

        token_ids should be a multiple of 7 in length (complete frames).
        """
        num_frames = len(token_ids) // TOKENS_PER_FRAME
        if num_frames == 0:
            return b""

        codes_0 = []
        codes_1 = []
        codes_2 = []

        for j in range(num_frames):
            base = j * TOKENS_PER_FRAME
            frame = token_ids[base : base + TOKENS_PER_FRAME]

            c0 = self._token_to_codebook_index(frame[0], 0)
            c1a = self._token_to_codebook_index(frame[1], 1)
            c2a = self._token_to_codebook_index(frame[2], 2)
            c2b = self._token_to_codebook_index(frame[3], 3)
            c1b = self._token_to_codebook_index(frame[4], 4)
            c2c = self._token_to_codebook_index(frame[5], 5)
            c2d = self._token_to_codebook_index(frame[6], 6)

            vals = [c0, c1a, c2a, c2b, c1b, c2c, c2d]
            if any(v is None for v in vals):
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
            audio_hat = self._model.decode([t0, t1, t2])

        audio_np = audio_hat.squeeze().cpu().numpy()
        audio_int16 = np.clip(audio_np * 32767, -32768, 32767).astype(np.int16)
        return audio_int16.tobytes()


# ---------------------------------------------------------------------------
# Request / response models
# ---------------------------------------------------------------------------


class SpeechRequest(BaseModel):
    input: str
    voice: str = DEFAULT_VOICE
    language: str = "en"
    response_format: str = "pcm"


class AudioConfig(BaseModel):
    sample_rate: int = SNAC_SAMPLE_RATE
    channels: int = 1
    encoding: str = "pcm_s16le"
    voices: list[str] = VOICES


# ---------------------------------------------------------------------------
# Application state
# ---------------------------------------------------------------------------

_state: dict = {}


def _get_engine():
    return _state["engine"]


def _get_decoder() -> SNACDecoder:
    return _state["decoder"]


def _get_tokenizer():
    return _state["tokenizer"]


# ---------------------------------------------------------------------------
# FastAPI lifespan
# ---------------------------------------------------------------------------


@asynccontextmanager
async def lifespan(app: FastAPI):
    model_name = app.state.model_name
    gpu_mem = app.state.gpu_memory_utilization

    logger.info(f"Loading Orpheus TTS model: {model_name}")
    logger.info(f"GPU memory utilization: {gpu_mem}")

    from vllm import AsyncEngineArgs, AsyncLLMEngine

    engine_args = AsyncEngineArgs(
        model=model_name,
        dtype="bfloat16",
        max_model_len=2048,
        gpu_memory_utilization=gpu_mem,
        enforce_eager=True,
        max_num_seqs=4,
        disable_log_requests=True,
    )
    engine = AsyncLLMEngine.from_engine_args(engine_args)
    _state["engine"] = engine

    # Load tokenizer for prompt formatting
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    _state["tokenizer"] = tokenizer

    # Load SNAC decoder
    decoder = SNACDecoder(device="cuda")
    decoder.load()
    _state["decoder"] = decoder

    # Warmup
    logger.info("Warming up Orpheus TTS...")
    t0 = time.time()
    warmup_text = "Hello, this is a warmup test for the text to speech system."
    try:
        audio = await _generate_speech(warmup_text, DEFAULT_VOICE)
        dur_ms = len(audio) / 2 / SNAC_SAMPLE_RATE * 1000
        logger.info(f"Warmup complete in {time.time() - t0:.1f}s, audio={dur_ms:.0f}ms")
    except Exception as e:
        logger.warning(f"Warmup failed (non-fatal): {e}")

    torch.cuda.empty_cache()
    _state["ready"] = True
    logger.info("Orpheus TTS server ready")

    yield

    _state.clear()


# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------


def _format_prompt_tokens(text: str, voice: str) -> list[int]:
    """Format input text into Orpheus prompt token IDs."""
    tokenizer = _get_tokenizer()
    prompt_text = f"{voice}: {text}"
    text_token_ids = tokenizer.encode(prompt_text, add_special_tokens=False)
    return [START_TOKEN] + text_token_ids + END_TOKENS


async def _generate_speech(text: str, voice: str) -> bytes:
    """Generate full audio for the given text and voice."""
    from vllm import SamplingParams

    engine = _get_engine()
    decoder = _get_decoder()

    prompt_ids = _format_prompt_tokens(text, voice)
    logits_processor = SNACLogitsProcessor(num_prompt_tokens=len(prompt_ids))

    sampling_params = SamplingParams(
        temperature=0.6,
        top_p=0.9,
        max_tokens=1200,
        stop_token_ids=[STOP_TOKEN],
        repetition_penalty=1.1,
        logits_processors=[logits_processor],
    )

    request_id = str(uuid.uuid4())
    all_token_ids: list[int] = []
    prev_len = 0

    async for result in engine.generate(
        prompt=None,
        sampling_params=sampling_params,
        request_id=request_id,
        prompt_token_ids=prompt_ids,
    ):
        output = result.outputs[0]
        new_ids = output.token_ids[prev_len:]
        prev_len = len(output.token_ids)

        for tid in new_ids:
            if tid == STOP_TOKEN:
                break
            start, end = POSITION_VALID_RANGES[len(all_token_ids) % TOKENS_PER_FRAME]
            if start <= tid < end:
                all_token_ids.append(tid)

    # Trim to complete frames
    usable = (len(all_token_ids) // TOKENS_PER_FRAME) * TOKENS_PER_FRAME
    all_token_ids = all_token_ids[:usable]

    if not all_token_ids:
        return b""

    return decoder.decode_frames(all_token_ids)


async def _generate_speech_streaming(text: str, voice: str):
    """Generate audio and yield PCM chunks as SNAC frames complete."""
    from vllm import SamplingParams

    engine = _get_engine()
    decoder = _get_decoder()

    prompt_ids = _format_prompt_tokens(text, voice)
    logits_processor = SNACLogitsProcessor(num_prompt_tokens=len(prompt_ids))

    sampling_params = SamplingParams(
        temperature=0.6,
        top_p=0.9,
        max_tokens=1200,
        stop_token_ids=[STOP_TOKEN],
        repetition_penalty=1.1,
        logits_processors=[logits_processor],
    )

    request_id = str(uuid.uuid4())
    token_buffer: list[int] = []
    prev_len = 0
    frames_decoded = 0
    min_frames_before_emit = 4  # Wait for 4 frames (28 tokens) before first decode

    async for result in engine.generate(
        prompt=None,
        sampling_params=sampling_params,
        request_id=request_id,
        prompt_token_ids=prompt_ids,
    ):
        output = result.outputs[0]
        new_ids = output.token_ids[prev_len:]
        prev_len = len(output.token_ids)

        for tid in new_ids:
            if tid == STOP_TOKEN:
                break
            pos = len(token_buffer) % TOKENS_PER_FRAME
            start, end = POSITION_VALID_RANGES[pos]
            if start <= tid < end:
                token_buffer.append(tid)

        total_frames = len(token_buffer) // TOKENS_PER_FRAME

        while total_frames > frames_decoded and total_frames >= min_frames_before_emit:
            # Decode using a window of up to 4 frames ending at the current position
            end_frame = frames_decoded + 1
            start_frame = max(0, end_frame - 4)
            window_start = start_frame * TOKENS_PER_FRAME
            window_end = end_frame * TOKENS_PER_FRAME
            window_tokens = token_buffer[window_start:window_end]

            audio_bytes = decoder.decode_frames(window_tokens)
            if audio_bytes:
                # Take only the audio from the last frame to avoid overlap
                num_window_frames = end_frame - start_frame
                if num_window_frames > 1:
                    samples_per_decode = len(audio_bytes) // 2  # int16 = 2 bytes
                    # Take roughly the last 1/num_window_frames of the audio
                    chunk_samples = samples_per_decode // num_window_frames
                    offset = (num_window_frames - 1) * chunk_samples
                    audio_bytes = audio_bytes[offset * 2 :]

                yield audio_bytes

            frames_decoded += 1

    # Decode any remaining complete frames
    total_frames = len(token_buffer) // TOKENS_PER_FRAME
    while total_frames > frames_decoded:
        end_frame = frames_decoded + 1
        start_frame = max(0, end_frame - 4)
        window_tokens = token_buffer[start_frame * TOKENS_PER_FRAME : end_frame * TOKENS_PER_FRAME]
        audio_bytes = decoder.decode_frames(window_tokens)
        if audio_bytes:
            num_window_frames = end_frame - start_frame
            if num_window_frames > 1:
                samples_per_decode = len(audio_bytes) // 2
                chunk_samples = samples_per_decode // num_window_frames
                offset = (num_window_frames - 1) * chunk_samples
                audio_bytes = audio_bytes[offset * 2 :]
            yield audio_bytes
        frames_decoded += 1


# ---------------------------------------------------------------------------
# FastAPI routes
# ---------------------------------------------------------------------------

app = FastAPI(title="Orpheus TTS Server", lifespan=lifespan)


@app.get("/health")
async def health():
    if not _state.get("ready"):
        raise HTTPException(status_code=503, detail="Model loading")
    return {"status": "ok", "model": "orpheus-tts"}


@app.get("/v1/audio/config")
async def audio_config():
    return AudioConfig()


@app.post("/v1/audio/speech")
async def speech(request: SpeechRequest):
    if not _state.get("ready"):
        raise HTTPException(status_code=503, detail="Model not ready")

    text = request.input.strip()
    if not text:
        raise HTTPException(status_code=400, detail="Empty input text")

    voice = request.voice.lower()
    if voice not in VOICES:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown voice '{voice}'. Available: {VOICES}",
        )

    logger.info(f"TTS request: voice={voice}, text={text[:80]}{'...' if len(text) > 80 else ''}")
    t0 = time.time()

    async def audio_stream():
        total_bytes = 0
        async for chunk in _generate_speech_streaming(text, voice):
            total_bytes += len(chunk)
            yield chunk

        elapsed = time.time() - t0
        dur_ms = total_bytes / 2 / SNAC_SAMPLE_RATE * 1000
        logger.info(f"TTS complete: {dur_ms:.0f}ms audio in {elapsed:.2f}s (RTF={elapsed/(dur_ms/1000):.2f}x)")

    return StreamingResponse(
        audio_stream(),
        media_type="application/octet-stream",
        headers={
            "X-Sample-Rate": str(SNAC_SAMPLE_RATE),
            "X-Channels": "1",
            "X-Encoding": "pcm_s16le",
        },
    )


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(description="Orpheus TTS Server")
    parser.add_argument("--port", type=int, default=8001)
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.15)
    args = parser.parse_args()

    app.state.model_name = args.model
    app.state.gpu_memory_utilization = args.gpu_memory_utilization

    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
