"""ASR backend selection helpers for Pipecat bots."""

import os

from loguru import logger

from nvidia_stt import NVidiaWebSocketSTTService
from voxtral_stt import VoxtralRealtimeSTTService


ASR_BACKEND = os.getenv("ASR_BACKEND", "nemotron").lower()
NVIDIA_ASR_URL = os.getenv("NVIDIA_ASR_URL", "ws://localhost:8080")
VOXTRAL_ASR_URL = os.getenv(
    "VOXTRAL_ASR_URL",
    "ws://host.docker.internal:8082/v1/realtime",
)
VOXTRAL_MODEL = os.getenv("VOXTRAL_MODEL", "mistralai/Voxtral-Mini-4B-Realtime-2602")


def create_stt_service(sample_rate: int = 16000):
    if ASR_BACKEND == "voxtral":
        logger.info(f"Using Voxtral Realtime ASR at {VOXTRAL_ASR_URL}")
        return VoxtralRealtimeSTTService(
            url=VOXTRAL_ASR_URL,
            model=VOXTRAL_MODEL,
            sample_rate=sample_rate,
        )

    if ASR_BACKEND != "nemotron":
        logger.warning(f"Unknown ASR_BACKEND={ASR_BACKEND}; falling back to nemotron")

    logger.info(f"Using Nemotron ASR at {NVIDIA_ASR_URL}")
    return NVidiaWebSocketSTTService(
        url=NVIDIA_ASR_URL,
        sample_rate=sample_rate,
    )


def describe_asr_backend() -> str:
    if ASR_BACKEND == "voxtral":
        return f"voxtral ({VOXTRAL_ASR_URL})"
    return f"nemotron ({NVIDIA_ASR_URL})"
