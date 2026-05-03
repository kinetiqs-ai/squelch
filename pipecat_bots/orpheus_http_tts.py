"""HTTP streaming client for Orpheus TTS server.

Connects to the Orpheus TTS HTTP server for speech synthesis.
Streams chunked PCM audio for low time-to-first-byte.

Usage:
    tts = OrpheusHTTPTTSService(server_url="http://localhost:8001")
    # In pipeline: ... -> llm -> tts -> transport.output() -> ...
"""

from typing import AsyncGenerator, Optional

import httpx
from loguru import logger
from pydantic import BaseModel

from pipecat.frames.frames import (
    ErrorFrame,
    Frame,
    TTSAudioRawFrame,
    TTSStartedFrame,
    TTSStoppedFrame,
)
from pipecat.services.tts_service import TTSService

ORPHEUS_SAMPLE_RATE = 24000
ORPHEUS_VOICES = ["tara", "leah", "jess", "leo", "dan", "mia", "zac", "zoe"]


class OrpheusHTTPTTSService(TTSService):
    """HTTP streaming client for Orpheus TTS server."""

    class InputParams(BaseModel):
        language: str = "en"

    def __init__(
        self,
        *,
        server_url: str = "http://localhost:8001",
        voice: str = "tara",
        params: Optional[InputParams] = None,
        **kwargs,
    ):
        super().__init__(sample_rate=ORPHEUS_SAMPLE_RATE, **kwargs)

        self._server_url = server_url.rstrip("/")
        self._voice = voice.lower()
        self._client = httpx.AsyncClient(timeout=60.0)

        self.set_model_name("orpheus-tts")
        self.set_voice(voice)

        logger.info(
            f"OrpheusHTTPTTS initialized: server={server_url}, voice={voice}"
        )

    def can_generate_metrics(self) -> bool:
        return True

    async def run_tts(self, text: str) -> AsyncGenerator[Frame, None]:
        await self.start_ttfb_metrics()
        yield TTSStartedFrame()

        text = text.strip()
        if not text:
            yield TTSStoppedFrame()
            return

        logger.debug(f"OrpheusHTTPTTS: Generating [{text[:50]}...]")

        try:
            req = self._client.build_request(
                "POST",
                f"{self._server_url}/v1/audio/speech",
                json={
                    "input": text,
                    "voice": self._voice,
                    "response_format": "pcm",
                },
            )
            resp = await self._client.send(req, stream=True)

            if resp.status_code != 200:
                body = await resp.aread()
                error_msg = f"TTS server error: {resp.status_code} - {body.decode()}"
                logger.error(error_msg)
                yield ErrorFrame(error=error_msg)
                yield TTSStoppedFrame()
                return

            first_chunk = True
            total_bytes = 0

            async for chunk in resp.aiter_bytes(chunk_size=4096):
                if first_chunk:
                    await self.stop_ttfb_metrics()
                    first_chunk = False
                total_bytes += len(chunk)
                yield TTSAudioRawFrame(
                    audio=chunk,
                    sample_rate=ORPHEUS_SAMPLE_RATE,
                    num_channels=1,
                )

            await resp.aclose()

            dur_ms = total_bytes / 2 / ORPHEUS_SAMPLE_RATE * 1000
            logger.info(f"OrpheusHTTPTTS: {dur_ms:.0f}ms audio received")

            await self.start_tts_usage_metrics(text)
            yield TTSStoppedFrame()

        except httpx.ConnectError as e:
            error_msg = f"Cannot connect to TTS server at {self._server_url}: {e}"
            logger.error(error_msg)
            yield ErrorFrame(error=error_msg)
            yield TTSStoppedFrame()

        except Exception as e:
            logger.error(f"OrpheusHTTPTTS error: {e}")
            yield ErrorFrame(error=str(e))
            yield TTSStoppedFrame()

    async def close(self):
        await self._client.aclose()

    def set_voice(self, voice: str):
        self._voice = voice.lower()
        super().set_voice(voice)
        logger.info(f"OrpheusHTTPTTS: Voice changed to {voice}")
