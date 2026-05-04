"""PipeCat TTS service for the local Orpheus HTTP streaming server."""

import re
from collections.abc import AsyncGenerator
from typing import Optional

import httpx
from loguru import logger
from pydantic import BaseModel

from pipecat.frames.frames import (
    CancelFrame,
    ErrorFrame,
    Frame,
    InterruptionFrame,
    TTSAudioRawFrame,
    TTSStartedFrame,
    TTSStoppedFrame,
)
from pipecat.processors.frame_processor import FrameDirection
from pipecat.services.tts_service import TTSService

try:
    from pipecat_bots.frames import ChunkedLLMContinueGenerationFrame
except ModuleNotFoundError:
    from frames import ChunkedLLMContinueGenerationFrame

ORPHEUS_SAMPLE_RATE = 24000

EMOJI_PATTERN = re.compile(
    "["
    "\U0001F600-\U0001F64F"
    "\U0001F300-\U0001F5FF"
    "\U0001F680-\U0001F6FF"
    "\U0001F700-\U0001F77F"
    "\U0001F780-\U0001F7FF"
    "\U0001F800-\U0001F8FF"
    "\U0001F900-\U0001F9FF"
    "\U0001FA00-\U0001FA6F"
    "\U0001FA70-\U0001FAFF"
    "\U00002702-\U000027B0"
    "\U0001F1E0-\U0001F1FF"
    "]+",
    flags=re.UNICODE,
)


def sanitize_text_for_tts(text: str) -> str:
    text = EMOJI_PATTERN.sub("", text)
    text = text.replace("\u2018", "'")
    text = text.replace("\u2019", "'")
    text = text.replace("\u201C", '"')
    text = text.replace("\u201D", '"')
    text = text.replace("\u2014", "-")
    text = text.replace("\u2013", "-")
    return text


class OrpheusHTTPTTSService(TTSService):
    """Streams PCM audio from the local Orpheus server.

    The buffered LLM service waits for ChunkedLLMContinueGenerationFrame before
    emitting the next text segment. This service sends that frame after each HTTP
    response is fully streamed, preserving the Magpie integration contract.
    """

    class InputParams(BaseModel):
        language: str = "en"
        request_timeout_s: float = 120.0

    def __init__(
        self,
        *,
        server_url: str = "http://localhost:8001",
        voice: str = "tara",
        sample_rate: Optional[int] = None,
        params: Optional[InputParams] = None,
        **kwargs,
    ):
        super().__init__(
            sample_rate=sample_rate or ORPHEUS_SAMPLE_RATE,
            aggregate_sentences=False,
            **kwargs,
        )
        self._server_url = server_url.rstrip("/")
        self._voice = voice.lower()
        self._params = params or OrpheusHTTPTTSService.InputParams()
        self._client = httpx.AsyncClient(timeout=self._params.request_timeout_s)
        self._generation_id = 0

        self.set_model_name("orpheus-http")
        self.set_voice(voice)
        logger.info(
            f"OrpheusHTTPTTS initialized: server={self._server_url}, "
            f"voice={self._voice}, sample_rate={self.sample_rate}"
        )

    def can_generate_metrics(self) -> bool:
        return True

    async def cancel(self, frame: CancelFrame):
        self._generation_id += 1
        await self.stop_all_metrics()
        await super().cancel(frame)

    async def _handle_interruption(self, frame: InterruptionFrame, direction: FrameDirection):
        self._generation_id += 1
        await self.stop_all_metrics()
        await self.push_frame(frame, direction)

    async def run_tts(self, text: str) -> AsyncGenerator[Frame, None]:
        text = sanitize_text_for_tts(text).strip()
        if not text:
            yield None
            return

        self._generation_id += 1
        generation_id = self._generation_id

        await self.start_ttfb_metrics()
        yield TTSStartedFrame()

        logger.debug(f"OrpheusHTTPTTS: generating [{text[:80]}...]")
        response = None
        first_chunk = True
        total_bytes = 0

        try:
            request = self._client.build_request(
                "POST",
                f"{self._server_url}/v1/audio/speech",
                json={
                    "input": text,
                    "voice": self._voice,
                    "language": self._params.language,
                    "response_format": "pcm",
                },
            )
            response = await self._client.send(request, stream=True)

            if response.status_code != 200:
                body = await response.aread()
                error = f"Orpheus TTS error {response.status_code}: {body.decode()}"
                logger.error(error)
                yield ErrorFrame(error=error)
                return

            async for chunk in response.aiter_bytes(chunk_size=4096):
                if generation_id != self._generation_id:
                    logger.debug("Discarding stale Orpheus audio after interruption")
                    break
                if not chunk:
                    continue
                total_bytes += len(chunk)
                if first_chunk:
                    await self.stop_ttfb_metrics()
                    first_chunk = False
                yield TTSAudioRawFrame(chunk, self.sample_rate or ORPHEUS_SAMPLE_RATE, 1)

            sample_rate = self.sample_rate or ORPHEUS_SAMPLE_RATE
            duration_ms = total_bytes / 2 / sample_rate * 1000
            logger.info(f"OrpheusHTTPTTS: received {duration_ms:.0f}ms audio")
            await self.start_tts_usage_metrics(text)

        except httpx.ConnectError as e:
            error = f"Cannot connect to Orpheus TTS server at {self._server_url}: {e}"
            logger.error(error)
            yield ErrorFrame(error=error)
        except Exception as e:
            logger.error(f"OrpheusHTTPTTS error: {e}")
            yield ErrorFrame(error=str(e))
        finally:
            if response is not None:
                await response.aclose()
            await self.stop_ttfb_metrics()
            yield TTSStoppedFrame()
            await self.push_frame(
                ChunkedLLMContinueGenerationFrame(),
                FrameDirection.UPSTREAM,
            )

    async def close(self):
        await self._client.aclose()

    def set_voice(self, voice: str):
        self._voice = voice.lower()
        super().set_voice(voice)
