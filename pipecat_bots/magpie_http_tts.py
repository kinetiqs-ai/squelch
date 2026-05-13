"""Pipecat TTS service for the local Magpie HTTP batch server."""

import asyncio
import re
import time
from typing import AsyncGenerator, Optional

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


DEFAULT_SAMPLE_RATE = 22050

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
    return text.strip()


class MagpieHTTPTTSService(TTSService):
    """Fetch complete sentence audio from Magpie's batch endpoint.

    Batch sentence synthesis is deliberate for the Thor bring-up path. It avoids
    the old frame-level decode stitching that caused audible boundary artifacts.
    """

    class InputParams(BaseModel):
        """Input parameters for Magpie HTTP TTS."""

        language: str = "en"
        request_timeout_s: float = 120.0

    def __init__(
        self,
        *,
        server_url: str = "http://localhost:8001",
        voice: str = "aria",
        language: str = "en",
        sample_rate: Optional[int] = None,
        params: Optional[InputParams] = None,
        **kwargs,
    ):
        """Initialize Magpie HTTP TTS client.

        Args:
            server_url: TTS server URL (default: http://localhost:8001)
            voice: Speaker voice (john, sofia, aria, jason, leo)
            language: Language code (en, es, de, fr, vi, it, zh)
            sample_rate: Output sample rate (default: fetched from server)
            params: Additional TTS parameters.
        """
        super().__init__(
            sample_rate=sample_rate or DEFAULT_SAMPLE_RATE,
            aggregate_sentences=False,
            **kwargs,
        )

        params = params or MagpieHTTPTTSService.InputParams()

        self._server_url = server_url.rstrip("/")
        self._voice = voice.lower()
        self._language = language.lower()
        self._sample_rate = sample_rate
        self._params = params
        self._generation_id = 0

        # HTTP client with connection pooling
        self._client = httpx.AsyncClient(timeout=params.request_timeout_s)
        self._config_fetched = False

        self.set_model_name("magpie-http")
        self.set_voice(voice)

        logger.info(
            f"MagpieHTTPTTS initialized: server={server_url}, "
            f"voice={voice}, language={language}"
        )

    async def _ensure_config(self):
        """Fetch server config if not already done."""
        if self._config_fetched:
            return

        try:
            resp = await self._client.get(f"{self._server_url}/v1/audio/config")
            if resp.status_code == 200:
                config = resp.json()
                server_sample_rate = config.get("sample_rate", DEFAULT_SAMPLE_RATE)

                # Update sample rate if not explicitly set
                if self._sample_rate is None:
                    self._sample_rate = server_sample_rate
                    # Update parent class sample rate
                    self._settings.sample_rate = server_sample_rate

                logger.info(
                    f"MagpieHTTPTTS config: sample_rate={server_sample_rate}Hz, "
                    f"voices={config.get('voices', [])}"
                )
            self._config_fetched = True
        except Exception as e:
            logger.warning(f"Failed to fetch TTS config: {e}")
            self._config_fetched = True  # Don't retry on every request

    def can_generate_metrics(self) -> bool:
        """Check if this service can generate processing metrics."""
        return True

    async def cancel(self, frame: CancelFrame):
        self._generation_id += 1
        logger.info(f"MagpieHTTPTTS: cancel current_generation={self._generation_id}")
        await self.stop_all_metrics()
        await super().cancel(frame)

    async def _handle_interruption(self, frame: InterruptionFrame, direction: FrameDirection):
        self._generation_id += 1
        logger.info(
            f"MagpieHTTPTTS: interruption current_generation={self._generation_id} "
            f"direction={direction.name}"
        )
        await self.stop_all_metrics()
        await super()._handle_interruption(frame, direction)

    async def run_tts(self, text: str) -> AsyncGenerator[Frame, None]:
        """Generate speech from text using Magpie TTS HTTP server.

        Args:
            text: The text to synthesize.

        Yields:
            TTSStartedFrame, TTSAudioRawFrame, TTSStoppedFrame
        """
        text = sanitize_text_for_tts(text)
        if not text:
            return

        self._generation_id += 1
        generation_id = self._generation_id

        # Fetch config on first request
        await self._ensure_config()

        await self.start_ttfb_metrics()
        logger.info(
            f"MagpieHTTPTTS: request_start generation={generation_id} "
            f"chars={len(text)} text={text!r}"
        )
        yield TTSStartedFrame()

        try:
            # Make HTTP request to TTS server
            request_started = time.time()
            resp = await self._client.post(
                f"{self._server_url}/v1/audio/speech",
                json={
                    "input": text,
                    "voice": self._voice,
                    "language": self._params.language,
                    "response_format": "pcm",
                },
            )
            request_elapsed_ms = (time.time() - request_started) * 1000
            logger.info(
                f"MagpieHTTPTTS: response_status generation={generation_id} "
                f"status={resp.status_code} elapsed_ms={request_elapsed_ms:.0f} "
                f"rtf={resp.headers.get('X-RTF')} "
                f"duration_ms={resp.headers.get('X-Duration-Ms')}"
            )

            if resp.status_code != 200:
                error_msg = f"TTS server error: {resp.status_code} - {resp.text}"
                logger.error(error_msg)
                yield ErrorFrame(error=error_msg)
                return

            await self.stop_ttfb_metrics()
            if generation_id != self._generation_id:
                logger.info(
                    f"MagpieHTTPTTS: generation_cancelled generation={generation_id} "
                    f"current_generation={self._generation_id}"
                )
                return

            audio_bytes = resp.content

            # Get sample rate from response headers or use default
            sample_rate = int(resp.headers.get("X-Sample-Rate", self._sample_rate or DEFAULT_SAMPLE_RATE))
            duration_ms = float(resp.headers.get("X-Duration-Ms", 0))

            logger.info(
                f"MagpieHTTPTTS: yield_audio generation={generation_id} "
                f"bytes={len(audio_bytes)} sample_rate={sample_rate} duration_ms={duration_ms:.0f}"
            )

            yield TTSAudioRawFrame(
                audio=audio_bytes,
                sample_rate=sample_rate,
                num_channels=1,
            )

            await self.start_tts_usage_metrics(text)

        except httpx.ConnectError as e:
            error_msg = f"Cannot connect to TTS server at {self._server_url}: {e}"
            logger.error(error_msg)
            yield ErrorFrame(error=error_msg)

        except Exception as e:
            logger.error(f"MagpieHTTPTTS error: {e}")
            yield ErrorFrame(error=str(e))
        finally:
            await self.stop_ttfb_metrics()
            logger.info(f"MagpieHTTPTTS: yield_stopped generation={generation_id}")
            yield TTSStoppedFrame()
            await self.push_frame(
                ChunkedLLMContinueGenerationFrame(),
                FrameDirection.UPSTREAM,
            )

    async def close(self):
        """Close HTTP client."""
        await self._client.aclose()

    def set_voice(self, voice: str):
        """Change the speaker voice.

        Args:
            voice: Speaker name (john, sofia, aria, jason, leo).
        """
        self._voice = voice.lower()
        super().set_voice(voice)
        logger.info(f"MagpieHTTPTTS: Voice changed to {voice}")

    def set_language(self, language: str):
        """Change the language.

        Args:
            language: Language code (en, es, de, fr, vi, it, zh).
        """
        self._language = language.lower()
        logger.info(f"MagpieHTTPTTS: Language changed to {language}")
