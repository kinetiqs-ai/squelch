"""Voxtral Realtime STT service for Pipecat.

This adapter targets vLLM's ``/v1/realtime`` websocket endpoint for
``mistralai/Voxtral-Mini-4B-Realtime-2602``.
"""

import asyncio
import base64
import json
import time
from typing import AsyncGenerator, Optional

import websockets
from loguru import logger

from pipecat.frames.frames import (
    AudioRawFrame,
    CancelFrame,
    EndFrame,
    ErrorFrame,
    Frame,
    InterimTranscriptionFrame,
    MetricsFrame,
    StartFrame,
    TranscriptionFrame,
    UserStartedSpeakingFrame,
    UserStoppedSpeakingFrame,
    VADUserStartedSpeakingFrame,
    VADUserStoppedSpeakingFrame,
)
from pipecat.metrics.metrics import TTFBMetricsData
from pipecat.processors.frame_processor import FrameDirection
from pipecat.services.stt_service import WebsocketSTTService
from pipecat.utils.time import time_now_iso8601


class VoxtralRealtimeSTTService(WebsocketSTTService):
    """Streaming STT via vLLM Realtime API.

    vLLM's realtime flow is:
    ``session.update`` once per websocket, then for each utterance:
    ``commit(final=False)`` to start generation, ``append`` audio chunks, and
    ``commit(final=True)`` to end the audio stream.

    Pipecat raw VAD can split one user turn into multiple speech fragments.
    This service starts Voxtral on the first speech start, keeps feeding audio
    across short VAD gaps, and finalizes only on Pipecat's user-turn stop.
    """

    def __init__(
        self,
        *,
        url: str = "ws://localhost:8082/v1/realtime",
        sample_rate: int = 16000,
        model: Optional[str] = None,
        language: str = "en",
        pending_frame_timeout_s: float = 1.0,
        final_padding_ms: int = 480,
        prebuffer_ms: int = 4000,
        **kwargs,
    ):
        super().__init__(sample_rate=sample_rate, **kwargs)
        self._url = url
        self._model = model
        self._language = language

        self._websocket = None
        self._receive_task: Optional[asyncio.Task] = None
        self._ready = False
        self._send_lock = asyncio.Lock()

        self._user_speaking = False
        self._turn_active = False
        self._next_turn_pending = False
        self._stream_started = False
        self._final_sent = False
        self._audio_bytes_sent = 0
        self._current_delta = ""
        self._audio_buffer = bytearray()
        self._audio_buffer_size = (sample_rate * 2 * prebuffer_ms) // 1000

        self._pending_user_stopped_frame: Optional[UserStoppedSpeakingFrame] = None
        self._pending_frame_direction: FrameDirection = FrameDirection.DOWNSTREAM
        self._pending_frame_timeout_task: Optional[asyncio.Task] = None
        self._pending_frame_timeout_s = pending_frame_timeout_s
        self._vad_stopped_time: Optional[float] = None
        self._final_padding_ms = final_padding_ms

    def can_generate_metrics(self) -> bool:
        return True

    async def start(self, frame: StartFrame):
        await super().start(frame)
        self._audio_buffer_size = max(self.sample_rate * 2, self._audio_buffer_size)
        await self._connect()

    async def stop(self, frame: EndFrame):
        await self._cancel_pending_frame_timeout()
        if self._pending_user_stopped_frame:
            await self.push_frame(self._pending_user_stopped_frame, self._pending_frame_direction)
            self._pending_user_stopped_frame = None
        if self._stream_started and not self._final_sent:
            await self._finalize_stream()
        await super().stop(frame)
        await self._disconnect()

    async def cancel(self, frame: CancelFrame):
        await self._cancel_pending_frame_timeout()
        self._pending_user_stopped_frame = None
        self._reset_turn_state()
        await super().cancel(frame)
        await self._disconnect()

    async def run_stt(self, audio: bytes) -> AsyncGenerator[Frame, None]:
        # Audio is handled in process_audio_frame so we can preserve Pipecat's
        # VAD prebuffer semantics while still using WebsocketSTTService.
        yield None

    async def process_audio_frame(self, frame: AudioRawFrame, direction: FrameDirection):
        if hasattr(frame, "user_id"):
            self._user_id = frame.user_id
        else:
            self._user_id = ""

        if not frame.audio:
            logger.warning(f"Empty audio frame received for STT service: {self.name}")
            return

        if self._turn_active and not self._final_sent:
            await self._start_stream_if_needed()
            await self._flush_audio_buffer()
            await self._send_audio(frame.audio)
            return

        if self._final_sent and not self._next_turn_pending:
            return

        self._audio_buffer += frame.audio
        if len(self._audio_buffer) > self._audio_buffer_size:
            discarded = len(self._audio_buffer) - self._audio_buffer_size
            self._audio_buffer = self._audio_buffer[discarded:]

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        if isinstance(frame, (UserStartedSpeakingFrame, VADUserStartedSpeakingFrame)):
            await self._handle_user_started_speaking()
            await super().process_frame(frame, direction)
            return

        if isinstance(frame, VADUserStoppedSpeakingFrame):
            await super().process_frame(frame, direction)
            await self._handle_vad_user_stopped_speaking()
            return

        if isinstance(frame, UserStoppedSpeakingFrame):
            if self._turn_active and self._stream_started and not self._final_sent:
                self._pending_user_stopped_frame = frame
                self._pending_frame_direction = direction
                self._start_pending_frame_timeout()
                self._vad_stopped_time = time.time()
                await self._finalize_stream()
                return
            if self._final_sent:
                self._pending_user_stopped_frame = frame
                self._pending_frame_direction = direction
                self._start_pending_frame_timeout()
                self._vad_stopped_time = time.time()
                return
            await super().process_frame(frame, direction)
            return

        await super().process_frame(frame, direction)

    async def _handle_user_started_speaking(self):
        await self._cancel_pending_frame_timeout()
        self._pending_user_stopped_frame = None
        if self._final_sent:
            logger.debug(f"{self} user started while final transcript pending; buffering next turn")
            self._user_speaking = True
            self._turn_active = True
            self._next_turn_pending = True
            return
        if not self._turn_active:
            self._current_delta = ""
        self._user_speaking = True
        self._turn_active = True
        self._vad_stopped_time = None
        await self._start_stream_if_needed()
        await self._flush_audio_buffer()

    async def _handle_vad_user_stopped_speaking(self):
        self._user_speaking = False
        if self._turn_active and self._stream_started and not self._final_sent:
            self._vad_stopped_time = time.time()
            logger.debug(f"{self} VAD stopped; keeping realtime stream open until user turn stop")

    async def _connect(self):
        await self._connect_websocket()
        self._receive_task = asyncio.create_task(self._receive_task_handler(self._report_error))
        await self._call_event_handler("on_connected", self)
        logger.info(f"{self} connected and ready")

    async def _connect_websocket(self):
        logger.debug(f"{self} connecting to {self._url}")
        self._websocket = await websockets.connect(self._url)
        self._ready = True

        try:
            message = await asyncio.wait_for(self._websocket.recv(), timeout=2.0)
            data = json.loads(message)
            logger.debug(f"{self} initial realtime event: {data.get('type')}")
        except asyncio.TimeoutError:
            logger.debug(f"{self} no initial session event from realtime server")

        if self._model:
            await self._websocket.send(
                json.dumps(
                    {
                        "type": "session.update",
                        "model": self._model,
                        "language": self._language,
                    }
                )
            )

    async def _disconnect(self):
        logger.debug(f"{self} disconnecting")
        self._ready = False
        if self._receive_task:
            self._receive_task.cancel()
            try:
                await self._receive_task
            except asyncio.CancelledError:
                pass
            self._receive_task = None
        await self._disconnect_websocket()
        await self._call_event_handler("on_disconnected", self)

    async def _disconnect_websocket(self):
        if self._websocket:
            try:
                await self._websocket.close()
            except Exception as exc:
                logger.debug(f"{self} error closing websocket: {exc}")
            self._websocket = None

    async def _start_stream_if_needed(self):
        if self._stream_started or not (self._websocket and self._ready):
            return
        await self._send_commit(final=False)
        self._stream_started = True
        self._final_sent = False
        logger.debug(f"{self} started realtime generation stream")

    async def _finalize_stream(self):
        if not (self._websocket and self._ready and self._stream_started and not self._final_sent):
            return
        await self._send_final_padding()
        await self._send_commit(final=True)
        self._final_sent = True
        self._user_speaking = False
        self._turn_active = False
        self._audio_buffer.clear()

    async def _send_commit(self, *, final: bool):
        try:
            async with self._send_lock:
                await self._websocket.send(
                    json.dumps(
                        {
                            "type": "input_audio_buffer.commit",
                            "final": final,
                        }
                    )
                )
                samples = self._audio_bytes_sent // 2
                duration_ms = (samples * 1000) // self.sample_rate
                logger.debug(f"{self} committed audio final={final} audio={duration_ms}ms")
        except Exception as exc:
            logger.error(f"{self} failed to commit audio: {exc}")
            await self._report_error(ErrorFrame(f"Failed to commit audio: {exc}"))

    async def _send_audio(self, audio: bytes):
        if not (self._websocket and self._ready and audio):
            return
        try:
            async with self._send_lock:
                await self._websocket.send(
                    json.dumps(
                        {
                            "type": "input_audio_buffer.append",
                            "audio": base64.b64encode(audio).decode("utf-8"),
                        }
                    )
                )
                self._audio_bytes_sent += len(audio)
        except Exception as exc:
            logger.error(f"{self} failed to send audio: {exc}")
            await self._report_error(ErrorFrame(f"Failed to send audio: {exc}"))

    async def _send_final_padding(self):
        if self._final_padding_ms <= 0 or self._audio_bytes_sent == 0:
            return
        samples = (self.sample_rate * self._final_padding_ms) // 1000
        silence = b"\x00\x00" * samples
        logger.debug(f"{self} appending final silence padding audio={self._final_padding_ms}ms")
        await self._send_audio(silence)

    async def _flush_audio_buffer(self):
        if not self._audio_buffer:
            return
        audio = bytes(self._audio_buffer)
        self._audio_buffer.clear()
        samples = len(audio) // 2
        duration_ms = (samples * 1000) // self.sample_rate
        logger.debug(f"{self} flushing VAD prebuffer audio={duration_ms}ms")
        await self._send_audio(audio)

    async def _receive_messages(self):
        if not self._websocket:
            return
        async for message in self._websocket:
            try:
                data = json.loads(message)
            except json.JSONDecodeError as exc:
                logger.error(f"{self} invalid JSON from realtime server: {exc}")
                continue

            msg_type = data.get("type")
            if msg_type in {"transcription.delta", "response.text.delta"}:
                await self._handle_delta(data)
            elif msg_type in {"transcription.done", "response.done"}:
                await self._handle_done(data)
            elif msg_type == "error":
                error = data.get("error") or data.get("message") or data
                logger.error(f"{self} realtime error: {error}")
                await self._report_error(ErrorFrame(f"Voxtral realtime error: {error}"))
            elif msg_type in {"session.created", "session.updated"}:
                logger.debug(f"{self} realtime event: {msg_type}")
            else:
                logger.trace(f"{self} realtime event ignored: {msg_type}")

    async def _handle_delta(self, data: dict):
        delta = data.get("delta", "")
        if not delta:
            return
        self._current_delta += delta
        await self.push_frame(
            InterimTranscriptionFrame(
                self._current_delta,
                self._user_id,
                time_now_iso8601(),
                language=None,
            )
        )

    async def _handle_done(self, data: dict):
        text = (data.get("text") or self._current_delta).strip()
        logger.debug(f"{self} final transcript: {text}")
        await self.stop_ttfb_metrics()

        if text:
            await self.push_frame(
                TranscriptionFrame(
                    text,
                    self._user_id,
                    time_now_iso8601(),
                    language=None,
                )
            )
            await self.stop_processing_metrics()

            if self._vad_stopped_time is not None:
                processing_time = time.time() - self._vad_stopped_time
                logger.info(f"{self} VoxtralSTT TTFB: {processing_time * 1000:.0f}ms")
                await self.push_frame(
                    MetricsFrame(
                        data=[
                            TTFBMetricsData(
                                processor="VoxtralSTT",
                                value=processing_time,
                            )
                        ]
                    )
                )

        next_turn_audio = bytes(self._audio_buffer) if self._next_turn_pending else b""
        await self._release_pending_frame()
        self._reset_turn_state()
        if next_turn_audio:
            logger.debug(f"{self} buffered next-turn audio while final pending")
            self._audio_buffer += next_turn_audio
            self._user_speaking = True
            self._turn_active = True
            self._next_turn_pending = False
            await self._start_stream_if_needed()
            await self._flush_audio_buffer()

    def _start_pending_frame_timeout(self):
        if self._pending_frame_timeout_task:
            self._pending_frame_timeout_task.cancel()
        self._pending_frame_timeout_task = asyncio.create_task(
            self._pending_frame_timeout_handler()
        )

    async def _pending_frame_timeout_handler(self):
        try:
            await asyncio.sleep(self._pending_frame_timeout_s)
            if self._pending_user_stopped_frame:
                logger.debug(f"{self} timeout waiting for final transcript")
                await self._push_current_delta_as_final()
                await self.push_frame(
                    self._pending_user_stopped_frame,
                    self._pending_frame_direction,
                )
                self._pending_user_stopped_frame = None
                self._reset_turn_state()
                await self._recover_websocket_after_timeout()
        except asyncio.CancelledError:
            pass

    async def _cancel_pending_frame_timeout(self):
        if self._pending_frame_timeout_task:
            self._pending_frame_timeout_task.cancel()
            try:
                await self._pending_frame_timeout_task
            except asyncio.CancelledError:
                pass
            self._pending_frame_timeout_task = None

    async def _release_pending_frame(self):
        if self._pending_user_stopped_frame:
            await self._cancel_pending_frame_timeout()
            await self.push_frame(
                self._pending_user_stopped_frame,
                self._pending_frame_direction,
            )
            self._pending_user_stopped_frame = None

    async def _push_current_delta_as_final(self):
        text = self._current_delta.strip()
        if not text:
            return
        logger.debug(f"{self} promoting interim transcript to final: {text}")
        await self.push_frame(
            TranscriptionFrame(
                text,
                self._user_id,
                time_now_iso8601(),
                language=None,
            )
        )
        await self.stop_processing_metrics()

    async def _recover_websocket_after_timeout(self):
        logger.debug(f"{self} reconnecting realtime websocket after finalization timeout")
        await self._disconnect_websocket()
        await self._connect_websocket()
        if not self._receive_task or self._receive_task.done():
            self._receive_task = asyncio.create_task(self._receive_task_handler(self._report_error))

    def _reset_turn_state(self):
        self._user_speaking = False
        self._turn_active = False
        self._next_turn_pending = False
        self._stream_started = False
        self._final_sent = False
        self._audio_bytes_sent = 0
        self._current_delta = ""
        self._audio_buffer.clear()
        self._vad_stopped_time = None

    async def start_metrics(self):
        await self.start_ttfb_metrics()
        await self.start_processing_metrics()
