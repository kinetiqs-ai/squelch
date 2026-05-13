"""Native ASR -> LLM -> TTS turn orchestration for audio-edge-agent sessions."""

from __future__ import annotations

import asyncio
import array
import json
import logging
import os
import re
import time
from dataclasses import dataclass, field
from typing import Any

import aiohttp
from audio_edge_agent.protocol import (
    AssistantAudioStart,
    AssistantAudioStop,
    FrameHeader,
    _now_us,
)

from native_voice.diagnostics import SessionDiagnostics

logger = logging.getLogger("uvicorn.error")


@dataclass
class VoiceOrchestratorConfig:
    llm_url: str = field(default_factory=lambda: os.getenv("SQUELCH_LLM_URL", "http://127.0.0.1:8100"))
    tts_url: str = field(default_factory=lambda: os.getenv("SQUELCH_TTS_URL", "http://127.0.0.1:8101"))
    tts_batch_url: str = field(default_factory=lambda: os.getenv("SQUELCH_TTS_BATCH_URL", "http://127.0.0.1:8102"))
    tts_mode: str = field(default_factory=lambda: os.getenv("SQUELCH_TTS_MODE", "stream"))
    tts_stream_preset: str = field(default_factory=lambda: os.getenv("SQUELCH_TTS_STREAM_PRESET", "quality"))
    tts_start_stream_preset: str = field(default_factory=lambda: os.getenv("SQUELCH_TTS_START_STREAM_PRESET", "startup_quality"))
    tts_segment_chars: int = field(default_factory=lambda: int(os.getenv("SQUELCH_TTS_SEGMENT_CHARS", "120")))
    voice: str = field(default_factory=lambda: os.getenv("SQUELCH_TTS_VOICE", "aria"))
    language: str = field(default_factory=lambda: os.getenv("SQUELCH_TTS_LANGUAGE", "en"))
    max_tokens: int = field(default_factory=lambda: int(os.getenv("SQUELCH_LLM_MAX_TOKENS", "96")))
    temperature: float = field(default_factory=lambda: float(os.getenv("SQUELCH_LLM_TEMPERATURE", "0.2")))
    frame_duration_ms: int = field(default_factory=lambda: int(os.getenv("SQUELCH_RETURN_FRAME_MS", "20")))
    return_preroll_ms: int = field(default_factory=lambda: int(os.getenv("SQUELCH_RETURN_PREROLL_MS", "0")))
    return_fade_in_ms: int = field(default_factory=lambda: int(os.getenv("SQUELCH_RETURN_FADE_IN_MS", "0")))
    request_timeout_s: float = field(default_factory=lambda: float(os.getenv("SQUELCH_AGENT_TIMEOUT_S", "120")))
    stream_llm_to_tts: bool = field(default_factory=lambda: os.getenv("SQUELCH_STREAM_LLM_TO_TTS", "0") == "1")


class VoiceOrchestrator:
    def __init__(
        self,
        session_id: str,
        outbound: asyncio.Queue[str | bytes | None],
        diagnostics: SessionDiagnostics,
        config: VoiceOrchestratorConfig | None = None,
    ) -> None:
        self.session_id = session_id
        self.outbound = outbound
        self.diagnostics = diagnostics
        self.config = config or VoiceOrchestratorConfig()
        self._lock = asyncio.Lock()
        self._tts_stream_lock = asyncio.Lock()
        self._tts_session: aiohttp.ClientSession | None = None
        self._tts_ws: aiohttp.ClientWebSocketResponse | None = None
        self._tts_warmup_task: asyncio.Task[None] | None = None
        self._tts_primed = False
        self._turn = 0
        self._history: list[dict[str, str]] = [
            {
                "role": "system",
                "content": (
                    "You are Squelch, a concise local voice assistant. "
                    "If asked what model is running, say this system is using Nemotron-3-Nano-30B-A3B Q4_K_M through llama.cpp. "
                    "Answer the user's latest utterance fully and directly at the level of detail requested. "
                    "Do not introduce yourself unless asked."
                ),
            }
        ]

    def schedule_final_transcript(self, text: str) -> asyncio.Task[None] | None:
        text = text.strip()
        if not text:
            return None
        return asyncio.create_task(self._run_turn(text))

    def start_tts_warmup(self) -> None:
        if self.config.tts_mode != "stream":
            return
        if self._tts_warmup_task is None or self._tts_warmup_task.done():
            self._tts_warmup_task = asyncio.create_task(self._warm_tts_stream())

    async def close(self) -> None:
        if self._tts_warmup_task is not None and not self._tts_warmup_task.done():
            self._tts_warmup_task.cancel()
            try:
                await self._tts_warmup_task
            except asyncio.CancelledError:
                pass
        async with self._tts_stream_lock:
            await self._close_tts_stream_locked()

    async def _warm_tts_stream(self) -> None:
        try:
            await self._ensure_tts_stream()
            await self._emit_event("tts_stream_warmed")
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # noqa: BLE001 - user-facing turns can recover through batch fallback.
            logger.exception("native_voice_tts_warmup_failed")
            await self._emit_event("tts_stream_warmup_error", message=str(exc))

    async def _run_turn(self, user_text: str) -> None:
        async with self._lock:
            self._turn += 1
            turn_id = self._turn
            started = time.monotonic()
            await self._emit_event("turn_start", turn_id=turn_id, text=user_text)
            self._history.append({"role": "user", "content": user_text})

            try:
                llm_started = time.monotonic()
                await self._emit_event("llm_start", turn_id=turn_id)
                if self.config.tts_mode == "stream" and self.config.stream_llm_to_tts:
                    tts_started = time.monotonic()
                    await self._emit_event("tts_start", turn_id=turn_id)
                    try:
                        assistant_text, metrics, llm_elapsed_ms = await self._stream_llm_to_tts(turn_id)
                    except Exception as exc:  # noqa: BLE001 - keep the known-good full-response path as recovery.
                        await self._emit_event(
                            "llm_tts_stream_error",
                            turn_id=turn_id,
                            message=str(exc),
                        )
                        logger.exception("native_voice_llm_tts_stream_failed turn=%s", turn_id)
                        assistant_text = await self._complete_llm()
                        metrics = await self._stream_tts(turn_id, assistant_text)
                else:
                    assistant_text = await self._complete_llm()
                    llm_elapsed_ms = int((time.monotonic() - llm_started) * 1000)
                    tts_started = time.monotonic()
                    await self._emit_event("tts_start", turn_id=turn_id)
                    if self.config.tts_mode == "hybrid":
                        try:
                            metrics = await self._hybrid_tts(turn_id, assistant_text)
                        except Exception as exc:  # noqa: BLE001 - keep the known-good batch path as recovery.
                            await self._emit_event(
                                "tts_hybrid_error",
                                turn_id=turn_id,
                                message=str(exc),
                            )
                            logger.exception("native_voice_tts_hybrid_failed turn=%s", turn_id)
                            audio = await self._synthesize_tts(assistant_text)
                            metrics = await self._send_assistant_audio(turn_id, audio)
                    elif self.config.tts_mode == "stream":
                        try:
                            metrics = await self._stream_tts(turn_id, assistant_text)
                        except Exception as exc:  # noqa: BLE001 - keep the known-good batch path as recovery.
                            await self._emit_event(
                                "tts_stream_error",
                                turn_id=turn_id,
                                message=str(exc),
                            )
                            logger.exception("native_voice_tts_stream_failed turn=%s", turn_id)
                            audio = await self._synthesize_tts(assistant_text)
                            metrics = await self._send_assistant_audio(turn_id, audio)
                    else:
                        audio = await self._synthesize_tts(assistant_text)
                        metrics = await self._send_assistant_audio(turn_id, audio)

                self._history.append({"role": "assistant", "content": assistant_text})
                self._trim_history()
                await self._emit_event(
                    "llm_final",
                    turn_id=turn_id,
                    text=assistant_text,
                    elapsed_ms=llm_elapsed_ms,
                )
                tts_elapsed_ms = int((time.monotonic() - tts_started) * 1000)
                await self._emit_event(
                    "tts_complete",
                    turn_id=turn_id,
                    byte_count=metrics.byte_count,
                    sample_rate=metrics.sample_rate,
                    duration_ms=metrics.duration_ms,
                    frame_count=metrics.frame_count,
                    first_audio_ms=metrics.first_audio_ms,
                    mode=metrics.mode,
                    rtf=metrics.rtf,
                    elapsed_ms=tts_elapsed_ms,
                )

                await self._emit_event(
                    "turn_complete",
                    turn_id=turn_id,
                    elapsed_ms=int((time.monotonic() - started) * 1000),
                )
            except Exception as exc:  # noqa: BLE001 - surface errors to the edge client.
                logger.exception("native_voice_turn_failed turn=%s", turn_id)
                await self._emit_event("assistant_error", turn_id=turn_id, message=str(exc))

    async def _stream_llm_to_tts(self, turn_id: int) -> tuple[str, "TtsMetrics", int]:
        sample_rate = 22050
        framer = AssistantAudioFramer(
            session_id=self.session_id,
            turn_id=turn_id,
            outbound=self.outbound,
            sample_rate=sample_rate,
            frame_duration_ms=self.config.frame_duration_ms,
            preroll_ms=self.config.return_preroll_ms,
            fade_in_ms=self.config.return_fade_in_ms,
        )
        await framer.start()
        started = time.monotonic()
        assistant_parts: list[str] = []
        segmenter = StreamingTextSegmenter(max_chars=self.config.tts_segment_chars)
        sent_segments = 0

        timeout = aiohttp.ClientTimeout(total=self.config.request_timeout_s)
        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.ws_connect(f"{self.config.tts_url}/ws/tts/stream") as websocket:
                await websocket.send_json(
                    {
                        "type": "init",
                        "voice": self.config.voice,
                        "language": self.config.language,
                        "default_mode": "stream",
                    }
                )

                async def read_tts_audio() -> None:
                    async for message in websocket:
                        if message.type == aiohttp.WSMsgType.BINARY:
                            first_audio_ms = await framer.write(message.data)
                            if first_audio_ms is not None:
                                await self._emit_event(
                                    "tts_first_audio",
                                    turn_id=turn_id,
                                    elapsed_ms=int((time.monotonic() - started) * 1000),
                                )
                            continue
                        if message.type != aiohttp.WSMsgType.TEXT:
                            continue
                        event = json.loads(message.data)
                        event_type = event.get("type")
                        if event_type == "error":
                            raise RuntimeError(str(event.get("message", "TTS stream failed")))
                        if event_type == "done":
                            break

                async def send_segment(segment: str) -> None:
                    nonlocal sent_segments
                    sent_segments += 1
                    preset = self.config.tts_start_stream_preset if sent_segments == 1 else self.config.tts_stream_preset
                    await self._emit_event(
                        "tts_text_segment",
                        turn_id=turn_id,
                        segment=sent_segments,
                        chars=len(segment),
                        preset=preset,
                        text=segment,
                    )
                    await websocket.send_json(
                        {
                            "type": "text",
                            "text": segment,
                            "mode": "stream",
                            "preset": preset,
                        }
                    )

                reader = asyncio.create_task(read_tts_audio())
                try:
                    async for delta in self._stream_llm():
                        assistant_parts.append(delta)
                        for segment in segmenter.push(delta):
                            await send_segment(segment)
                    for segment in segmenter.flush():
                        await send_segment(segment)
                    if sent_segments == 0:
                        raise RuntimeError("LLM returned empty completion")
                    assistant_text = _clean_assistant_text("".join(assistant_parts))
                    if not assistant_text:
                        raise RuntimeError("LLM returned empty completion")
                    llm_elapsed_ms = int((time.monotonic() - started) * 1000)
                    await websocket.send_json({"type": "close"})
                    await reader
                finally:
                    if not reader.done():
                        reader.cancel()
                        try:
                            await reader
                        except asyncio.CancelledError:
                            pass

        metrics = await framer.close()
        return assistant_text, TtsMetrics(
            byte_count=metrics.byte_count,
            sample_rate=metrics.sample_rate,
            duration_ms=metrics.duration_ms,
            frame_count=metrics.frame_count,
            first_audio_ms=metrics.first_audio_ms,
            mode="llm_stream_tts_stream",
            rtf=None,
        ), llm_elapsed_ms

    async def _stream_llm(self):
        prompt = self._render_prompt()
        payload = {
            "prompt": prompt,
            "n_predict": self.config.max_tokens,
            "temperature": self.config.temperature,
            "repeat_penalty": 1.05,
            "stream": True,
            "stop": ["<|im_end|>", "<|im_start|>"],
        }
        timeout = aiohttp.ClientTimeout(total=self.config.request_timeout_s)
        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.post(f"{self.config.llm_url}/completion", json=payload) as response:
                response.raise_for_status()
                buffer = ""
                async for chunk in response.content.iter_any():
                    buffer += chunk.decode("utf-8", errors="replace")
                    while "\n" in buffer:
                        line, buffer = buffer.split("\n", 1)
                        delta, stopped = _parse_llm_stream_line(line)
                        if delta:
                            yield delta
                        if stopped:
                            return
                if buffer:
                    delta, _ = _parse_llm_stream_line(buffer)
                    if delta:
                        yield delta

    async def _complete_llm(self) -> str:
        prompt = self._render_prompt()
        payload = {
            "prompt": prompt,
            "n_predict": self.config.max_tokens,
            "temperature": self.config.temperature,
            "repeat_penalty": 1.05,
            "stream": False,
            "stop": ["<|im_end|>", "<|im_start|>"],
        }
        timeout = aiohttp.ClientTimeout(total=self.config.request_timeout_s)
        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.post(f"{self.config.llm_url}/completion", json=payload) as response:
                response.raise_for_status()
                data = await response.json()

        text = str(data.get("content") or data.get("text") or "").strip()
        if not text:
            raise RuntimeError("LLM returned empty completion")
        return _clean_assistant_text(text)

    async def _synthesize_tts(self, text: str, tts_url: str | None = None) -> "SynthesizedAudio":
        tts_url = tts_url or self.config.tts_url
        payload = {
            "input": text,
            "voice": self.config.voice,
            "language": self.config.language,
            "response_format": "pcm",
        }
        timeout = aiohttp.ClientTimeout(total=self.config.request_timeout_s)
        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.post(f"{tts_url}/v1/audio/speech", json=payload) as response:
                response.raise_for_status()
                pcm = await response.read()
                sample_rate = int(response.headers.get("X-Sample-Rate", "22050"))
                channels = int(response.headers.get("X-Channels", "1"))
                duration_ms = int(response.headers.get("X-Duration-Ms", "0") or "0")
                rtf = response.headers.get("X-RTF")

        if channels != 1:
            raise RuntimeError(f"TTS returned unsupported channel count: {channels}")
        if not pcm:
            raise RuntimeError("TTS returned empty audio")
        if duration_ms <= 0:
            duration_ms = int(len(pcm) / (sample_rate * 2) * 1000)
        return SynthesizedAudio(pcm=pcm, sample_rate=sample_rate, duration_ms=duration_ms, rtf=rtf)

    async def _hybrid_tts(self, turn_id: int, text: str) -> "TtsMetrics":
        segments = _split_tts_segments(text)
        if not segments:
            raise RuntimeError("No TTS segments produced")

        sample_rate = 22050
        framer = AssistantAudioFramer(
            session_id=self.session_id,
            turn_id=turn_id,
            outbound=self.outbound,
            sample_rate=sample_rate,
            frame_duration_ms=self.config.frame_duration_ms,
            preroll_ms=self.config.return_preroll_ms,
            fade_in_ms=self.config.return_fade_in_ms,
        )
        await framer.start()
        started = time.monotonic()
        first_segment = segments[0]
        remaining_segments = segments[1:]
        await self._emit_event(
            "tts_hybrid_segments",
            turn_id=turn_id,
            segment_count=len(segments),
            first_segment_chars=len(first_segment),
            remaining_segment_count=len(remaining_segments),
        )

        async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=self.config.request_timeout_s)) as session:
            async with session.ws_connect(f"{self.config.tts_url}/ws/tts/stream") as websocket:
                batch_queue: asyncio.Queue[tuple[int, SynthesizedAudio | Exception | None]] = asyncio.Queue()
                batch_task = (
                    asyncio.create_task(
                        self._produce_batch_segments(remaining_segments, batch_queue)
                    )
                    if remaining_segments
                    else None
                )
                try:
                    await websocket.send_json(
                        {
                            "type": "init",
                            "voice": self.config.voice,
                            "language": self.config.language,
                            "default_mode": "stream",
                        }
                    )
                    await websocket.send_json(
                        {
                            "type": "text",
                            "text": first_segment,
                            "mode": "stream",
                            "preset": self.config.tts_stream_preset,
                        }
                    )
                    await websocket.send_json({"type": "close"})

                    async for message in websocket:
                        if message.type == aiohttp.WSMsgType.BINARY:
                            first_audio_ms = await framer.write(message.data)
                            if first_audio_ms is not None:
                                await self._emit_event(
                                    "tts_first_audio",
                                    turn_id=turn_id,
                                    elapsed_ms=int((time.monotonic() - started) * 1000),
                                )
                            continue
                        if message.type != aiohttp.WSMsgType.TEXT:
                            continue
                        event = json.loads(message.data)
                        event_type = event.get("type")
                        if event_type == "error":
                            raise RuntimeError(str(event.get("message", "TTS stream failed")))
                        if event_type == "done":
                            break
                finally:
                    if batch_task is not None and batch_task.done() and batch_task.exception() is not None:
                        batch_task.exception()

                if batch_task is not None:
                    while True:
                        index, item = await batch_queue.get()
                        if item is None:
                            break
                        if isinstance(item, Exception):
                            raise item
                        audio = item
                        if audio.sample_rate != framer.sample_rate:
                            raise RuntimeError(
                                f"TTS segment {index} sample rate {audio.sample_rate} != {framer.sample_rate}"
                            )
                        await self._emit_event(
                            "tts_batch_segment_ready",
                            turn_id=turn_id,
                            segment=index,
                            byte_count=len(audio.pcm),
                            duration_ms=audio.duration_ms,
                            rtf=audio.rtf,
                        )
                        await framer.write(audio.pcm)
                    await batch_task

        metrics = await framer.close()
        return TtsMetrics(
            byte_count=metrics.byte_count,
            sample_rate=metrics.sample_rate,
            duration_ms=metrics.duration_ms,
            frame_count=metrics.frame_count,
            first_audio_ms=metrics.first_audio_ms,
            mode="hybrid",
            rtf=None,
        )

    async def _produce_batch_segments(
        self,
        segments: list[str],
        queue: asyncio.Queue[tuple[int, "SynthesizedAudio" | Exception | None]],
    ) -> None:
        try:
            for index, segment in enumerate(segments, start=2):
                await queue.put((index, await self._synthesize_tts(segment, self.config.tts_batch_url)))
            await queue.put((0, None))
        except Exception as exc:  # noqa: BLE001 - return through queue to preserve ordering logic.
            await queue.put((0, exc))

    async def _ensure_tts_stream(self) -> aiohttp.ClientWebSocketResponse:
        warmup_task = self._tts_warmup_task
        if (
            warmup_task is not None
            and warmup_task is not asyncio.current_task()
            and not warmup_task.done()
        ):
            await asyncio.shield(warmup_task)

        async with self._tts_stream_lock:
            if self._tts_ws is not None and not self._tts_ws.closed:
                return self._tts_ws

            await self._close_tts_stream_locked()
            timeout = aiohttp.ClientTimeout(total=self.config.request_timeout_s)
            self._tts_session = aiohttp.ClientSession(timeout=timeout)
            self._tts_ws = await self._tts_session.ws_connect(f"{self.config.tts_url}/ws/tts/stream")
            await self._tts_ws.send_json(
                {
                    "type": "init",
                    "voice": self.config.voice,
                    "language": self.config.language,
                    "default_mode": "stream",
                }
            )
            await self._wait_for_tts_event_locked({"stream_created"})

            if not self._tts_primed:
                await self._tts_ws.send_json(
                    {
                        "type": "text",
                        "text": "Okay.",
                        "mode": "stream",
                        "preset": self.config.tts_stream_preset,
                    }
                )
                await self._drain_tts_segment_locked()
                self._tts_primed = True

            return self._tts_ws

    async def _drain_tts_segment_locked(self) -> None:
        if self._tts_ws is None:
            raise RuntimeError("TTS stream is not connected")
        while True:
            message = await self._tts_ws.receive()
            if message.type == aiohttp.WSMsgType.BINARY:
                continue
            if message.type != aiohttp.WSMsgType.TEXT:
                raise RuntimeError(f"TTS stream closed during warmup: {message.type}")
            event = json.loads(message.data)
            event_type = event.get("type")
            if event_type == "error":
                raise RuntimeError(str(event.get("message", "TTS stream failed")))
            if event_type == "segment_complete":
                return

    async def _wait_for_tts_event_locked(self, event_types: set[str]) -> dict[str, Any]:
        if self._tts_ws is None:
            raise RuntimeError("TTS stream is not connected")
        while True:
            message = await self._tts_ws.receive()
            if message.type == aiohttp.WSMsgType.BINARY:
                continue
            if message.type != aiohttp.WSMsgType.TEXT:
                raise RuntimeError(f"TTS stream closed while waiting for {event_types}: {message.type}")
            event = json.loads(message.data)
            event_type = event.get("type")
            if event_type == "error":
                raise RuntimeError(str(event.get("message", "TTS stream failed")))
            if event_type in event_types:
                return event

    async def _close_tts_stream_locked(self) -> None:
        if self._tts_ws is not None:
            try:
                await self._tts_ws.close()
            finally:
                self._tts_ws = None
                self._tts_primed = False
        if self._tts_session is not None:
            try:
                await self._tts_session.close()
            finally:
                self._tts_session = None

    async def _stream_tts_persistent(self, turn_id: int, text: str) -> "TtsMetrics":
        sample_rate = 22050
        framer = AssistantAudioFramer(
            session_id=self.session_id,
            turn_id=turn_id,
            outbound=self.outbound,
            sample_rate=sample_rate,
            frame_duration_ms=self.config.frame_duration_ms,
            preroll_ms=self.config.return_preroll_ms,
            fade_in_ms=self.config.return_fade_in_ms,
        )
        await framer.start()
        started = time.monotonic()
        segments = _split_tts_segments(text, max_chars=self.config.tts_segment_chars)
        if not segments:
            raise RuntimeError("No TTS segments produced")

        websocket = await self._ensure_tts_stream()
        try:
            for segment in segments:
                await websocket.send_json(
                    {
                        "type": "text",
                        "text": segment,
                        "mode": "stream",
                        "preset": self.config.tts_stream_preset,
                    }
                )

            completed_segments = 0
            while completed_segments < len(segments):
                message = await websocket.receive()
                if message.type == aiohttp.WSMsgType.BINARY:
                    first_audio_ms = await framer.write(message.data)
                    if first_audio_ms is not None:
                        await self._emit_event(
                            "tts_first_audio",
                            turn_id=turn_id,
                            elapsed_ms=int((time.monotonic() - started) * 1000),
                        )
                    continue
                if message.type != aiohttp.WSMsgType.TEXT:
                    raise RuntimeError(f"TTS stream closed during response: {message.type}")
                event = json.loads(message.data)
                event_type = event.get("type")
                if event_type == "error":
                    raise RuntimeError(str(event.get("message", "TTS stream failed")))
                if event_type == "segment_complete":
                    completed_segments += 1
                elif event_type == "done":
                    break
        except Exception:
            async with self._tts_stream_lock:
                await self._close_tts_stream_locked()
                self._tts_primed = False
            raise

        metrics = await framer.close()
        return TtsMetrics(
            byte_count=metrics.byte_count,
            sample_rate=metrics.sample_rate,
            duration_ms=metrics.duration_ms,
            frame_count=metrics.frame_count,
            first_audio_ms=metrics.first_audio_ms,
            mode="stream",
            rtf=None,
        )

    async def _stream_tts(self, turn_id: int, text: str) -> "TtsMetrics":
        sample_rate = 22050
        framer = AssistantAudioFramer(
            session_id=self.session_id,
            turn_id=turn_id,
            outbound=self.outbound,
            sample_rate=sample_rate,
            frame_duration_ms=self.config.frame_duration_ms,
            preroll_ms=self.config.return_preroll_ms,
            fade_in_ms=self.config.return_fade_in_ms,
        )
        await framer.start()
        started = time.monotonic()
        timeout = aiohttp.ClientTimeout(total=self.config.request_timeout_s)
        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.ws_connect(f"{self.config.tts_url}/ws/tts/stream") as websocket:
                await websocket.send_json(
                    {
                        "type": "init",
                        "voice": self.config.voice,
                        "language": self.config.language,
                        "default_mode": "stream",
                    }
                )
                for index, segment in enumerate(_split_tts_segments(text, max_chars=self.config.tts_segment_chars)):
                    preset = self.config.tts_start_stream_preset if index == 0 else self.config.tts_stream_preset
                    await websocket.send_json(
                        {
                            "type": "text",
                            "text": segment,
                            "mode": "stream",
                            "preset": preset,
                        }
                    )
                await websocket.send_json({"type": "close"})

                async for message in websocket:
                    if message.type == aiohttp.WSMsgType.BINARY:
                        first_audio_ms = await framer.write(message.data)
                        if first_audio_ms is not None:
                            await self._emit_event(
                                "tts_first_audio",
                                turn_id=turn_id,
                                elapsed_ms=int((time.monotonic() - started) * 1000),
                            )
                        continue
                    if message.type != aiohttp.WSMsgType.TEXT:
                        continue
                    event = json.loads(message.data)
                    event_type = event.get("type")
                    if event_type == "error":
                        raise RuntimeError(str(event.get("message", "TTS stream failed")))
                    if event_type == "done":
                        break

        metrics = await framer.close()
        return TtsMetrics(
            byte_count=metrics.byte_count,
            sample_rate=metrics.sample_rate,
            duration_ms=metrics.duration_ms,
            frame_count=metrics.frame_count,
            first_audio_ms=metrics.first_audio_ms,
            mode="stream",
            rtf=None,
        )

    async def _send_assistant_audio(self, turn_id: int, audio: "SynthesizedAudio") -> "TtsMetrics":
        stream_id = f"turn-{turn_id}-{int(time.time() * 1000)}"
        samples_per_frame = max(1, int(audio.sample_rate * self.config.frame_duration_ms / 1000))
        frame_size = samples_per_frame * 2

        await self.outbound.put(
            AssistantAudioStart(
                session_id=self.session_id,
                stream_id=stream_id,
                sample_rate=audio.sample_rate,
                channels=1,
                sample_format="s16le",
                frame_size_bytes=frame_size,
                frame_duration_ms=self.config.frame_duration_ms,
            ).to_json_bytes().decode()
        )

        frame_count = 0
        for offset in range(0, len(audio.pcm), frame_size):
            payload = audio.pcm[offset : offset + frame_size]
            if len(payload) < frame_size:
                payload += b"\x00" * (frame_size - len(payload))
            header = FrameHeader(
                sequence=frame_count,
                timestamp_us=_now_us(),
                payload_length=len(payload),
            )
            await self.outbound.put(header.encode(payload))
            frame_count += 1

        await self.outbound.put(
            AssistantAudioStop(
                session_id=self.session_id,
                stream_id=stream_id,
                frame_count=frame_count,
            ).to_json_bytes().decode()
        )
        return TtsMetrics(
            byte_count=len(audio.pcm),
            sample_rate=audio.sample_rate,
            duration_ms=audio.duration_ms,
            frame_count=frame_count,
            first_audio_ms=0,
            mode="batch",
            rtf=audio.rtf,
        )

    async def _emit_event(self, event_type: str, **fields: Any) -> None:
        event = {
            "type": event_type,
            "session_id": self.session_id,
            "timestamp_us": _now_us(),
            **fields,
        }
        self.diagnostics.agent.write(event)
        await self.outbound.put(json.dumps(event, sort_keys=True))

    def _render_prompt(self) -> str:
        lines: list[str] = []
        for message in self._history[-9:]:
            role = message["role"]
            content = message["content"].strip()
            lines.append(f"<|im_start|>{role}\n{content}<|im_end|>")
        lines.append("<|im_start|>assistant\n<think></think>")
        return "\n".join(lines)

    def _trim_history(self) -> None:
        if len(self._history) <= 9:
            return
        system = self._history[:1]
        recent = self._history[-8:]
        self._history = system + recent


@dataclass(frozen=True)
class SynthesizedAudio:
    pcm: bytes
    sample_rate: int
    duration_ms: int
    rtf: str | None


@dataclass(frozen=True)
class TtsMetrics:
    byte_count: int
    sample_rate: int
    duration_ms: int
    frame_count: int
    first_audio_ms: int | None
    mode: str
    rtf: str | None


class AssistantAudioFramer:
    def __init__(
        self,
        session_id: str,
        turn_id: int,
        outbound: asyncio.Queue[str | bytes | None],
        sample_rate: int,
        frame_duration_ms: int,
        preroll_ms: int = 0,
        fade_in_ms: int = 0,
    ) -> None:
        self.session_id = session_id
        self.turn_id = turn_id
        self.outbound = outbound
        self.sample_rate = sample_rate
        self.frame_duration_ms = frame_duration_ms
        self.frame_size = max(1, int(sample_rate * frame_duration_ms / 1000)) * 2
        self.preroll_frames = max(0, int(round(preroll_ms / frame_duration_ms)))
        self.fade_in_samples = max(0, int(sample_rate * fade_in_ms / 1000))
        self.fade_in_cursor = 0
        self.stream_id = f"turn-{turn_id}-{int(time.time() * 1000)}"
        self.started_at = time.monotonic()
        self.started = False
        self.first_audio_ms: int | None = None
        self.frame_count = 0
        self.byte_count = 0
        self.buffer = bytearray()

    async def start(self) -> None:
        if not self.started:
            self.started = True
            await self.outbound.put(
                AssistantAudioStart(
                    session_id=self.session_id,
                    stream_id=self.stream_id,
                    sample_rate=self.sample_rate,
                    channels=1,
                    sample_format="s16le",
                    frame_size_bytes=self.frame_size,
                    frame_duration_ms=self.frame_duration_ms,
                ).to_json_bytes().decode()
            )
            for _ in range(self.preroll_frames):
                await self._send_frame(b"\x00" * self.frame_size)

    async def write(self, chunk: bytes) -> int | None:
        if not chunk:
            return None
        first_audio_event: int | None = None
        if not self.started:
            await self.start()
        if self.first_audio_ms is None:
            self.first_audio_ms = int((time.monotonic() - self.started_at) * 1000)
            first_audio_event = self.first_audio_ms
        chunk = self._apply_fade_in(chunk)
        self.buffer.extend(chunk)
        self.byte_count += len(chunk)
        while len(self.buffer) >= self.frame_size:
            payload = bytes(self.buffer[: self.frame_size])
            del self.buffer[: self.frame_size]
            await self._send_frame(payload)
        return first_audio_event

    async def close(self) -> TtsMetrics:
        if not self.started:
            await self.write(b"\x00" * self.frame_size)
            self.byte_count = 0
        if self.buffer:
            payload = bytes(self.buffer)
            payload += b"\x00" * (self.frame_size - len(payload))
            self.buffer.clear()
            await self._send_frame(payload)
        await self.outbound.put(
            AssistantAudioStop(
                session_id=self.session_id,
                stream_id=self.stream_id,
                frame_count=self.frame_count,
            ).to_json_bytes().decode()
        )
        return TtsMetrics(
            byte_count=self.byte_count,
            sample_rate=self.sample_rate,
            duration_ms=int(self.byte_count / (self.sample_rate * 2) * 1000),
            frame_count=self.frame_count,
            first_audio_ms=self.first_audio_ms,
            mode="stream",
            rtf=None,
        )

    async def _send_frame(self, payload: bytes) -> None:
        header = FrameHeader(
            sequence=self.frame_count,
            timestamp_us=_now_us(),
            payload_length=len(payload),
        )
        await self.outbound.put(header.encode(payload))
        self.frame_count += 1

    def _apply_fade_in(self, chunk: bytes) -> bytes:
        if self.fade_in_samples <= 0 or self.fade_in_cursor >= self.fade_in_samples:
            return chunk
        if len(chunk) < 2:
            return chunk

        sample_count = len(chunk) // 2
        samples = array.array("h")
        samples.frombytes(chunk[: sample_count * 2])
        if samples.itemsize != 2:
            return chunk

        fade_remaining = self.fade_in_samples - self.fade_in_cursor
        fade_count = min(sample_count, fade_remaining)
        for index in range(fade_count):
            gain = (self.fade_in_cursor + index) / self.fade_in_samples
            samples[index] = int(samples[index] * gain)
        self.fade_in_cursor += fade_count

        faded = samples.tobytes()
        if len(chunk) > len(faded):
            faded += chunk[len(faded) :]
        return faded


def _clean_assistant_text(text: str) -> str:
    text = text.replace("<|im_end|>", "").replace("<|im_start|>", "")
    text = text.replace("<think></think>", "")
    text = text.replace("\u2018", "'").replace("\u2019", "'")
    text = text.replace("\u201c", '"').replace("\u201d", '"')
    text = text.replace("\u2010", "-").replace("\u2011", "-").replace("\u2013", "-").replace("\u2014", "-")
    text = text.replace("\u202f", " ").replace("\u00a0", " ")
    text = text.strip()
    if text.lower().startswith("assistant\n"):
        text = text[len("assistant\n") :].strip()
    return text


def _parse_llm_stream_line(line: str) -> tuple[str, bool]:
    line = line.strip()
    if not line:
        return "", False
    if line.startswith("data:"):
        line = line.removeprefix("data:").strip()
    if not line:
        return "", False
    if line == "[DONE]":
        return "", True
    data = json.loads(line)
    return str(data.get("content") or ""), bool(data.get("stop"))


class StreamingTextSegmenter:
    def __init__(self, max_chars: int) -> None:
        self.max_chars = max_chars
        self.min_chars = 45
        self.hard_chars = max(max_chars + 60, 160)
        self.buffer = ""

    def push(self, text: str) -> list[str]:
        self.buffer = _normalize_stream_text(f"{self.buffer}{text}")
        return self._pop_ready(force=False)

    def flush(self) -> list[str]:
        self.buffer = _normalize_stream_text(self.buffer)
        if not self.buffer:
            return []
        segments = _split_tts_segments(self.buffer, max_chars=self.max_chars)
        self.buffer = ""
        return segments

    def _pop_ready(self, *, force: bool) -> list[str]:
        if force:
            return self.flush()

        segments: list[str] = []
        while self.buffer:
            split_at = self._ready_split_index(self.buffer)
            if split_at is None:
                break
            segment = self.buffer[:split_at].strip()
            self.buffer = self.buffer[split_at:].lstrip()
            if segment:
                segments.append(segment)
        return segments

    def _ready_split_index(self, text: str) -> int | None:
        for match in re.finditer(r'[.!?]["\')\]]?(?:\s+|$)', text):
            if match.end() >= self.min_chars:
                return match.end()

        if len(text) < self.max_chars:
            return None

        window = text[: self.hard_chars]
        clause_index = _last_boundary_index(window, r'[,;:]["\')\]]?(?:\s+|$)', min_index=self.min_chars)
        if clause_index is not None:
            return clause_index

        sentence_index = _last_boundary_index(window, r'[.!?]["\')\]]?(?:\s+|$)', min_index=20)
        if sentence_index is not None:
            return sentence_index

        word_index = text.rfind(" ", self.min_chars, min(len(text), self.max_chars))
        if word_index > 0:
            return word_index + 1
        if len(text) >= self.hard_chars:
            return self.hard_chars
        return None


def _normalize_stream_text(text: str) -> str:
    text = text.replace("\r", " ").replace("\n", " ")
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"\s+([,.;:!?])", r"\1", text)
    return text.lstrip()


def _last_boundary_index(text: str, pattern: str, *, min_index: int) -> int | None:
    boundary: int | None = None
    for match in re.finditer(pattern, text):
        if match.end() >= min_index:
            boundary = match.end()
    return boundary


def _split_tts_segments(text: str, max_chars: int = 120) -> list[str]:
    min_chars = 45
    hard_chars = max(max_chars + 60, 160)
    text = " ".join(text.split())
    if len(text) <= max_chars:
        return [text]

    pieces = [piece.strip() for piece in re.split(r"(?<=[.!?])\s+", text) if piece.strip()]
    segments: list[str] = []
    for piece in pieces:
        if len(piece) <= max_chars:
            segments.append(piece)
        else:
            segments.extend(_split_long_tts_sentence(piece, max_chars=max_chars, min_chars=min_chars))

    return _merge_short_tts_segments(segments, max_chars=max_chars, hard_chars=hard_chars, min_chars=min_chars) or [text]


def _split_long_tts_sentence(text: str, *, max_chars: int, min_chars: int) -> list[str]:
    clauses = [part.strip() for part in re.split(r"(?<=[,;:])\s+", text) if part.strip()]
    if len(clauses) == 1:
        return _split_tts_words(text, max_chars=max_chars)

    segments: list[str] = []
    current = ""
    for clause in clauses:
        if not current:
            current = clause
            continue
        candidate = f"{current} {clause}"
        if len(candidate) <= max_chars or len(current) < min_chars:
            current = candidate
        else:
            segments.append(current)
            current = clause
    if current:
        segments.append(current)

    final: list[str] = []
    for segment in segments:
        if len(segment) <= max_chars + 60:
            final.append(segment)
        else:
            final.extend(_split_tts_words(segment, max_chars=max_chars))
    return final


def _split_tts_words(text: str, *, max_chars: int) -> list[str]:
    words = text.split()
    segments: list[str] = []
    current = ""
    for word in words:
        candidate = word if not current else f"{current} {word}"
        if len(candidate) <= max_chars:
            current = candidate
        else:
            if current:
                segments.append(current)
            current = word
    if current:
        segments.append(current)
    return segments


def _merge_short_tts_segments(
    segments: list[str],
    *,
    max_chars: int,
    hard_chars: int,
    min_chars: int,
) -> list[str]:
    merged: list[str] = []
    index = 0
    while index < len(segments):
        current = segments[index]
        while index + 1 < len(segments) and len(current) < min_chars:
            candidate = f"{current} {segments[index + 1]}"
            if len(candidate) > hard_chars:
                break
            current = candidate
            index += 1

        if merged and len(current) < min_chars:
            candidate = f"{merged[-1]} {current}"
            if len(candidate) <= hard_chars:
                merged[-1] = candidate
            else:
                merged.append(current)
        else:
            merged.append(current)
        index += 1
    return merged
