"""Audio-edge-agent protocol receiver for native Thor Riva ASR validation."""

from __future__ import annotations

import asyncio
import json
import logging
import queue
import threading
import time
import warnings
import wave
from collections.abc import Iterable
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import riva.client
with warnings.catch_warnings():
    warnings.filterwarnings("ignore", message="pkg_resources is deprecated.*", category=UserWarning)
    import webrtcvad
from audio_edge_agent.protocol import (
    CHANNELS,
    FRAME_DURATION_MS,
    FRAME_SIZE_BYTES,
    HEADER_SIZE,
    PROTOCOL_VERSION,
    SAMPLE_FORMAT,
    SAMPLE_RATE,
    AsrTranscriptEvent,
    AssistantAudioStart,
    AssistantAudioStop,
    AudioStart,
    AudioStop,
    ErrorCode,
    FrameHeader,
    MessageType,
    ProtocolError,
    ReceiverError,
    ReceiverReady,
    TelemetryMessage,
    _now_us,
)
from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from native_voice.diagnostics import SessionDiagnostics
from native_voice.riva_pipeline import (
    RIVA_CHUNK_BYTES,
    RIVA_CHUNK_FRAMES,
    SAMPLE_RATE_HZ,
    TranscriptAssembler,
    build_streaming_config,
    transcript_messages,
)


router = APIRouter()
logger = logging.getLogger("uvicorn.error")
ROOT_DIR = Path(__file__).resolve().parents[1]

EDGE_FRAMES_PER_RIVA_CHUNK = RIVA_CHUNK_BYTES // FRAME_SIZE_BYTES
if RIVA_CHUNK_BYTES % FRAME_SIZE_BYTES:
    raise RuntimeError("Riva chunk size must be an integer multiple of edge frames")

_SEGMENT_END = object()


@dataclass
class VadDecision:
    voiced: bool
    rms: float
    state: str
    feed_frames: list[bytes]
    transition: str | None = None


class VadGate:
    def __init__(
        self,
        aggressiveness: int = 3,
        preroll_frames: int = 25,
        start_window_frames: int = 8,
        start_voiced_frames: int = 4,
        end_silence_frames: int = 35,
        rms_start_threshold: float = 0.004,
    ) -> None:
        self._vad = webrtcvad.Vad(aggressiveness)
        self._preroll: deque[bytes] = deque(maxlen=preroll_frames)
        self._recent_voiced: deque[bool] = deque(maxlen=start_window_frames)
        self._start_voiced_frames = start_voiced_frames
        self._end_silence_frames = end_silence_frames
        self._rms_start_threshold = rms_start_threshold
        self._active = False
        self._silence_frames = 0
        self.speech_segments = 0
        self.voiced_frames = 0
        self.unvoiced_frames = 0
        self.asr_frames = 0

    @property
    def state(self) -> str:
        return "speech" if self._active else "idle"

    def process(self, payload: bytes) -> VadDecision:
        rms = _pcm_rms(payload)
        voiced = self._vad.is_speech(payload, SAMPLE_RATE) and rms >= self._rms_start_threshold
        if voiced:
            self.voiced_frames += 1
        else:
            self.unvoiced_frames += 1

        feed_frames: list[bytes] = []
        transition: str | None = None

        if not self._active:
            self._preroll.append(payload)
            self._recent_voiced.append(voiced)
            if sum(self._recent_voiced) >= self._start_voiced_frames:
                self._active = True
                self._silence_frames = 0
                self.speech_segments += 1
                transition = "speech_start"
                feed_frames.extend(self._preroll)
                self._preroll.clear()
            self.asr_frames += len(feed_frames)
            return VadDecision(voiced, rms, self.state, feed_frames, transition)

        feed_frames.append(payload)
        if voiced:
            self._silence_frames = 0
        else:
            self._silence_frames += 1
            if self._silence_frames >= self._end_silence_frames:
                self._active = False
                self._silence_frames = 0
                self._recent_voiced.clear()
                self._preroll.clear()
                transition = "speech_end"

        self.asr_frames += len(feed_frames)
        return VadDecision(voiced, rms, self.state, feed_frames, transition)


def _pcm_rms(payload: bytes) -> float:
    samples = memoryview(payload).cast("h")
    if not samples:
        return 0.0
    total = sum(int(sample) * int(sample) for sample in samples)
    return (total / len(samples)) ** 0.5 / 32768.0


def _decode_json_message(text: str) -> dict[str, Any]:
    try:
        payload = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ProtocolError(f"invalid JSON: {exc}") from exc
    if not isinstance(payload, dict):
        raise ProtocolError("JSON message must be an object")
    return payload


def _validate_audio_start(raw: str) -> AudioStart:
    payload = _decode_json_message(raw)
    expected = {
        "protocol_version": PROTOCOL_VERSION,
        "sample_rate": SAMPLE_RATE,
        "channels": CHANNELS,
        "sample_format": SAMPLE_FORMAT,
        "frame_size_bytes": FRAME_SIZE_BYTES,
        "frame_duration_ms": FRAME_DURATION_MS,
    }
    for key, value in expected.items():
        if payload.get(key) != value:
            raise ProtocolError(f"{key}={payload.get(key)!r} does not match expected {value!r}")
    return AudioStart.from_json_bytes(raw.encode())


async def _send_error(
    websocket: WebSocket,
    code: ErrorCode,
    detail: str,
    session_id: str | None = None,
) -> None:
    error = ReceiverError(code=code, detail=detail, session_id=session_id)
    await websocket.send_text(error.to_json_bytes().decode())


def _protocol_error_code(exc: ProtocolError) -> ErrorCode:
    detail = str(exc)
    if "Bad magic" in detail:
        return ErrorCode.BAD_MAGIC
    if "Unsupported protocol version" in detail:
        return ErrorCode.BAD_VERSION
    if "payload" in detail or "too short" in detail:
        return ErrorCode.BAD_PAYLOAD_LENGTH
    return ErrorCode.BAD_MESSAGE_TYPE


class RivaWorker:
    def __init__(
        self,
        session_id: str,
        audio_queue: queue.Queue[bytes | object | None],
        event_loop: asyncio.AbstractEventLoop,
        outbound: asyncio.Queue[str | bytes | None],
        diagnostics: SessionDiagnostics,
        started_at: float,
        leading_silence_ms: int = 0,
    ) -> None:
        self.session_id = session_id
        self.audio_queue = audio_queue
        self.event_loop = event_loop
        self.outbound = outbound
        self.diagnostics = diagnostics
        self.started_at = started_at
        self.leading_silence_ms = leading_silence_ms
        self.stop_event = threading.Event()
        self.assembler = TranscriptAssembler()
        self.thread = threading.Thread(target=self._run, name="riva-audio-ingress-worker", daemon=True)

    def start(self) -> None:
        self.thread.start()

    def stop(self) -> None:
        self.finish(drain=False)

    def finish(self, drain: bool) -> None:
        if not drain:
            self.stop_event.set()
        try:
            self.audio_queue.put(None, timeout=2.0)
        except queue.Full:
            logger.warning("audio_ingress_queue_full_while_stopping")

    def join(self, timeout: float = 10.0) -> None:
        self.thread.join(timeout=timeout)

    def abort(self) -> None:
        self.stop_event.set()
        try:
            self.audio_queue.put_nowait(None)
        except queue.Full:
            pass

    def _audio_chunks(self, first_payload: bytes) -> Iterable[bytes]:
        if self.leading_silence_ms > 0:
            silence_chunks = max(
                1,
                round((self.leading_silence_ms / 1000) * SAMPLE_RATE_HZ / RIVA_CHUNK_FRAMES),
            )
            silence = b"\x00\x00" * RIVA_CHUNK_FRAMES
            for _ in range(silence_chunks):
                yield silence

        pending: list[bytes] = [first_payload]
        while not self.stop_event.is_set():
            while len(pending) < EDGE_FRAMES_PER_RIVA_CHUNK:
                payload = self.audio_queue.get()
                if payload is None:
                    self.stop_event.set()
                    break
                if payload is _SEGMENT_END:
                    break
                pending.append(payload)

            if pending:
                yield b"".join(pending)
                pending.clear()

            if self.stop_event.is_set() or payload is _SEGMENT_END:
                break

    def _run(self) -> None:
        auth = riva.client.Auth(uri="localhost:50051")
        asr_service = riva.client.ASRService(auth)
        while not self.stop_event.is_set():
            first_payload = self.audio_queue.get()
            if first_payload is None:
                break
            if first_payload is _SEGMENT_END:
                continue
            try:
                responses = asr_service.streaming_response_generator(
                    audio_chunks=self._audio_chunks(first_payload),
                    streaming_config=build_streaming_config(),
                )
                for response in responses:
                    for message in transcript_messages(response, self.started_at, self.assembler):
                        self.diagnostics.asr.write(message)
                        self._emit_transcript_event(message)
                        logger.info(
                            "audio_ingress_asr seq=%s final=%s elapsed_ms=%s text=%r committed=%r",
                            message["sequence"],
                            message["final"],
                            message["elapsed_ms"],
                            message["text"],
                            message["committed_text"],
                        )
            except Exception as exc:  # noqa: BLE001 - diagnostics should capture service failures.
                self.diagnostics.asr.write(
                    {
                        "type": "error",
                        "message": str(exc),
                        "elapsed_ms": int((time.monotonic() - self.started_at) * 1000),
                    }
                )
                logger.exception("audio_ingress_riva_worker_failed")

    def _emit_transcript_event(self, message: dict[str, Any]) -> None:
        event = AsrTranscriptEvent(
            session_id=self.session_id,
            text=message["text"],
            final=bool(message["final"]),
            sequence=int(message["sequence"]),
            elapsed_ms=message.get("elapsed_ms"),
            confidence=message.get("confidence"),
            stability=message.get("stability"),
        )
        payload = event.to_json_bytes().decode()

        def enqueue() -> None:
            try:
                self.outbound.put_nowait(payload)
            except asyncio.QueueFull:
                logger.warning("audio_ingress_outbound_event_queue_full session=%s", self.session_id)

        self.event_loop.call_soon_threadsafe(enqueue)


@router.websocket("/ws/audio-ingress")
async def websocket_audio_ingress(websocket: WebSocket) -> None:
    await websocket.accept()
    loop = asyncio.get_running_loop()

    first = await websocket.receive()
    first_text = first.get("text")
    if first_text is None:
        await _send_error(websocket, ErrorCode.MISSING_AUDIO_START, "first message must be audio_start")
        await websocket.close()
        return

    try:
        audio_start = _validate_audio_start(first_text)
    except ProtocolError as exc:
        await _send_error(websocket, ErrorCode.MISSING_AUDIO_START, str(exc))
        await websocket.close()
        return

    session_id = audio_start.session_id
    diagnostics = SessionDiagnostics(session_id=session_id)
    started_at = time.monotonic()
    audio_queue: queue.Queue[bytes | object | None] = queue.Queue(maxsize=500)
    outbound: asyncio.Queue[str | bytes | None] = asyncio.Queue(maxsize=500)
    vad_gate = VadGate()
    worker = RivaWorker(
        session_id=session_id,
        audio_queue=audio_queue,
        event_loop=loop,
        outbound=outbound,
        diagnostics=diagnostics,
        started_at=started_at,
        leading_silence_ms=0,
    )
    worker.start()

    async def sender() -> None:
        while True:
            payload = await outbound.get()
            if payload is None:
                return
            if isinstance(payload, bytes):
                await websocket.send_bytes(payload)
            else:
                await websocket.send_text(payload)

    sender_task = asyncio.create_task(sender())
    return_test = websocket.query_params.get("return_test")
    return_test_task: asyncio.Task[None] | None = None
    if return_test == "harvard":
        return_test_task = asyncio.create_task(_send_harvard_return_test(outbound, session_id))

    frame_count = 0
    sequence_gaps = 0
    queue_drops = 0
    last_sequence: int | None = None
    last_receive_us: int | None = None
    closed_reason = "disconnect"

    diagnostics.transport.write(
        {
            "type": "audio_start",
            "session_id": session_id,
            "device_name": audio_start.device_name,
            "device_uid": audio_start.device_uid,
            "capture_rate_hz": audio_start.capture_rate_hz,
            "timestamp_us": audio_start.timestamp_us,
            "receiver_timestamp_us": _now_us(),
            "diagnostics_dir": str(diagnostics.run_dir),
        }
    )

    await websocket.send_text(ReceiverReady(session_id=session_id).to_json_bytes().decode())
    logger.info("audio_ingress_ready session=%s diagnostics=%s", session_id, diagnostics.run_dir)

    try:
        while True:
            message = await websocket.receive()

            data = message.get("bytes")
            if data is not None:
                receive_us = _now_us()
                try:
                    header, payload = FrameHeader.decode(data)
                except ProtocolError as exc:
                    await _send_error(websocket, _protocol_error_code(exc), str(exc), session_id)
                    diagnostics.transport.write(
                        {
                            "type": "frame_error",
                            "detail": str(exc),
                            "receiver_timestamp_us": receive_us,
                        }
                    )
                    continue

                if header.payload_length != FRAME_SIZE_BYTES or len(payload) != FRAME_SIZE_BYTES:
                    detail = (
                        f"payload length {len(payload)} with declared "
                        f"{header.payload_length}; expected {FRAME_SIZE_BYTES}"
                    )
                    await _send_error(websocket, ErrorCode.BAD_PAYLOAD_LENGTH, detail, session_id)
                    diagnostics.transport.write(
                        {
                            "type": "frame_error",
                            "detail": detail,
                            "sequence": header.sequence,
                            "receiver_timestamp_us": receive_us,
                        }
                    )
                    continue
                expected_total = FRAME_SIZE_BYTES + HEADER_SIZE
                if len(data) != expected_total:
                    detail = f"binary frame length {len(data)} does not match expected {expected_total}"
                    await _send_error(websocket, ErrorCode.BAD_PAYLOAD_LENGTH, detail, session_id)
                    diagnostics.transport.write(
                        {
                            "type": "frame_error",
                            "detail": detail,
                            "sequence": header.sequence,
                            "receiver_timestamp_us": receive_us,
                        }
                    )
                    continue

                gap = 0
                if last_sequence is not None:
                    expected = (last_sequence + 1) & 0xFFFFFFFF
                    gap = (header.sequence - expected) & 0xFFFFFFFF
                    if gap:
                        sequence_gaps += 1
                interarrival_ms = (
                    None
                    if last_receive_us is None
                    else round((receive_us - last_receive_us) / 1000.0, 3)
                )
                last_sequence = header.sequence
                last_receive_us = receive_us
                frame_count += 1
                vad = vad_gate.process(payload)

                diagnostics.write_audio(payload)
                diagnostics.transport.write(
                    {
                        "type": "frame",
                        "frame_count": frame_count,
                        "sequence": header.sequence,
                        "timestamp_us": header.timestamp_us,
                        "receiver_timestamp_us": receive_us,
                        "interarrival_ms": interarrival_ms,
                        "payload_length": len(payload),
                        "sequence_gap": gap,
                        "audio_queue_size": audio_queue.qsize(),
                        "vad_voiced": vad.voiced,
                        "vad_rms": round(vad.rms, 6),
                        "vad_state": vad.state,
                        "vad_transition": vad.transition,
                        "asr_feed_frames": len(vad.feed_frames),
                    }
                )
                if vad.transition is not None:
                    diagnostics.transport.write(
                        {
                            "type": "vad_transition",
                            "transition": vad.transition,
                            "frame_count": frame_count,
                            "sequence": header.sequence,
                            "receiver_timestamp_us": receive_us,
                            "rms": round(vad.rms, 6),
                        }
                    )
                for asr_payload in vad.feed_frames:
                    try:
                        audio_queue.put_nowait(asr_payload)
                    except queue.Full:
                        queue_drops += 1
                        diagnostics.transport.write(
                            {
                                "type": "queue_drop",
                                "frame_count": frame_count,
                                "sequence": header.sequence,
                                "receiver_timestamp_us": receive_us,
                            }
                        )
                if vad.transition == "speech_end":
                    try:
                        audio_queue.put_nowait(_SEGMENT_END)
                    except queue.Full:
                        queue_drops += 1
                        diagnostics.transport.write(
                            {
                                "type": "queue_drop",
                                "frame_count": frame_count,
                                "sequence": header.sequence,
                                "receiver_timestamp_us": receive_us,
                                "reason": "segment_end",
                            }
                        )
                continue

            text = message.get("text")
            if text is None:
                continue

            try:
                payload = _decode_json_message(text)
                message_type = payload.get("type")
                if message_type == MessageType.TELEMETRY.value:
                    telemetry = TelemetryMessage.from_json_bytes(text.encode())
                    if telemetry.session_id != session_id:
                        await _send_error(
                            websocket,
                            ErrorCode.SESSION_MISMATCH,
                            f"telemetry session {telemetry.session_id!r} != {session_id!r}",
                            session_id,
                        )
                        continue
                    diagnostics.transport.write(
                        {
                            "type": "telemetry",
                            "frame_count": telemetry.frame_count,
                            "rms": telemetry.rms,
                            "peak": telemetry.peak,
                            "clipping_count": telemetry.clipping_count,
                            "timestamp_us": telemetry.timestamp_us,
                            "receiver_timestamp_us": _now_us(),
                        }
                    )
                    continue

                if message_type == MessageType.AUDIO_STOP.value:
                    stop = AudioStop.from_json_bytes(text.encode())
                    if stop.session_id != session_id:
                        await _send_error(
                            websocket,
                            ErrorCode.SESSION_MISMATCH,
                            f"stop session {stop.session_id!r} != {session_id!r}",
                            session_id,
                        )
                        continue
                    diagnostics.transport.write(
                        {
                            "type": "audio_stop",
                            "frame_count": stop.frame_count,
                            "local_frame_count": frame_count,
                            "timestamp_us": stop.timestamp_us,
                            "receiver_timestamp_us": _now_us(),
                        }
                    )
                    closed_reason = "audio_stop"
                    break

                await _send_error(
                    websocket,
                    ErrorCode.BAD_MESSAGE_TYPE,
                    f"unsupported message type {message_type!r}",
                    session_id,
                )
            except ProtocolError as exc:
                await _send_error(websocket, ErrorCode.BAD_MESSAGE_TYPE, str(exc), session_id)
    except WebSocketDisconnect:
        closed_reason = "disconnect"
    except RuntimeError as exc:
        if "disconnect message" not in str(exc):
            raise
        closed_reason = "disconnect"
    finally:
        if return_test_task is not None:
            return_test_task.cancel()
            try:
                await return_test_task
            except asyncio.CancelledError:
                pass
        worker.finish(drain=closed_reason == "audio_stop")
        await asyncio.to_thread(worker.join, 10.0)
        try:
            outbound.put_nowait(None)
        except asyncio.QueueFull:
            pass
        try:
            await asyncio.wait_for(sender_task, timeout=3.0)
        except asyncio.TimeoutError:
            sender_task.cancel()
        except Exception:
            sender_task.cancel()
        elapsed_ms = int((time.monotonic() - started_at) * 1000)
        diagnostics.write_summary(
            {
                "session_id": session_id,
                "closed_reason": closed_reason,
                "elapsed_ms": elapsed_ms,
                "frames_received": frame_count,
                "sequence_gap_events": sequence_gaps,
                "queue_drops": queue_drops,
                "vad_speech_segments": vad_gate.speech_segments,
                "vad_voiced_frames": vad_gate.voiced_frames,
                "vad_unvoiced_frames": vad_gate.unvoiced_frames,
                "asr_frames_fed": vad_gate.asr_frames,
                "asr_committed_text": worker.assembler.committed_text,
                "asr_event_count": worker.assembler.event_id,
            }
        )
        diagnostics.close()
        logger.info(
            "audio_ingress_closed session=%s reason=%s frames=%d gaps=%d drops=%d elapsed_ms=%d diagnostics=%s",
            session_id,
            closed_reason,
            frame_count,
            sequence_gaps,
            queue_drops,
            elapsed_ms,
            diagnostics.run_dir,
        )


async def _send_harvard_return_test(
    outbound: asyncio.Queue[str | bytes | None],
    session_id: str,
) -> None:
    await asyncio.sleep(1.0)
    wav_path = ROOT_DIR / "tests" / "fixtures" / "harvard_16k.wav"
    stream_id = f"return-test-{int(time.time() * 1000)}"
    await outbound.put(AssistantAudioStart(session_id=session_id, stream_id=stream_id).to_json_bytes().decode())

    frame_count = 0
    with wave.open(str(wav_path), "rb") as wav:
        if wav.getframerate() != SAMPLE_RATE or wav.getnchannels() != CHANNELS or wav.getsampwidth() != 2:
            raise RuntimeError(f"Unsupported return-test fixture format: {wav_path}")
        while True:
            payload = wav.readframes(FRAME_SIZE_BYTES // 2)
            if not payload:
                break
            if len(payload) < FRAME_SIZE_BYTES:
                payload += b"\x00" * (FRAME_SIZE_BYTES - len(payload))
            header = FrameHeader(sequence=frame_count, timestamp_us=_now_us())
            await outbound.put(header.encode(payload))
            frame_count += 1
            await asyncio.sleep(FRAME_DURATION_MS / 1000)

    await outbound.put(
        AssistantAudioStop(
            session_id=session_id,
            stream_id=stream_id,
            frame_count=frame_count,
        ).to_json_bytes().decode()
    )
