"""Audio-edge-agent protocol receiver for native Thor Riva ASR validation."""

from __future__ import annotations

import asyncio
import json
import logging
import queue
import threading
import time
from collections.abc import Iterable
from typing import Any

import riva.client
from audio_edge_agent.protocol import (
    CHANNELS,
    FRAME_DURATION_MS,
    FRAME_SIZE_BYTES,
    HEADER_SIZE,
    PROTOCOL_VERSION,
    SAMPLE_FORMAT,
    SAMPLE_RATE,
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
    LEADING_SILENCE_MS,
    RIVA_CHUNK_BYTES,
    RIVA_CHUNK_FRAMES,
    SAMPLE_RATE_HZ,
    TranscriptAssembler,
    build_streaming_config,
    transcript_messages,
)


router = APIRouter()
logger = logging.getLogger("uvicorn.error")

EDGE_FRAMES_PER_RIVA_CHUNK = RIVA_CHUNK_BYTES // FRAME_SIZE_BYTES
if RIVA_CHUNK_BYTES % FRAME_SIZE_BYTES:
    raise RuntimeError("Riva chunk size must be an integer multiple of edge frames")


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
        audio_queue: queue.Queue[bytes | None],
        diagnostics: SessionDiagnostics,
        started_at: float,
    ) -> None:
        self.audio_queue = audio_queue
        self.diagnostics = diagnostics
        self.started_at = started_at
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

    def _audio_chunks(self) -> Iterable[bytes]:
        silence_chunks = max(
            1,
            round((LEADING_SILENCE_MS / 1000) * SAMPLE_RATE_HZ / RIVA_CHUNK_FRAMES),
        )
        silence = b"\x00\x00" * RIVA_CHUNK_FRAMES
        for _ in range(silence_chunks):
            yield silence

        pending: list[bytes] = []
        while not self.stop_event.is_set():
            payload = self.audio_queue.get()
            if payload is None:
                break
            pending.append(payload)
            if len(pending) == EDGE_FRAMES_PER_RIVA_CHUNK:
                yield b"".join(pending)
                pending.clear()

        if pending:
            yield b"".join(pending)

    def _run(self) -> None:
        try:
            auth = riva.client.Auth(uri="localhost:50051")
            asr_service = riva.client.ASRService(auth)
            responses = asr_service.streaming_response_generator(
                audio_chunks=self._audio_chunks(),
                streaming_config=build_streaming_config(),
            )
            for response in responses:
                for message in transcript_messages(response, self.started_at, self.assembler):
                    self.diagnostics.asr.write(message)
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


@router.websocket("/ws/audio-ingress")
async def websocket_audio_ingress(websocket: WebSocket) -> None:
    await websocket.accept()

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
    audio_queue: queue.Queue[bytes | None] = queue.Queue(maxsize=500)
    worker = RivaWorker(audio_queue=audio_queue, diagnostics=diagnostics, started_at=started_at)
    worker.start()

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
                    }
                )
                try:
                    audio_queue.put_nowait(payload)
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
    finally:
        worker.finish(drain=closed_reason == "audio_stop")
        await asyncio.to_thread(worker.join, 10.0)
        elapsed_ms = int((time.monotonic() - started_at) * 1000)
        diagnostics.write_summary(
            {
                "session_id": session_id,
                "closed_reason": closed_reason,
                "elapsed_ms": elapsed_ms,
                "frames_received": frame_count,
                "sequence_gap_events": sequence_gaps,
                "queue_drops": queue_drops,
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
