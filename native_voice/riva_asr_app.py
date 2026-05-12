"""Minimal browser-to-Riva streaming ASR harness for Thor validation."""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import queue
import threading
import time
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import riva.client
import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, JSONResponse

from native_voice.audio_ingress import router as audio_ingress_router
from native_voice.riva_pipeline import (
    LEADING_SILENCE_MS,
    RIVA_CHUNK_FRAMES,
    SAMPLE_RATE_HZ,
    TranscriptAssembler,
    build_streaming_config,
    transcript_messages,
)


APP_DIR = Path(__file__).resolve().parent
INDEX_HTML = APP_DIR / "riva_asr_client.html"

app = FastAPI(title="Squelch Native Riva ASR")
app.include_router(audio_ingress_router)
logger = logging.getLogger("uvicorn.error")


@app.get("/")
async def index() -> HTMLResponse:
    return HTMLResponse(INDEX_HTML.read_text(encoding="utf-8"))


@app.get("/health")
async def health() -> JSONResponse:
    return JSONResponse(
        {
            "ok": True,
            "service": "native-riva-asr",
            "browser_ws": "/ws/asr",
            "audio_ingress_ws": "/ws/audio-ingress",
        }
    )


@app.websocket("/ws/asr")
async def websocket_asr(websocket: WebSocket) -> None:
    await websocket.accept()
    loop = asyncio.get_running_loop()
    outbound: asyncio.Queue[dict[str, Any] | None] = asyncio.Queue()
    audio_queue: queue.Queue[bytes | None] = queue.Queue(maxsize=400)
    stop_event = threading.Event()
    started_at = time.monotonic()
    assembler = TranscriptAssembler()

    def audio_chunks() -> Iterable[bytes]:
        silence_chunks = max(
            1,
            round((LEADING_SILENCE_MS / 1000) * SAMPLE_RATE_HZ / RIVA_CHUNK_FRAMES),
        )
        silence = b"\x00\x00" * RIVA_CHUNK_FRAMES
        for _ in range(silence_chunks):
            yield silence

        while not stop_event.is_set():
            chunk = audio_queue.get()
            if chunk is None:
                break
            yield chunk

    def enqueue(message: dict[str, Any] | None) -> None:
        loop.call_soon_threadsafe(outbound.put_nowait, message)

    def riva_worker() -> None:
        try:
            auth = riva.client.Auth(uri="localhost:50051")
            asr_service = riva.client.ASRService(auth)
            responses = asr_service.streaming_response_generator(
                audio_chunks=audio_chunks(),
                streaming_config=build_streaming_config(),
            )
            for response in responses:
                for message in transcript_messages(response, started_at, assembler):
                    logger.info(
                        "riva_asr_event seq=%s final=%s text=%r committed=%r interim=%r",
                        message["sequence"],
                        message["final"],
                        message["text"],
                        message["committed_text"],
                        message["active_interim"],
                    )
                    enqueue(message)
        except Exception as exc:  # noqa: BLE001 - surface service errors to browser.
            enqueue({"type": "error", "message": str(exc)})
        finally:
            enqueue(None)

    async def sender() -> None:
        while True:
            message = await outbound.get()
            if message is None:
                return
            await websocket.send_json(message)

    worker = threading.Thread(target=riva_worker, name="riva-asr-worker", daemon=True)
    worker.start()
    sender_task = asyncio.create_task(sender())
    await websocket.send_json(
        {
            "type": "ready",
            "sample_rate": SAMPLE_RATE_HZ,
            "leading_silence_ms": LEADING_SILENCE_MS,
        }
    )

    try:
        while True:
            message = await websocket.receive()
            if message.get("bytes") is not None:
                try:
                    audio_queue.put_nowait(message["bytes"])
                except queue.Full:
                    await websocket.send_json({"type": "warning", "message": "audio queue full"})
                continue

            text = message.get("text")
            if text is None:
                continue
            try:
                payload = json.loads(text)
            except json.JSONDecodeError:
                payload = {}
            if payload.get("type") in {"stop", "end"}:
                audio_queue.put(None)
                break
    except WebSocketDisconnect:
        pass
    finally:
        stop_event.set()
        try:
            audio_queue.put_nowait(None)
        except queue.Full:
            pass
        await asyncio.wait_for(sender_task, timeout=3.0)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Native browser-to-Riva ASR harness")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=7860)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
