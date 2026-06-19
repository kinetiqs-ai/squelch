"""Native audio-edge-agent ingress app for the Squelch Thor voice stack."""

from __future__ import annotations

import argparse

import uvicorn
from fastapi import FastAPI
from fastapi.responses import JSONResponse

from native_voice.audio_ingress import router as audio_ingress_router


app = FastAPI(title="Squelch Native Voice Ingress")
app.include_router(audio_ingress_router)


@app.get("/health")
async def health() -> JSONResponse:
    return JSONResponse(
        {
            "ok": True,
            "service": "native-voice-ingress",
            "audio_ingress_ws": "/ws/audio-ingress",
        }
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Native audio-edge-agent ingress")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=7860)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
