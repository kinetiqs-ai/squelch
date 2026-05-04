#!/usr/bin/env python3
"""Compare Nemotron and Voxtral ASR on the same WAV/PCM input."""

import argparse
import asyncio
import base64
import json
import time
import wave
from pathlib import Path

import numpy as np
import websockets


def load_audio(path: Path, target_sample_rate: int = 16000) -> bytes:
    try:
        with wave.open(str(path), "rb") as wf:
            channels = wf.getnchannels()
            sample_width = wf.getsampwidth()
            sample_rate = wf.getframerate()
            raw = wf.readframes(wf.getnframes())
    except wave.Error:
        raw = path.read_bytes()
        channels = 1
        sample_width = 2
        sample_rate = 24000

    if sample_width != 2:
        raise ValueError(f"Unsupported sample width: {sample_width}")

    samples = np.frombuffer(raw, dtype=np.int16)
    if channels == 2:
        samples = samples.reshape(-1, 2).mean(axis=1).astype(np.int16)
    elif channels != 1:
        raise ValueError(f"Unsupported channel count: {channels}")

    if sample_rate != target_sample_rate:
        ratio = target_sample_rate / sample_rate
        new_length = int(len(samples) * ratio)
        indices = np.linspace(0, len(samples) - 1, new_length)
        samples = np.interp(indices, np.arange(len(samples)), samples).astype(np.int16)

    return samples.tobytes()


async def transcribe_nemotron(url: str, pcm: bytes, chunk_ms: int = 160) -> tuple[str, float]:
    samples_per_chunk = int(16000 * chunk_ms / 1000)
    bytes_per_chunk = samples_per_chunk * 2
    started = time.time()
    final_text = ""

    async with websockets.connect(url) as ws:
        try:
            message = await asyncio.wait_for(ws.recv(), timeout=5)
            data = json.loads(message)
            if data.get("type") != "ready":
                print(f"Nemotron initial event: {data}")
        except asyncio.TimeoutError:
            pass

        for offset in range(0, len(pcm), bytes_per_chunk):
            await ws.send(pcm[offset : offset + bytes_per_chunk])
            await asyncio.sleep(chunk_ms / 1000)

        await ws.send(json.dumps({"type": "reset", "finalize": True}))

        while True:
            data = json.loads(await asyncio.wait_for(ws.recv(), timeout=15))
            if data.get("type") == "transcript" and data.get("is_final"):
                final_text = data.get("text", "")
                break

    return final_text, time.time() - started


async def transcribe_voxtral(url: str, pcm: bytes, chunk_ms: int = 160) -> tuple[str, float]:
    samples_per_chunk = int(16000 * chunk_ms / 1000)
    bytes_per_chunk = samples_per_chunk * 2
    started = time.time()
    final_text = ""

    async with websockets.connect(url) as ws:
        try:
            initial = json.loads(await asyncio.wait_for(ws.recv(), timeout=5))
            print(f"Voxtral initial event: {initial.get('type')}")
        except asyncio.TimeoutError:
            pass

        await ws.send(
            json.dumps(
                {
                    "type": "session.update",
                    "model": "mistralai/Voxtral-Mini-4B-Realtime-2602",
                }
            )
        )
        await ws.send(json.dumps({"type": "input_audio_buffer.commit"}))

        for offset in range(0, len(pcm), bytes_per_chunk):
            chunk = pcm[offset : offset + bytes_per_chunk]
            await ws.send(
                json.dumps(
                    {
                        "type": "input_audio_buffer.append",
                        "audio": base64.b64encode(chunk).decode("utf-8"),
                    }
                )
            )
            await asyncio.sleep(chunk_ms / 1000)

        await ws.send(json.dumps({"type": "input_audio_buffer.commit", "final": True}))

        while True:
            data = json.loads(await asyncio.wait_for(ws.recv(), timeout=30))
            msg_type = data.get("type")
            if msg_type in {"transcription.delta", "response.text.delta"}:
                final_text += data.get("delta", "")
            elif msg_type in {"transcription.done", "response.done"}:
                final_text = data.get("text") or final_text
                break
            if msg_type == "error":
                raise RuntimeError(f"Voxtral error: {data}")

    return final_text, time.time() - started


def word_error_rate(reference: str, hypothesis: str) -> float:
    ref = reference.lower().split()
    hyp = hypothesis.lower().split()
    if not ref:
        return 0.0 if not hyp else 1.0

    dp = [[0] * (len(hyp) + 1) for _ in range(len(ref) + 1)]
    for i in range(len(ref) + 1):
        dp[i][0] = i
    for j in range(len(hyp) + 1):
        dp[0][j] = j
    for i in range(1, len(ref) + 1):
        for j in range(1, len(hyp) + 1):
            cost = 0 if ref[i - 1] == hyp[j - 1] else 1
            dp[i][j] = min(
                dp[i - 1][j] + 1,
                dp[i][j - 1] + 1,
                dp[i - 1][j - 1] + cost,
            )
    return dp[-1][-1] / len(ref)


async def main():
    parser = argparse.ArgumentParser(description="Compare Nemotron and Voxtral ASR")
    parser.add_argument("audio", type=Path)
    parser.add_argument("--reference", default="")
    parser.add_argument("--nemotron-url", default="ws://localhost:8080")
    parser.add_argument("--voxtral-url", default="ws://localhost:8082/v1/realtime")
    parser.add_argument("--chunk-ms", type=int, default=160)
    parser.add_argument("--backend", choices=["both", "nemotron", "voxtral"], default="both")
    args = parser.parse_args()

    pcm = load_audio(args.audio)
    duration_s = len(pcm) / 2 / 16000
    print(f"Audio: {duration_s:.2f}s PCM16 16kHz mono")

    if args.backend in {"both", "nemotron"}:
        text, elapsed = await transcribe_nemotron(args.nemotron_url, pcm, args.chunk_ms)
        print("\nNemotron")
        print(f"  latency: {elapsed:.2f}s")
        print(f"  text: {text}")
        if args.reference:
            print(f"  WER: {word_error_rate(args.reference, text):.2%}")

    if args.backend in {"both", "voxtral"}:
        text, elapsed = await transcribe_voxtral(args.voxtral_url, pcm, args.chunk_ms)
        print("\nVoxtral")
        print(f"  latency: {elapsed:.2f}s")
        print(f"  text: {text}")
        if args.reference:
            print(f"  WER: {word_error_rate(args.reference, text):.2%}")


if __name__ == "__main__":
    asyncio.run(main())
