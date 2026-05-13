"""Shared Riva ASR streaming helpers for native Thor audio ingress tests."""

from __future__ import annotations

import time
from typing import Any

import riva.client


SAMPLE_RATE_HZ = 16000
RIVA_CHUNK_FRAMES = 1600
RIVA_CHUNK_BYTES = RIVA_CHUNK_FRAMES * 2
LEADING_SILENCE_MS = 600


class TranscriptAssembler:
    def __init__(self) -> None:
        self.active_interim = ""
        self.event_id = 0
        self.final_segments: list[str] = []

    @property
    def committed_text(self) -> str:
        return " ".join(self.final_segments).strip()

    def observe(self, text: str, final: bool, elapsed_ms: int) -> dict[str, Any]:
        self.event_id += 1
        if not final:
            self.active_interim = text
        else:
            if not self.final_segments or self.final_segments[-1] != text:
                self.final_segments.append(text)
            self.active_interim = ""

        return {
            "type": "transcript",
            "sequence": self.event_id,
            "text": text,
            "final": final,
            "committed_text": self.committed_text,
            "active_interim": self.active_interim,
            "final_segment_count": len(self.final_segments),
            "elapsed_ms": elapsed_ms,
        }


def build_streaming_config() -> riva.client.StreamingRecognitionConfig:
    return riva.client.StreamingRecognitionConfig(
        config=riva.client.RecognitionConfig(
            language_code="en-US",
            max_alternatives=1,
            profanity_filter=False,
            enable_automatic_punctuation=True,
            verbatim_transcripts=True,
            encoding=riva.client.AudioEncoding.LINEAR_PCM,
            sample_rate_hertz=SAMPLE_RATE_HZ,
            audio_channel_count=1,
        ),
        interim_results=True,
    )


def transcript_messages(
    response: Any, started_at: float, assembler: TranscriptAssembler
) -> list[dict[str, Any]]:
    elapsed_ms = int((time.monotonic() - started_at) * 1000)
    messages = []
    for result in response.results:
        if not result.alternatives:
            continue
        alternative = result.alternatives[0]
        text = alternative.transcript.strip()
        if not text:
            continue
        final = bool(result.is_final)
        message = assembler.observe(text, final, elapsed_ms)
        message["stability"] = float(getattr(result, "stability", 0.0))
        message["confidence"] = float(getattr(alternative, "confidence", 0.0))
        messages.append(message)
    return messages
