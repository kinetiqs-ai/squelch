"""Diagnostics writers for Thor native audio ingress sessions."""

from __future__ import annotations

import json
import wave
from datetime import datetime
from pathlib import Path
from typing import Any


class JsonlWriter:
    def __init__(self, path: Path) -> None:
        self.path = path
        self._file = path.open("a", encoding="utf-8")

    def write(self, event: dict[str, Any]) -> None:
        self._file.write(json.dumps(event, sort_keys=True) + "\n")
        self._file.flush()

    def close(self) -> None:
        self._file.close()


class SessionDiagnostics:
    def __init__(
        self,
        session_id: str,
        root: Path = Path("runs/audio-ingress"),
        sample_rate_hz: int = 16000,
    ) -> None:
        safe_session = "".join(
            char if char.isalnum() or char in {"-", "_"} else "_" for char in session_id
        )
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        self.run_dir = root / f"{stamp}-{safe_session[:12]}"
        self.run_dir.mkdir(parents=True, exist_ok=True)

        self._wav_file = (self.run_dir / "received.wav").open("wb")
        self._wav = wave.open(self._wav_file, "wb")
        self._wav.setnchannels(1)
        self._wav.setsampwidth(2)
        self._wav.setframerate(sample_rate_hz)

        self.transport = JsonlWriter(self.run_dir / "transport.jsonl")
        self.asr = JsonlWriter(self.run_dir / "asr_events.jsonl")
        self._summary_path = self.run_dir / "summary.json"
        self._closed = False

    def write_audio(self, payload: bytes) -> None:
        self._wav.writeframes(payload)

    def write_summary(self, summary: dict[str, Any]) -> None:
        self._summary_path.write_text(
            json.dumps(summary, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._wav.close()
        self._wav_file.close()
        self.transport.close()
        self.asr.close()
