"""Audio diagnostics for ASR pipeline debugging."""

import asyncio
import json
import time
import wave
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from loguru import logger

from pipecat.frames.frames import (
    AudioRawFrame,
    Frame,
    UserStartedSpeakingFrame,
    UserStoppedSpeakingFrame,
    VADUserStartedSpeakingFrame,
    VADUserStoppedSpeakingFrame,
)
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor


class ASRAudioDiagnosticsProcessor(FrameProcessor):
    """Save pre-STT audio segments for ASR debugging.

    Place this processor before STT. It records the audio that actually reaches
    the downstream pipeline around VAD speech boundaries, so we can determine
    whether missing words are lost before or inside the ASR adapter.
    """

    def __init__(
        self,
        *,
        output_dir: str = "diagnostics/asr",
        prebuffer_secs: float = 4.0,
        postbuffer_secs: float = 0.5,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self._output_dir = Path(output_dir)
        self._prebuffer_secs = prebuffer_secs
        self._postbuffer_secs = postbuffer_secs

        self._prebuffer = bytearray()
        self._utterance = bytearray()
        self._session_audio = bytearray()
        self._sample_rate = 16000
        self._num_channels = 1
        self._turn_id = 0
        self._recording = False
        self._pending_stop_at: Optional[float] = None
        self._turn_started_at: Optional[float] = None
        self._vad_started_at: Optional[float] = None
        self._vad_stopped_at: Optional[float] = None
        self._session_started_at: Optional[float] = None

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)

        if isinstance(frame, AudioRawFrame):
            await self._handle_audio(frame)
        elif isinstance(frame, VADUserStartedSpeakingFrame):
            self._handle_vad_started()
        elif isinstance(frame, UserStartedSpeakingFrame):
            self._handle_user_started()
        elif isinstance(frame, (VADUserStoppedSpeakingFrame, UserStoppedSpeakingFrame)):
            self._handle_stop_frame(frame)

        await self.push_frame(frame, direction)

    async def _handle_audio(self, frame: AudioRawFrame):
        self._sample_rate = frame.sample_rate
        self._num_channels = frame.num_channels
        if self._session_started_at is None:
            self._session_started_at = time.time()
        self._session_audio += frame.audio

        if self._recording:
            self._utterance += frame.audio
            if self._pending_stop_at is not None:
                elapsed = time.time() - self._pending_stop_at
                if elapsed >= self._postbuffer_secs:
                    await self._save_turn(reason="postbuffer_elapsed")
            return

        self._prebuffer += frame.audio
        max_bytes = int(self._sample_rate * self._num_channels * 2 * self._prebuffer_secs)
        if len(self._prebuffer) > max_bytes:
            self._prebuffer = self._prebuffer[-max_bytes:]

    def _handle_vad_started(self):
        self._vad_started_at = time.time()
        self._start_turn("vad_started")

    def _handle_user_started(self):
        if self._turn_started_at is None:
            self._turn_started_at = time.time()
        if not self._recording:
            self._start_turn("user_started")

    def _start_turn(self, trigger: str):
        if self._recording:
            return
        self._turn_id += 1
        self._recording = True
        self._pending_stop_at = None
        self._utterance = bytearray(self._prebuffer)
        pre_ms = self._duration_ms(len(self._prebuffer))
        logger.info(
            f"ASR diagnostics turn={self._turn_id} started trigger={trigger} "
            f"prebuffer={pre_ms}ms"
        )

    def _handle_stop_frame(self, frame: Frame):
        now = time.time()
        if isinstance(frame, VADUserStoppedSpeakingFrame):
            self._vad_stopped_at = now
        if self._recording and self._pending_stop_at is None:
            self._pending_stop_at = now

    async def _save_turn(self, *, reason: str):
        if not self._recording:
            return

        audio = bytes(self._utterance)
        turn_id = self._turn_id
        sample_rate = self._sample_rate
        num_channels = self._num_channels
        duration_ms = self._duration_ms(len(audio))
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
        base = self._output_dir / f"turn_{turn_id:04d}_{timestamp}"
        wav_path = base.with_suffix(".wav")
        json_path = base.with_suffix(".json")
        session_wav_path = self._output_dir / "session_latest.wav"
        session_json_path = self._output_dir / "session_latest.json"

        metadata = {
            "turn_id": turn_id,
            "reason": reason,
            "sample_rate": sample_rate,
            "num_channels": num_channels,
            "bytes": len(audio),
            "duration_ms": duration_ms,
            "prebuffer_secs": self._prebuffer_secs,
            "postbuffer_secs": self._postbuffer_secs,
            "turn_started_at": self._turn_started_at,
            "vad_started_at": self._vad_started_at,
            "vad_stopped_at": self._vad_stopped_at,
            "wav_path": str(wav_path),
            "session_wav_path": str(session_wav_path),
        }
        session_metadata = {
            "sample_rate": sample_rate,
            "num_channels": num_channels,
            "bytes": len(self._session_audio),
            "duration_ms": self._duration_ms(len(self._session_audio)),
            "session_started_at": self._session_started_at,
            "last_turn_id": turn_id,
            "wav_path": str(session_wav_path),
        }

        self._recording = False
        self._pending_stop_at = None
        self._utterance = bytearray()
        self._prebuffer = bytearray()
        self._turn_started_at = None
        self._vad_started_at = None
        self._vad_stopped_at = None

        session_audio = bytes(self._session_audio)
        await asyncio.to_thread(self._write_files, wav_path, json_path, audio, metadata)
        await asyncio.to_thread(
            self._write_files,
            session_wav_path,
            session_json_path,
            session_audio,
            session_metadata,
        )
        logger.info(
            f"ASR diagnostics saved turn={turn_id} duration={duration_ms}ms "
            f"wav={wav_path}"
        )

    def _duration_ms(self, byte_count: int) -> int:
        if self._sample_rate <= 0 or self._num_channels <= 0:
            return 0
        samples = byte_count // (2 * self._num_channels)
        return int((samples * 1000) / self._sample_rate)

    def _write_files(self, wav_path: Path, json_path: Path, audio: bytes, metadata: dict):
        wav_path.parent.mkdir(parents=True, exist_ok=True)
        with wave.open(str(wav_path), "wb") as wf:
            wf.setnchannels(metadata["num_channels"])
            wf.setsampwidth(2)
            wf.setframerate(metadata["sample_rate"])
            wf.writeframes(audio)
        json_path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")
