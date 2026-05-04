#!/usr/bin/env python3
#
# Simplified Pipecat bot for testing ASR accuracy.
#
# Same as bot_interleaved_streaming.py but with simple Silero VAD only (no SmartTurn).
# Uses longer silence threshold (800ms) for more conservative turn detection.
#
# Environment variables:
#   NVIDIA_ASR_URL        ASR WebSocket URL (default: ws://localhost:8080)
#   NVIDIA_LLAMA_CPP_URL  llama.cpp API URL (default: http://localhost:8000)
#   NVIDIA_TTS_URL        Orpheus TTS server URL (default: http://localhost:8001)
#
# Usage:
#   uv run pipecat_bots/bot_simple_vad.py
#   uv run pipecat_bots/bot_simple_vad.py -t webrtc
#

import os

from dotenv import load_dotenv
from loguru import logger

from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.audio.vad.vad_analyzer import VADParams
from pipecat.frames.frames import LLMRunFrame
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineParams, PipelineTask
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.aggregators.llm_response_universal import LLMContextAggregatorPair
from pipecat.processors.frameworks.rtvi import RTVIConfig, RTVIObserver, RTVIProcessor
from pipecat.runner.types import RunnerArguments
from pipecat.runner.utils import create_transport
from pipecat.transports.base_transport import BaseTransport, TransportParams
from pipecat.transports.daily.transport import DailyParams
from pipecat.transports.websocket.fastapi import FastAPIWebsocketParams

# Import our custom local services
from nvidia_stt import NVidiaWebSocketSTTService
from orpheus_http_tts import OrpheusHTTPTTSService
from llama_cpp_chunked_llm import LlamaCppChunkedLLMService

load_dotenv(override=True)

# Configuration from environment
NVIDIA_ASR_URL = os.getenv("NVIDIA_ASR_URL", "ws://localhost:8080")
NVIDIA_LLAMA_CPP_URL = os.getenv("NVIDIA_LLAMA_CPP_URL", "http://localhost:8000")
NVIDIA_TTS_URL = os.getenv("NVIDIA_TTS_URL", "http://localhost:8001")

# Transport configurations with SIMPLE VAD only (no SmartTurn)
# Using stop_secs=0.8 (800ms) for conservative turn detection
transport_params = {
    "daily": lambda: DailyParams(
        audio_in_enabled=True,
        audio_out_enabled=True,
        vad_analyzer=SileroVADAnalyzer(params=VADParams(stop_secs=0.8)),
        # NO turn_analyzer - just simple VAD
    ),
    "twilio": lambda: FastAPIWebsocketParams(
        audio_in_enabled=True,
        audio_out_enabled=True,
        vad_analyzer=SileroVADAnalyzer(params=VADParams(stop_secs=0.8)),
        # NO turn_analyzer
    ),
    "webrtc": lambda: TransportParams(
        audio_in_enabled=True,
        audio_out_enabled=True,
        vad_analyzer=SileroVADAnalyzer(params=VADParams(stop_secs=0.8)),
        # NO turn_analyzer
    ),
}


async def run_bot(transport: BaseTransport, runner_args: RunnerArguments):
    logger.info("Starting simple VAD bot (NO SmartTurn)")
    logger.info(f"  ASR URL: {NVIDIA_ASR_URL}")
    logger.info(f"  LLM URL: {NVIDIA_LLAMA_CPP_URL}")
    logger.info(f"  TTS URL: {NVIDIA_TTS_URL}")
    logger.info(f"  VAD: Silero with stop_secs=0.8 (NO SmartTurn)")
    logger.info(f"  Transport: {type(transport).__name__}")

    # NVIDIA Parakeet ASR via WebSocket
    stt = NVidiaWebSocketSTTService(
        url=NVIDIA_ASR_URL,
        sample_rate=16000,
    )

    # Orpheus TTS via local HTTP streaming server.
    tts = OrpheusHTTPTTSService(
        server_url=NVIDIA_TTS_URL,
        voice=os.getenv("ORPHEUS_VOICE", "tara"),
        params=OrpheusHTTPTTSService.InputParams(
            language="en",
        ),
    )
    logger.info("Using Orpheus HTTP TTS")

    # Chunked LLM - sentence-boundary streaming direct to llama.cpp
    llm = LlamaCppChunkedLLMService(
        llama_url=NVIDIA_LLAMA_CPP_URL,
        params=LlamaCppChunkedLLMService.InputParams(
            first_chunk_min_tokens=10,
            first_chunk_max_tokens=24,
        ),
    )
    logger.info("Using LlamaCppChunkedLLMService (sentence-boundary streaming)")

    messages = [
        {
            "role": "system",
            "content": (
                "You are a helpful AI assistant running on an NVIDIA DGX Spark. "
                "You are built with Nemotron Three Nano, a large language model developed by NVIDIA. "
                "Your goal is to have a natural conversation with the user. "
                "Keep your responses concise and conversational since they will be spoken aloud. "
                "Avoid special characters. Use only simple, plain text sentences. "
                "Always punctuate your responses using standard sentence punctuation: commas, periods, question marks, exclamation points, etc."
                "Always spell out numbers as words."
            ),
        },
        {
            "role": "user",
            "content": "Say hello and ask how you can help.",
        },
    ]

    context = LLMContext(messages)
    context_aggregator = LLMContextAggregatorPair(context)

    rtvi = RTVIProcessor(config=RTVIConfig(config=[]))

    pipeline = Pipeline(
        [
            transport.input(),
            rtvi,
            stt,
            context_aggregator.user(),
            llm,
            tts,
            transport.output(),
            context_aggregator.assistant(),
        ]
    )

    task = PipelineTask(
        pipeline,
        params=PipelineParams(
            enable_metrics=True,
            enable_usage_metrics=True,
        ),
        observers=[RTVIObserver(rtvi)],
        idle_timeout_secs=runner_args.pipeline_idle_timeout_secs,
    )

    @rtvi.event_handler("on_client_ready")
    async def on_client_ready(rtvi):
        logger.info("RTVI client ready")
        await rtvi.set_bot_ready()
        await task.queue_frames([LLMRunFrame()])

    @transport.event_handler("on_client_disconnected")
    async def on_client_disconnected(transport, client):
        logger.info("Client disconnected")
        await task.cancel()

    runner = PipelineRunner(handle_sigint=runner_args.handle_sigint)
    await runner.run(task)


async def bot(runner_args: RunnerArguments):
    """Main bot entry point compatible with Pipecat runner."""
    transport = await create_transport(runner_args, transport_params)
    await run_bot(transport, runner_args)


if __name__ == "__main__":
    from pipecat.runner.run import main

    main()
