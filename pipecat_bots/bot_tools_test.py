#!/usr/bin/env python3
#
# PipeCat tool calling validation bot for Issue #197 spike.
#
# Tests tool/function calling through the full PipeCat pipeline:
#   ASR (NVidiaWebSocketSTTService) -> OpenAILLMService (llama.cpp) -> TTS (OrpheusHTTPTTSService)
#
# Uses OpenAILLMService instead of LlamaCppBufferedLLMService because the
# buffered service does not implement tool calling. OpenAILLMService connects
# to the same llama.cpp endpoint (port 8000) via its OpenAI-compatible API.
#
# Tool: get_current_time — returns current date/time. Ask "what time is it?"
#
# Environment variables:
#   ASR_BACKEND           nemotron (default) or voxtral
#   NVIDIA_ASR_URL        Nemotron ASR WebSocket URL (default: ws://localhost:8080)
#   VOXTRAL_ASR_URL       Voxtral Realtime URL (default: ws://localhost:8082/v1/realtime)
#   NVIDIA_LLM_URL        llama.cpp OpenAI API URL (default: http://localhost:8000/v1)
#   NVIDIA_LLM_MODEL      Model name as returned by llama.cpp (default: auto-detected)
#   NVIDIA_TTS_URL        Orpheus TTS server URL (default: http://localhost:8001)
#
# Usage (run from reference repo root):
#   uv run spike/pipecat-reference/bot_tools_test.py -t webrtc --host 0.0.0.0
#   Open: http://<tailscale-ip>:7861/client
#
# NOTE: Runs on port 7861 to avoid conflict with bot_interleaved_streaming.py (7860).

import asyncio
import os
import sys
from datetime import datetime

from dotenv import load_dotenv
from loguru import logger

from pipecat.adapters.schemas.function_schema import FunctionSchema
from pipecat.adapters.schemas.tools_schema import ToolsSchema
from pipecat.audio.turn.smart_turn.base_smart_turn import SmartTurnParams
from pipecat.audio.turn.smart_turn.local_smart_turn_v3 import LocalSmartTurnAnalyzerV3
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
from pipecat.services.llm_service import FunctionCallParams
from pipecat.services.openai.llm import OpenAILLMService
from pipecat.transports.base_transport import BaseTransport, TransportParams
from pipecat.transports.daily.transport import DailyParams
from pipecat.transports.websocket.fastapi import FastAPIWebsocketParams
import httpx

# Add pipecat_bots/ to path so we can import the custom services
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../pipecat_bots"))

from asr_factory import create_stt_service, describe_asr_backend  # noqa: E402
from orpheus_http_tts import OrpheusHTTPTTSService  # noqa: E402

load_dotenv(override=True)

NVIDIA_LLM_URL = os.getenv("NVIDIA_LLM_URL", "http://localhost:8000/v1")
NVIDIA_LLM_MODEL = os.getenv("NVIDIA_LLM_MODEL", "")
NVIDIA_TTS_URL = os.getenv("NVIDIA_TTS_URL", "http://localhost:8001")

VAD_CONFIDENCE = float(os.getenv("VAD_CONFIDENCE", "0.7"))
VAD_START_SECS = float(os.getenv("VAD_START_SECS", "0.12"))
VAD_STOP_SECS = float(os.getenv("VAD_STOP_SECS", "0.2"))
VAD_MIN_VOLUME = float(os.getenv("VAD_MIN_VOLUME", "0.25"))


def vad_params() -> VADParams:
    return VADParams(
        confidence=VAD_CONFIDENCE,
        start_secs=VAD_START_SECS,
        stop_secs=VAD_STOP_SECS,
        min_volume=VAD_MIN_VOLUME,
    )

transport_params = {
    "daily": lambda: DailyParams(
        audio_in_enabled=True,
        audio_out_enabled=True,
        vad_audio_passthrough=True,
        vad_analyzer=SileroVADAnalyzer(params=vad_params()),
        turn_analyzer=LocalSmartTurnAnalyzerV3(params=SmartTurnParams()),
    ),
    "twilio": lambda: FastAPIWebsocketParams(
        audio_in_enabled=True,
        audio_out_enabled=True,
        vad_audio_passthrough=True,
        vad_analyzer=SileroVADAnalyzer(params=vad_params()),
        turn_analyzer=LocalSmartTurnAnalyzerV3(params=SmartTurnParams()),
    ),
    "webrtc": lambda: TransportParams(
        audio_in_enabled=True,
        audio_out_enabled=True,
        vad_audio_passthrough=True,
        vad_analyzer=SileroVADAnalyzer(params=vad_params()),
        turn_analyzer=LocalSmartTurnAnalyzerV3(params=SmartTurnParams()),
        port=7861,
    ),
}


async def _get_model_name(base_url: str) -> str:
    """Fetch the first available model name from the llama.cpp /v1/models endpoint."""
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(f"{base_url.removesuffix('/v1').rstrip('/')}/v1/models", timeout=5)
            data = response.json()
            if data.get("data"):
                return data["data"][0]["id"]
    except Exception as exc:
        logger.warning(f"Could not auto-detect model name: {exc}")
    return "local-model"


async def run_bot(transport: BaseTransport, runner_args: RunnerArguments):
    logger.info("Starting tool calling test bot")
    logger.info(f"  ASR: {describe_asr_backend()}")
    logger.info(f"  LLM URL: {NVIDIA_LLM_URL}")
    logger.info(f"  TTS URL: {NVIDIA_TTS_URL}")

    # Auto-detect model name if not set (llama.cpp reports a path-based name)
    model_name = NVIDIA_LLM_MODEL or await _get_model_name(NVIDIA_LLM_URL)
    logger.info(f"  LLM Model: {model_name}")

    tools = ToolsSchema(standard_tools=[
        FunctionSchema(
            name="get_current_time",
            description="Returns the current date and time on the server.",
            properties={},
            required=[],
        )
    ])

    stt = create_stt_service(sample_rate=16000)

    tts = OrpheusHTTPTTSService(
        server_url=NVIDIA_TTS_URL,
        voice=os.getenv("ORPHEUS_VOICE", "tara"),
        params=OrpheusHTTPTTSService.InputParams(
            language="en",
        ),
    )

    llm = OpenAILLMService(
        api_key="not-needed",
        base_url=NVIDIA_LLM_URL,
        model=model_name,
    )

    async def get_current_time(params: FunctionCallParams):
        now = datetime.now().strftime("%A, %B %d, %Y at %I:%M %p")
        logger.info(f"[TOOL] get_current_time called → {now}")
        await params.result_callback({"current_time": now})

    llm.register_function("get_current_time", get_current_time)

    @llm.event_handler("on_function_calls_started")
    async def on_function_calls_started(service, function_calls):
        logger.info(f"[TOOL] Function calls started: {[fc.function_name for fc in function_calls]}")

    messages = [
        {
            "role": "system",
            "content": (
                "You are a helpful AI assistant. "
                "You have access to a function called get_current_time that returns the current date and time. "
                "When the user asks what time it is or what the current date is, call get_current_time. "
                "Keep responses short and conversational since they will be spoken aloud. "
                "Use only plain text. Spell out numbers as words."
            ),
        },
        {
            "role": "user",
            "content": "Say hello and tell the user they can ask you what time it is.",
        },
    ]

    context = LLMContext(messages, tools)
    context_aggregator = LLMContextAggregatorPair(context)

    rtvi = RTVIProcessor(config=RTVIConfig(config=[]))

    pipeline = Pipeline([
        transport.input(),
        rtvi,
        stt,
        context_aggregator.user(),
        llm,
        tts,
        transport.output(),
        context_aggregator.assistant(),
    ])

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
        logger.info("RTVI client ready — tool calling bot ready")
        await rtvi.set_bot_ready()
        await task.queue_frames([LLMRunFrame()])

    @transport.event_handler("on_client_disconnected")
    async def on_client_disconnected(transport, client):
        logger.info("Client disconnected")
        await task.cancel()

    runner = PipelineRunner(handle_sigint=runner_args.handle_sigint)
    await runner.run(task)


async def bot(runner_args: RunnerArguments):
    transport = await create_transport(runner_args, transport_params)
    await run_bot(transport, runner_args)


if __name__ == "__main__":
    from pipecat.runner.run import main
    main()
