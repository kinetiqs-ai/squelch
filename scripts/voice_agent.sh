#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

RIVA_CONTAINER="${SQUELCH_RIVA_CONTAINER:-riva-speech}"
LLM_CONTAINER="${SQUELCH_LLM_CONTAINER:-squelch-llm-native}"
TTS_CONTAINER="${SQUELCH_TTS_CONTAINER:-squelch-magpie-native}"

INGRESS_HOST="${SQUELCH_INGRESS_HOST:-127.0.0.1}"
INGRESS_PORT="${SQUELCH_INGRESS_PORT:-7860}"
LLM_HEALTH_URL="${SQUELCH_LLM_HEALTH_URL:-http://127.0.0.1:8100/health}"
TTS_HEALTH_URL="${SQUELCH_TTS_HEALTH_URL:-http://127.0.0.1:8101/health}"
INGRESS_HEALTH_URL="${SQUELCH_INGRESS_HEALTH_URL:-http://127.0.0.1:${INGRESS_PORT}/health}"
RIVA_PORT="${SQUELCH_RIVA_PORT:-50051}"
TAILSCALE_WS_URL="${SQUELCH_TAILSCALE_WS_URL:-wss://ramp-genesis.tail314cde.ts.net:7860/ws/audio-ingress}"

RUN_DIR="${SQUELCH_RUN_DIR:-runs/services}"
INGRESS_LOG="${SQUELCH_INGRESS_LOG:-${RUN_DIR}/native_audio_ingress.log}"
INGRESS_PID_FILE="${SQUELCH_INGRESS_PID_FILE:-${RUN_DIR}/native_audio_ingress.pid}"

MODEL_TIMEOUT_S="${SQUELCH_MODEL_TIMEOUT_S:-420}"
INGRESS_TIMEOUT_S="${SQUELCH_INGRESS_TIMEOUT_S:-45}"
POLL_INTERVAL_S="${SQUELCH_POLL_INTERVAL_S:-2}"

CONTAINERS=("$RIVA_CONTAINER" "$LLM_CONTAINER" "$TTS_CONTAINER")

usage() {
    cat << EOF
Usage: $0 COMMAND

Commands:
  start     Start Riva ASR, llama.cpp, Magpie TTS, and native ingress
  stop      Stop native ingress and model containers
  restart   Stop, then start the voice agent
  status    Print process/container and endpoint status

Environment:
  SQUELCH_MODEL_TIMEOUT_S      Seconds to wait for model services (default: 420)
  SQUELCH_INGRESS_TIMEOUT_S    Seconds to wait for native ingress (default: 45)
  SQUELCH_TAILSCALE_WS_URL     External audio-edge-agent websocket URL
EOF
}

container_exists() {
    docker inspect "$1" >/dev/null 2>&1
}

container_running() {
    docker inspect -f '{{.State.Running}}' "$1" 2>/dev/null | grep -q '^true$'
}

require_containers() {
    local missing=()
    for container in "${CONTAINERS[@]}"; do
        if ! container_exists "$container"; then
            missing+=("$container")
        fi
    done
    if ((${#missing[@]} > 0)); then
        printf 'ERROR: missing required container(s): %s\n' "${missing[*]}" >&2
        exit 1
    fi
}

http_ok() {
    curl -fsS "$1" >/dev/null 2>&1
}

tcp_ok() {
    ss -ltn "sport = :$1" | awk 'NR > 1 {found=1} END {exit found ? 0 : 1}'
}

wait_for_http() {
    local name="$1"
    local url="$2"
    local timeout="$3"
    local elapsed=0

    printf 'Waiting for %s' "$name"
    while ((elapsed < timeout)); do
        if http_ok "$url"; then
            printf ' ready\n'
            return 0
        fi
        printf '.'
        sleep "$POLL_INTERVAL_S"
        elapsed=$((elapsed + POLL_INTERVAL_S))
    done
    printf ' timeout\n' >&2
    return 1
}

wait_for_tcp() {
    local name="$1"
    local port="$2"
    local timeout="$3"
    local elapsed=0

    printf 'Waiting for %s' "$name"
    while ((elapsed < timeout)); do
        if tcp_ok "$port"; then
            printf ' ready\n'
            return 0
        fi
        printf '.'
        sleep "$POLL_INTERVAL_S"
        elapsed=$((elapsed + POLL_INTERVAL_S))
    done
    printf ' timeout\n' >&2
    return 1
}

ingress_pids() {
    pgrep -f 'native_voice.riva_asr_app' || true
}

ingress_running() {
    [[ -n "$(ingress_pids)" ]]
}

stop_ingress() {
    local pids
    pids="$(ingress_pids)"
    if [[ -z "$pids" ]]; then
        rm -f "$INGRESS_PID_FILE"
        return 0
    fi

    echo "Stopping native ingress..."
    pkill -f 'native_voice.riva_asr_app' || true
    for _ in $(seq 1 20); do
        if ! ingress_running; then
            rm -f "$INGRESS_PID_FILE"
            return 0
        fi
        sleep 0.5
    done
    pkill -9 -f 'native_voice.riva_asr_app' || true
    rm -f "$INGRESS_PID_FILE"
}

start_ingress() {
    mkdir -p "$RUN_DIR"
    if ingress_running; then
        echo "Native ingress already running: $(ingress_pids | tr '\n' ' ')"
        return 0
    fi

    echo "Starting native ingress..."
    : > "$INGRESS_LOG"
    setsid -f bash -c "./scripts/start_native_audio_ingress.sh > '$INGRESS_LOG' 2>&1"

    local elapsed=0
    while ((elapsed < INGRESS_TIMEOUT_S)); do
        if http_ok "$INGRESS_HEALTH_URL"; then
            ingress_pids | head -n 1 > "$INGRESS_PID_FILE"
            echo "Native ingress ready"
            return 0
        fi
        if ! ingress_running; then
            echo "ERROR: native ingress exited during startup" >&2
            tail -80 "$INGRESS_LOG" >&2 || true
            return 1
        fi
        sleep "$POLL_INTERVAL_S"
        elapsed=$((elapsed + POLL_INTERVAL_S))
    done

    echo "ERROR: native ingress did not become healthy in ${INGRESS_TIMEOUT_S}s" >&2
    tail -80 "$INGRESS_LOG" >&2 || true
    return 1
}

cmd_start() {
    require_containers

    echo "Starting model containers..."
    docker start "$RIVA_CONTAINER" "$LLM_CONTAINER" "$TTS_CONTAINER" >/dev/null

    wait_for_tcp "Riva ASR on ${RIVA_PORT}" "$RIVA_PORT" "$MODEL_TIMEOUT_S"
    wait_for_http "llama.cpp LLM" "$LLM_HEALTH_URL" "$MODEL_TIMEOUT_S"
    wait_for_http "Magpie TTS" "$TTS_HEALTH_URL" "$MODEL_TIMEOUT_S"
    start_ingress

    echo
    echo "Voice agent is ready:"
    echo "  $TAILSCALE_WS_URL"
}

cmd_stop() {
    stop_ingress

    require_containers
    echo "Stopping model containers..."
    docker stop "$TTS_CONTAINER" "$LLM_CONTAINER" "$RIVA_CONTAINER" >/dev/null
    echo "Voice agent stopped"
}

status_line() {
    local name="$1"
    shift
    if "$@"; then
        printf '  %-18s OK\n' "$name"
    else
        printf '  %-18s DOWN\n' "$name"
    fi
}

cmd_status() {
    echo "Containers:"
    for container in "${CONTAINERS[@]}"; do
        if container_exists "$container"; then
            docker ps -a --filter "name=^/${container}$" --format '  {{.Names}}\t{{.Status}}\t{{.Ports}}'
        else
            echo "  $container missing"
        fi
    done

    echo
    echo "Endpoints:"
    status_line "Riva ASR" tcp_ok "$RIVA_PORT"
    status_line "LLM" http_ok "$LLM_HEALTH_URL"
    status_line "Magpie TTS" http_ok "$TTS_HEALTH_URL"
    status_line "Native ingress" http_ok "$INGRESS_HEALTH_URL"

    echo
    if ingress_running; then
        echo "Native ingress PIDs: $(ingress_pids | tr '\n' ' ')"
    else
        echo "Native ingress PIDs: none"
    fi
    echo "Ingress log: $INGRESS_LOG"
    echo "External URL: $TAILSCALE_WS_URL"
}

case "${1:-}" in
    start) cmd_start ;;
    stop) cmd_stop ;;
    restart) cmd_stop; cmd_start ;;
    status) cmd_status ;;
    help|--help|-h) usage ;;
    *) usage; exit 1 ;;
esac
