#!/bin/bash
# Host-side manager for the experimental Voxtral Realtime ASR container.

set -e

CONTAINER_NAME="${VOXTRAL_CONTAINER_NAME:-voxtral-asr}"
IMAGE_NAME="${VOXTRAL_IMAGE:-voxtral-asr:realtime}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

print_usage() {
    cat << 'EOF'
Voxtral Realtime ASR Manager

Usage:
  ./scripts/voxtral.sh build
  ./scripts/voxtral.sh start
  ./scripts/voxtral.sh stop
  ./scripts/voxtral.sh restart
  ./scripts/voxtral.sh status
  ./scripts/voxtral.sh logs

Environment:
  VOXTRAL_IMAGE                   Docker image (default: voxtral-asr:realtime)
  VOXTRAL_CONTAINER_NAME          Container name (default: voxtral-asr)
  VOXTRAL_MODEL                   HF model id
  VOXTRAL_GPU_MEMORY_UTILIZATION  vLLM GPU fraction (default: 0.35)
  VOXTRAL_MAX_MODEL_LEN           vLLM max model length (default: 32768)
  HUGGINGFACE_ACCESS_TOKEN        Optional HF token
EOF
}

is_running() {
    docker ps --format '{{.Names}}' | grep -q "^${CONTAINER_NAME}$"
}

exists() {
    docker ps -a --format '{{.Names}}' | grep -q "^${CONTAINER_NAME}$"
}

cmd_build() {
    docker build -f "$PROJECT_DIR/Dockerfile.voxtral-asr" -t "$IMAGE_NAME" "$PROJECT_DIR"
}

cmd_start() {
    if is_running; then
        echo "Container '$CONTAINER_NAME' is already running"
        return 0
    fi
    if exists; then
        docker rm "$CONTAINER_NAME" >/dev/null
    fi

    DOCKER_ARGS=(
        run
        --name "$CONTAINER_NAME"
        --gpus all
        --ipc=host
        -d
        -v "$PROJECT_DIR:/workspace"
        -v "$HOME/.cache/huggingface:/root/.cache/huggingface"
        -p 8082:8082
        -e "HF_HOME=/root/.cache/huggingface"
        -e "HF_HUB_DISABLE_XET=${HF_HUB_DISABLE_XET:-1}"
        -e "VOXTRAL_MODEL=${VOXTRAL_MODEL:-mistralai/Voxtral-Mini-4B-Realtime-2602}"
        -e "VOXTRAL_GPU_MEMORY_UTILIZATION=${VOXTRAL_GPU_MEMORY_UTILIZATION:-0.35}"
        -e "VOXTRAL_MAX_MODEL_LEN=${VOXTRAL_MAX_MODEL_LEN:-32768}"
    )

    if [[ -n "${HUGGINGFACE_ACCESS_TOKEN:-}" ]]; then
        DOCKER_ARGS+=(-e "HUGGINGFACE_ACCESS_TOKEN=$HUGGINGFACE_ACCESS_TOKEN")
        DOCKER_ARGS+=(-e "HF_TOKEN=$HUGGINGFACE_ACCESS_TOKEN")
    fi

    DOCKER_ARGS+=("$IMAGE_NAME")
    docker "${DOCKER_ARGS[@]}"
    echo "Voxtral ASR starting on ws://localhost:8082/v1/realtime"
}

cmd_stop() {
    if is_running; then
        docker stop "$CONTAINER_NAME" >/dev/null
    fi
    if exists; then
        docker rm "$CONTAINER_NAME" >/dev/null
    fi
    echo "Voxtral ASR stopped"
}

cmd_status() {
    if is_running; then
        docker ps --filter "name=^/${CONTAINER_NAME}$" --format '{{.Names}}\t{{.Status}}\t{{.Ports}}'
    else
        echo "Container '$CONTAINER_NAME' is not running"
    fi
}

cmd_logs() {
    docker logs -f "$CONTAINER_NAME"
}

COMMAND="${1:-help}"
shift || true

case "$COMMAND" in
    build) cmd_build "$@" ;;
    start) cmd_start "$@" ;;
    stop) cmd_stop ;;
    restart) cmd_stop; cmd_start "$@" ;;
    status) cmd_status ;;
    logs) cmd_logs ;;
    help|--help|-h) print_usage ;;
    *) echo "Unknown command: $COMMAND"; print_usage; exit 1 ;;
esac
