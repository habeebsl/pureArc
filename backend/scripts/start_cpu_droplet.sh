#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

if [[ ! -d venv ]]; then
  python3 -m venv venv
fi

source venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements-cpu.txt

if [[ -f .env ]]; then
  set -a
  source .env
  set +a
fi

mkdir -p logs

export PUREARC_API_BASE_URL="${PUREARC_API_BASE_URL:-http://127.0.0.1:8000}"
export PUREARC_VIDEO_SOURCE="${PUREARC_VIDEO_SOURCE:-}"
export PUREARC_SAVE_OUTPUT="${PUREARC_SAVE_OUTPUT:-0}"
export PUREARC_RUN_MAIN="${PUREARC_RUN_MAIN:-1}"

if [[ -z "${PUREARC_API_BASE_URL}" ]]; then
  echo "PUREARC_API_BASE_URL is required"
  exit 1
fi

if pgrep -f "uvicorn api.app:app" >/dev/null 2>&1; then
  echo "API already running"
else
  nohup python -m uvicorn api.app:app --host 0.0.0.0 --port 8000 > logs/api.log 2>&1 &
  echo "Started API (pid=$!)"
fi

if [[ "$PUREARC_RUN_MAIN" == "1" ]]; then
  if [[ -z "$PUREARC_VIDEO_SOURCE" ]]; then
    echo "PUREARC_VIDEO_SOURCE is empty. Skipping main.py."
    echo "Set PUREARC_VIDEO_SOURCE to a camera index (e.g. 0) or a video file path."
  elif [[ "$PUREARC_VIDEO_SOURCE" =~ ^[0-9]+$ ]] || [[ -f "$PUREARC_VIDEO_SOURCE" ]]; then
    if pgrep -f "python main.py" >/dev/null 2>&1; then
      echo "main.py already running"
    else
      nohup python main.py > logs/main.log 2>&1 &
      echo "Started main.py (pid=$!)"
    fi
  else
    echo "PUREARC_VIDEO_SOURCE does not exist: $PUREARC_VIDEO_SOURCE"
    echo "Skipping main.py startup."
  fi
else
  echo "PUREARC_RUN_MAIN=0, skipping main.py startup"
fi

echo ""
echo "PureArc CPU services started"
echo "API:  http://<droplet-ip>:8000/health"
echo "Logs: logs/api.log and logs/main.log"
