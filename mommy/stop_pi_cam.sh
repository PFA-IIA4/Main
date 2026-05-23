#!/usr/bin/env bash
set -euo pipefail

BASE_DIR="${1:-$HOME/mommy}"
PORT="${PORT:-8090}"
FRAME_FILE="$BASE_DIR/frame.jpg"

pkill -f 'ffmpeg.*live\.mjpg' 2>/dev/null || true
pkill -f "ffmpeg.*$FRAME_FILE" 2>/dev/null || true
pkill -f "python3 -m http.server $PORT --directory $BASE_DIR" 2>/dev/null || true

echo "Pi camera services stopped."
