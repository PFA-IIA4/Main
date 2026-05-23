#!/usr/bin/env bash
set -euo pipefail

BASE_DIR="${1:-$HOME/mommy}"
FPS="${FPS:-15}"
SIZE="${SIZE:-640x480}"
PORT="${PORT:-8090}"
LOG_DIR="$BASE_DIR/.cam_logs"
FRAME_FILE="$BASE_DIR/frame.jpg"
HTML_FILE="$BASE_DIR/cam.html"

mkdir -p "$BASE_DIR" "$LOG_DIR"

# Browser-friendly viewer that refreshes a JPEG frame.
cat > "$HTML_FILE" <<'HTML'
<!doctype html>
<html>
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width,initial-scale=1" />
  <title>Pi Cam Live</title>
  <style>
    body { font-family: sans-serif; margin: 0; background: #0b0f14; color: #e8edf3; }
    .wrap { max-width: 960px; margin: 18px auto; padding: 0 14px; }
    .card { background: #161b22; border-radius: 12px; padding: 12px; }
    img { width: 100%; max-height: 78vh; object-fit: contain; border-radius: 10px; background: #000; display: block; }
    .meta { margin-top: 8px; font-size: 14px; color: #9fb0c3; }
  </style>
</head>
<body>
  <div class="wrap">
    <h2>Raspberry Pi Camera Live Feed</h2>
    <div class="card">
      <img id="feed" alt="live camera frame" src="/frame.jpg" />
      <div class="meta">Auto-refreshing JPEG feed</div>
    </div>
  </div>
  <script>
    const img = document.getElementById('feed');
    function tick() { img.src = '/frame.jpg?t=' + Date.now(); }
    setInterval(tick, 120);
    tick();
  </script>
</body>
</html>
HTML

# Stop previous instances from older runs.
pkill -f 'ffmpeg.*live\.mjpg' 2>/dev/null || true
pkill -f "ffmpeg.*$FRAME_FILE" 2>/dev/null || true
pkill -f "python3 -m http.server $PORT --directory $BASE_DIR" 2>/dev/null || true

# Start camera frame writer.
nohup ffmpeg -y -hide_banner -loglevel error \
  -f v4l2 -framerate "$FPS" -video_size "$SIZE" -i /dev/video0 \
  -q:v 5 -update 1 "$FRAME_FILE" \
  > "$LOG_DIR/frame.log" 2>&1 &
FFMPEG_PID=$!

sleep 1
if ! kill -0 "$FFMPEG_PID" 2>/dev/null; then
  echo "Failed to start camera writer. Last log lines:"
  tail -n 40 "$LOG_DIR/frame.log" || true
  exit 1
fi

# Start simple web server.
nohup python3 -m http.server "$PORT" --directory "$BASE_DIR" \
  > "$LOG_DIR/web.log" 2>&1 &
WEB_PID=$!

sleep 1
if ! kill -0 "$WEB_PID" 2>/dev/null; then
  echo "Failed to start web server. Last log lines:"
  tail -n 40 "$LOG_DIR/web.log" || true
  exit 1
fi

echo "Pi camera started."
echo "Open one of these URLs:"
for ip in $(hostname -I); do
  [ -n "$ip" ] && echo "  http://$ip:$PORT/cam.html"
done
echo "Logs: $LOG_DIR"
