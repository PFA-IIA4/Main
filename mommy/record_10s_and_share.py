#!/usr/bin/env python3
import argparse
import datetime as dt
import shutil
import socket
import subprocess
import sys
from pathlib import Path


def run(cmd):
    print("Running:", " ".join(cmd))
    subprocess.run(cmd, check=True)


def is_port_open(port):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(0.2)
        return s.connect_ex(("127.0.0.1", port)) == 0


def get_ipv4_addresses():
    try:
        out = subprocess.check_output(["hostname", "-I"], text=True).strip()
        return [ip for ip in out.split() if ip and "." in ip]
    except Exception:
        return []


def write_listen_page(out_dir):
    html = """<!doctype html>
<html>
<head>
  <meta charset=\"utf-8\" />
  <meta name=\"viewport\" content=\"width=device-width,initial-scale=1\" />
  <title>Latest Mic Recording</title>
  <style>
    body { font-family: sans-serif; background:#0f1115; color:#eaf0ff; margin:0; }
    .wrap { max-width:720px; margin:40px auto; padding:0 16px; }
    .card { background:#1a1f29; border-radius:14px; padding:20px; }
    audio { width:100%; margin-top:10px; }
    .meta { color:#a9b5c9; margin-top:8px; font-size:14px; }
    button { margin-top:12px; padding:8px 12px; border-radius:8px; border:0; cursor:pointer; }
  </style>
</head>
<body>
  <div class=\"wrap\">
    <div class=\"card\">
      <h2>Latest INMP441 Recording</h2>
      <audio id=\"a\" controls autoplay>
        <source src=\"latest.mp3\" type=\"audio/mpeg\" />
      </audio>
      <div class=\"meta\" id=\"m\"></div>
      <button onclick=\"reloadAudio()\">Reload Latest</button>
    </div>
  </div>
  <script>
    const m = document.getElementById('m');
    m.textContent = 'If autoplay is blocked, press play.';
    function reloadAudio() {
      const a = document.getElementById('a');
      a.src = 'latest.mp3?t=' + Date.now();
      a.play().catch(() => {});
    }
  </script>
</body>
</html>
"""
    (out_dir / "listen.html").write_text(html, encoding="utf-8")


def ensure_server(out_dir, port, log_file):
    if is_port_open(port):
        return

    log_file.parent.mkdir(parents=True, exist_ok=True)
    with log_file.open("a", encoding="utf-8") as f:
        subprocess.Popen(
            [
                sys.executable,
                "-m",
                "http.server",
                str(port),
                "--directory",
                str(out_dir),
            ],
            stdout=f,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )


def main():
    parser = argparse.ArgumentParser(
        description="Record 10s voice from INMP441 and serve it for laptop playback"
    )
    parser.add_argument("--duration", type=int, default=10, help="Recording length in seconds")
    parser.add_argument("--device", default="plughw:1,0", help="ALSA capture device")
    parser.add_argument("--rate", type=int, default=16000, help="Sample rate")
    parser.add_argument("--channels", type=int, default=1, help="Number of channels")
    parser.add_argument("--port", type=int, default=8091, help="HTTP port for listening page")
    parser.add_argument(
        "--out-dir",
        default=str(Path(__file__).resolve().parent),
        help="Directory to save recordings and web files",
    )
    args = parser.parse_args()

    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    now = dt.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    wav_file = out_dir / f"vocals_{now}.wav"
    mp3_file = out_dir / f"vocals_{now}.mp3"
    latest_mp3 = out_dir / "latest.mp3"
    log_file = out_dir / ".cam_logs" / "mic_web.log"

    run(
        [
            "arecord",
            "-D",
            args.device,
            "-f",
            "S16_LE",
            "-r",
            str(args.rate),
            "-c",
            str(args.channels),
            "-d",
            str(args.duration),
            str(wav_file),
        ]
    )

    run(
        [
            "ffmpeg",
            "-y",
            "-hide_banner",
            "-loglevel",
            "error",
            "-i",
            str(wav_file),
            "-codec:a",
            "libmp3lame",
            "-q:a",
            "2",
            str(mp3_file),
        ]
    )

    shutil.copy2(mp3_file, latest_mp3)
    write_listen_page(out_dir)
    ensure_server(out_dir, args.port, log_file)

    print("\nDone.")
    print("Saved WAV:", wav_file)
    print("Saved MP3:", mp3_file)
    print("Latest MP3:", latest_mp3)

    ips = get_ipv4_addresses()
    if ips:
        print("\nOpen on your laptop:")
        for ip in ips:
            print(f"http://{ip}:{args.port}/listen.html")
    else:
        print("\nOpen:")
        print(f"http://127.0.0.1:{args.port}/listen.html")


if __name__ == "__main__":
    main()
