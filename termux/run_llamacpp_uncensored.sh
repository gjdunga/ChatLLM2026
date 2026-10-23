#!/data/data/com.termux/files/usr/bin/bash
# Fully on-device fallback: no PC, no APK, no ADB. Runs an abliterated
# Qwen3 4B Instruct 2507 (GGUF Q4_K_M) with llama.cpp inside Termux and serves
# a chat web UI on http://127.0.0.1:8080 (open it in the phone's browser).
#
# Use this when the MLC route is blocked (no x86_64 PC for conversion, or
# Android denies adb shell writes to Android/data). CPU only; expect roughly
# 10 to 20 tokens/s on a Snapdragon 8 Elite class SoC.
set -euo pipefail

REPO="${REPO:-mradermacher/Huihui-Qwen3-4B-Instruct-2507-abliterated-GGUF}"
FILE="${FILE:-Huihui-Qwen3-4B-Instruct-2507-abliterated.Q4_K_M.gguf}"
SIZE_EXPECTED="${SIZE_EXPECTED:-2497281312}"
MODELS="$HOME/models"
THREADS="${THREADS:-6}"   # 8 Elite: 2 prime + 6 perf cores; leave headroom for the UI
CTX="${CTX:-8192}"

pkg update -y
pkg install -y llama-cpp curl

mkdir -p "$MODELS"
dst="$MODELS/$FILE"
if [ ! -f "$dst" ] || [ "$(stat -c %s "$dst")" != "$SIZE_EXPECTED" ]; then
  echo "downloading $FILE (resumable)"
  curl -fL --retry 5 -C - -o "$dst" "https://huggingface.co/$REPO/resolve/main/$FILE"
fi
actual=$(stat -c %s "$dst")
if [ "$actual" != "$SIZE_EXPECTED" ]; then
  echo "ERROR: size $actual != $SIZE_EXPECTED, delete $dst and rerun" >&2
  exit 1
fi

# Keep the CPU awake while serving; Android otherwise throttles Termux in the background.
command -v termux-wake-lock >/dev/null && termux-wake-lock || true

echo "serving on http://127.0.0.1:8080  (Ctrl+C to stop)"
exec llama-server -m "$dst" -t "$THREADS" -c "$CTX" \
  --host 127.0.0.1 --port 8080 --jinja
