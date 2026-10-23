#!/usr/bin/env bash
# Install the prebuilt MLC Chat APK (if missing) and side-load a converted
# weights folder into it as an extra model, over ADB.
#
# Runs on: Termux on the phone itself (adb over Wireless debugging to
# 127.0.0.1), or any Linux/macOS host with adb and a USB/Wi-Fi connected phone.
#   Termux deps: pkg install android-tools python unzip curl
#
# Usage: install_mlc_model.sh /path/to/dist/<MODEL_ID>
#   ADB_SERIAL=127.0.0.1:37123 install_mlc_model.sh ...   (pick a device)
set -euo pipefail

MODEL_DIR="${1:?usage: install_mlc_model.sh /path/to/<MODEL_ID>}"
MODEL_DIR="${MODEL_DIR%/}"
MODEL_ID="$(basename "$MODEL_DIR")"
LIB_PREFIX="${LIB_PREFIX:-llama_q4f16_0_}"
PKG="ai.mlc.mlcchat"
APK_URL="${APK_URL:-https://github.com/mlc-ai/binary-mlc-llm-libs/releases/download/Android-09262024/mlc-chat.apk}"
APP_DIR="/sdcard/Android/data/$PKG/files"
HERE="$(cd "$(dirname "$0")" && pwd)"
TMP="$(mktemp -d "${TMPDIR:-/tmp}/mlcinst.XXXXXX")"
trap 'rm -rf "$TMP"' EXIT

ADB=(adb)
[ -n "${ADB_SERIAL:-}" ] && ADB=(adb -s "$ADB_SERIAL")
sh_dev() { "${ADB[@]}" shell "$@" | tr -d '\r'; }

for t in adb python3 unzip curl; do
  command -v "$t" >/dev/null || { echo "ERROR: missing $t" >&2; exit 1; }
done

# 0. Local sanity: the folder must look like a converted MLC model.
for f in mlc-chat-config.json ndarray-cache.json tokenizer.json; do
  [ -f "$MODEL_DIR/$f" ] || { echo "ERROR: $MODEL_DIR/$f missing" >&2; exit 1; }
done
nshards=$(ls "$MODEL_DIR"/params_shard_*.bin | wc -l)
echo "model: $MODEL_ID ($nshards shards, $(du -sh "$MODEL_DIR" | cut -f1))"

# 1. Device reachable.
state=$("${ADB[@]}" get-state 2>/dev/null || true)
if [ "$state" != "device" ]; then
  echo "ERROR: no ADB device. In Termux: Settings > Developer options > Wireless debugging," >&2
  echo "  'Pair device with pairing code', then: adb pair 127.0.0.1:<pair-port> <code>" >&2
  echo "  then: adb connect 127.0.0.1:<port shown on the Wireless debugging screen>" >&2
  exit 1
fi
echo "device: $(sh_dev getprop ro.product.model) Android $(sh_dev getprop ro.build.version.release)"

# 2. APK: install if absent. Keep a local copy to read the bundled config.
if sh_dev pm path "$PKG" | grep -q '^package:'; then
  apk_remote=$(sh_dev pm path "$PKG" | grep base.apk | head -1 | sed 's/^package://')
  "${ADB[@]}" pull "$apk_remote" "$TMP/app.apk" >/dev/null
  echo "MLC Chat already installed"
else
  echo "downloading MLC Chat APK"
  curl -fL --retry 4 -o "$TMP/app.apk" "$APK_URL"
  # Newer Android refuses very old targetSdk APKs; retry with the documented bypass.
  "${ADB[@]}" install "$TMP/app.apk" || "${ADB[@]}" install --bypass-low-target-sdk-block "$TMP/app.apk"
fi

# 3. First launch creates the app's external files dir; then stop it so it
#    cannot rewrite the config while we edit it.
"${ADB[@]}" shell monkey -p "$PKG" -c android.intent.category.LAUNCHER 1 >/dev/null 2>&1 || true
sleep 4
"${ADB[@]}" shell am force-stop "$PKG"
sh_dev mkdir -p "$APP_DIR"
if ! sh_dev "touch $APP_DIR/.probe && rm $APP_DIR/.probe && echo ok" | grep -q ok; then
  echo "ERROR: adb shell cannot write $APP_DIR on this Android build." >&2
  echo "  Fallback: use the Termux llama.cpp route (termux/run_llamacpp_uncensored.sh)." >&2
  exit 1
fi

# 4. Config: the on-device copy wins if the app already wrote one.
if sh_dev "[ -f $APP_DIR/mlc-app-config.json ] && echo y" | grep -q y; then
  "${ADB[@]}" pull "$APP_DIR/mlc-app-config.json" "$TMP/in.json" >/dev/null
else
  unzip -p "$TMP/app.apk" assets/mlc-app-config.json > "$TMP/in.json"
fi
python3 -I "$HERE/merge_app_config.py" "$TMP/in.json" "$TMP/out.json" "$MODEL_ID" "$LIB_PREFIX"

# 5. Weights. Push into a staging name, verify, then swap in, so an
#    interrupted push never leaves a half model under the real id.
stage="$APP_DIR/.$MODEL_ID.partial"
sh_dev rm -rf "$stage"
sh_dev mkdir -p "$stage"
"${ADB[@]}" push "$MODEL_DIR"/. "$stage/"
remote_bytes=$(sh_dev "du -sk $stage" | cut -f1)
local_bytes=$(du -sk "$MODEL_DIR" | cut -f1)
echo "pushed ${remote_bytes} KiB (local ${local_bytes} KiB)"
remote_shards=$(sh_dev "ls $stage" | grep -c '^params_shard_.*\.bin$' || true)
[ "$remote_shards" = "$nshards" ] || { echo "ERROR: shard count $remote_shards != $nshards" >&2; exit 1; }
sh_dev rm -rf "$APP_DIR/$MODEL_ID"
sh_dev mv "$stage" "$APP_DIR/$MODEL_ID"

"${ADB[@]}" push "$TMP/out.json" "$APP_DIR/mlc-app-config.json" >/dev/null

# 6. Start the app. The new model appears in the list as $MODEL_ID.
"${ADB[@]}" shell monkey -p "$PKG" -c android.intent.category.LAUNCHER 1 >/dev/null 2>&1 || true
echo
echo "Installed. Open MLC Chat and tap the chat icon on $MODEL_ID."
echo "Crash triage: adb logcat -s TVM_RUNTIME AndroidRuntime MLCChat"
