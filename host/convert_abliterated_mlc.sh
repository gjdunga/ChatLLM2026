#!/usr/bin/env bash
# Convert an abliterated (refusal-direction removed) Llama 3.2 3B Instruct into
# MLC q4f16_0 weights that the prebuilt MLC Chat APK can run WITHOUT rebuilding
# the app, by reusing the APK's compiled llama_q4f16_0 model library.
#
# Runs on: Linux x86_64 (or WSL2, or a Colab VM). Not on the phone: MLC only
# publishes x86_64 Linux wheels, so convert_weight cannot run in Termux.
# Needs: python3 (3.10+), ~12 GB free disk, ~10 GB RAM, network to huggingface.co.
#
# Output: ./dist/<MODEL_ID>/ ready for device/install_mlc_model.sh
set -euo pipefail

SRC_REPO="${SRC_REPO:-huihui-ai/Llama-3.2-3B-Instruct-abliterated}"
REF_REPO="${REF_REPO:-mlc-ai/Llama-3.2-3B-Instruct-q4f16_0-MLC}"
MODEL_ID="${MODEL_ID:-Llama-3.2-3B-Instruct-abliterated-q4f16_0-MLC}"
QUANT="q4f16_0"   # must match the APK lib: llama_q4f16_0_2d32572d8a4ab2af20a1f587ef6c8c63
WORK="${WORK:-$PWD/work}"
DIST="${DIST:-$PWD/dist}"
HERE="$(cd "$(dirname "$0")" && pwd)"

if [ "$(uname -m)" != "x86_64" ]; then
  echo "ERROR: MLC CPU wheels exist only for x86_64 Linux. Use a PC, WSL2 or Colab." >&2
  exit 1
fi

mkdir -p "$WORK" "$DIST"
cd "$WORK"

# 1. Isolated Python env with the MLC CPU build (conversion needs no GPU).
if [ ! -x venv/bin/python ]; then
  python3 -m venv venv
fi
venv/bin/pip install -q --upgrade pip
venv/bin/pip install -q --pre -f https://mlc.ai/wheels mlc-llm-nightly-cpu mlc-ai-nightly-cpu
venv/bin/pip install -q psutil fastapi uvicorn shortuuid requests tqdm safetensors huggingface_hub
venv/bin/pip install -q torch --index-url https://download.pytorch.org/whl/cpu

# 2. Source (abliterated, bf16 safetensors) and reference (the exact MLC build the APK expects).
venv/bin/hf download "$SRC_REPO" --local-dir src \
  --include "*.json" --include "*.safetensors"
venv/bin/hf download "$REF_REPO" --local-dir ref \
  --include "mlc-chat-config.json" --include "ndarray-cache.json" \
  --include "tokenizer.json" --include "tokenizer_config.json"

# Guard: the compiled lib hard-codes these dimensions. A different base model
# would load and then produce garbage or crash in a kernel.
venv/bin/python - "$WORK/src/config.json" <<'PY'
import json, sys
c = json.load(open(sys.argv[1]))
want = dict(hidden_size=3072, intermediate_size=8192, num_hidden_layers=28,
            num_attention_heads=24, num_key_value_heads=8, vocab_size=128256)
bad = {k: (c.get(k), v) for k, v in want.items() if c.get(k) != v}
if bad:
    sys.exit(f"ERROR: source is not Llama 3.2 3B shaped: {bad}")
print("source dims OK (Llama 3.2 3B)")
PY

# 3. Quantize. Output layout is params_shard_*.bin + tensor-cache.json.
rm -rf conv
venv/bin/python -m mlc_llm convert_weight src \
  --quantization "$QUANT" --model-type llama --device cpu -o conv 2>&1 | grep -v "Warning: Cannot parse target"

# 4. Assemble the on-device folder.
OUT="$DIST/$MODEL_ID"
rm -rf "$OUT"
mkdir -p "$OUT"
cp conv/params_shard_*.bin "$OUT/"
# The Sep 2024 APK reads ndarray-cache.json; current builds read tensor-cache.json.
# Same format, so ship both names.
cp conv/tensor-cache.json "$OUT/ndarray-cache.json"
cp conv/tensor-cache.json "$OUT/tensor-cache.json"
# Chat config and tokenizer come from the reference build so that
# context_window_size / prefill_chunk_size / conv_template match what the
# compiled lib was built and tested with. Abliteration does not touch the vocab.
cp ref/mlc-chat-config.json ref/tokenizer.json ref/tokenizer_config.json "$OUT/"

# 5. Fail closed if the parameter layout differs from the reference.
venv/bin/python -I "$HERE/verify_layout.py" ref/ndarray-cache.json "$OUT"

du -sh "$OUT"
echo
echo "Done: $OUT"
echo "Next: push to the phone with device/install_mlc_model.sh $OUT"
