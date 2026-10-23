# ChatLLM2026: uncensored local LLM on a Galaxy phone with MLC Chat

Goal: run a quantized, refusal-removed ("abliterated") LLM fully offline on the
phone, using the MLC LLM Android engine, driven from Termux plus ADB.

## Device facts (check these first)

| Item | Value | Why it matters |
|---|---|---|
| SM-S938U | Galaxy **S25 Ultra** (US), not S22 Ultra (that is SM-S908U) | Picks SoC and RAM below |
| SoC | Snapdragon 8 Elite for Galaxy, Adreno 830 GPU | MLC uses OpenCL on Adreno |
| RAM | 12 GB (16 GB on 1 TB SKU) | Bounds model size: about 4 to 5 GB of GPU memory is safe |
| OS | Android 17 / One UI 9 (as stated) | Scoped storage rules for `Android/data` |

If the phone is really an S22 Ultra (SM-S908U, 8 Gen 1, 8 or 12 GB), everything
below still applies; tokens/s will be roughly half.

## Terms (used in one sense only)

* **Model library (model_lib)**: GPU kernels compiled by TVM for one exact
  architecture, dimension set and quantization, statically linked into the APK
  (`libtvm4j_runtime_packed.so`). It cannot be added after install.
* **Weights**: the quantized tensors (`params_shard_*.bin` plus the
  `ndarray-cache.json` index). These are plain files and can be swapped.
* **Abliterated**: a finetune where the refusal direction was projected out of
  the weights. Architecture and dimensions are unchanged from the base model.

## The constraint that decides the design

Facts verified against the prebuilt APK
(`binary-mlc-llm-libs` release `Android-09262024`, the latest one published):

1. It ships only five model libraries:
   `phi3_q4f16_0`, `qwen2_q4f16_1` (Qwen2.5 1.5B), `gemma2_q4f16_1`,
   `llama_q4f16_0_2d32572d8a4ab2af20a1f587ef6c8c63` (Llama 3.2 3B),
   `mistral_q4f16_1` (Mistral 7B v0.3).
2. The app reads `Android/data/ai.mlc.mlcchat/files/mlc-app-config.json`
   **instead of** its bundled config when that file exists, and takes
   `model_lib` from each record, not from the weights.
3. That APK looks up the weight index as `ndarray-cache.json` (current
   upstream renamed it `tensor-cache.json`).

Consequence: any abliterated model that has **the same architecture, the same
dimensions and the same quantization** as a built-in model can run on the
stock APK by shipping new weights and borrowing the built-in `model_lib`.
Everything else (Qwen3, Qwen3.5, Gemma 3, newer Llama) needs a rebuilt APK,
which means Android NDK, Rust and a TVM compile on a PC.

Chosen model: **`huihui-ai/Llama-3.2-3B-Instruct-abliterated`**, converted to
`q4f16_0`, run with the built-in Llama 3.2 3B library. It is the only
well-known abliterated model whose shape matches a library in the APK.

The existing community MLC uploads (Qwen3.5 2B/4B abliterated, Lexi 8B, etc.)
were checked and rejected: none of their libraries are in the prebuilt APK, so
they fail with "Model lib ... is not supported".

## Pipeline

```
[PC, x86_64 Linux / WSL2 / Colab]                [Phone]
host/convert_abliterated_mlc.sh  ── dist/ ──>  device/install_mlc_model.sh
  download bf16 abliterated weights               (Termux + adb to 127.0.0.1,
  quantize to q4f16_0 (CPU)                        or PC + USB adb)
  copy chat config + tokenizer from               install APK if absent
    mlc-ai reference build                        push weights to Android/data
  verify_layout.py: every param name,             add record to mlc-app-config.json
    shape, dtype == reference                       borrowing llama_q4f16_0 lib
```

Conversion cannot run on the phone: MLC publishes only x86_64 Linux wheels for
`mlc_llm`, so Termux (aarch64, bionic libc) cannot import it.

### Step 1: convert (PC)

```
bash host/convert_abliterated_mlc.sh
```

Produces `dist/Llama-3.2-3B-Instruct-abliterated-q4f16_0-MLC/` (about 1.7 GB)
and exits non-zero if the layout check fails.

### Step 2: move the folder to the phone

Either keep the phone on USB and run step 3 on the PC, or copy the folder to
`/sdcard/Download/` and run step 3 in Termux.

### Step 3: install (Termux on the phone)

One-time Termux setup:

```
pkg install android-tools python unzip curl git
termux-setup-storage
```

Wireless debugging to the phone itself (needs split screen: Settings on one
side, Termux on the other):

1. Settings > Developer options > Wireless debugging: on.
2. Tap "Pair device with pairing code"; note the pairing port and code.
3. `adb pair 127.0.0.1:<pairing-port> <code>`
4. `adb connect 127.0.0.1:<port shown on the Wireless debugging main screen>`
5. `adb devices` must list `127.0.0.1:<port>  device`.

Then:

```
bash device/install_mlc_model.sh ~/storage/downloads/Llama-3.2-3B-Instruct-abliterated-q4f16_0-MLC
```

Open MLC Chat; the new entry appears in the model list next to the built-ins.

## Fallback: fully on-device, newer model (no PC)

```
bash termux/run_llamacpp_uncensored.sh
```

Runs `Huihui-Qwen3-4B-Instruct-2507-abliterated` (GGUF Q4_K_M, 2.5 GB) with
llama.cpp on the CPU and serves a chat UI at `http://127.0.0.1:8080`. This is
not MLC, but it is a stronger model than Llama 3.2 3B and needs nothing beyond
Termux.

## Failure modes

| Symptom | Layer | Cause | Action |
|---|---|---|---|
| `verify_layout.py` FAIL | Model | Source is not Llama 3.2 3B shaped, or MLC changed param naming | Do not ship; the lib would read wrong tensors |
| "Model lib ... is not supported" | Implementation | Record's `model_lib` not in APK | Rerun installer; check `LIB_PREFIX` |
| `adb shell cannot write Android/data` | OS policy | Android 17 tightened shell access to app-private external storage | Use the llama.cpp fallback, or rebuild the APK with `bundle_weight` |
| Model downloads instead of opening | Implementation | A file is missing on device; the app tries `model_url` | The URL is deliberately dead, so it fails visibly; rerun the push |
| App crashes on load, logcat shows OpenCL errors | Implementation | Adreno 830 driver vs a 2024 TVM build | Check `adb logcat -s TVM_RUNTIME AndroidRuntime`; fall back to llama.cpp |
| Install refused for low target SDK | OS policy | Newer Android blocks very old `targetSdk` APKs | Installer retries with `--bypass-low-target-sdk-block` |
| Model still refuses some prompts | Model | Abliteration removes most, not all, refusal behavior | Edit `conv_template.system_message` in the model's `mlc-chat-config.json` before pushing; try the Qwen3 fallback |
| OOM / app killed mid-chat | Device | Context too long for 12 GB with One UI overhead | Restart chat; config caps context at the reference value |

## Notes on the "uncensored" part

Abliteration changes behavior, not capability or accuracy. The model will
answer more, including confidently wrong answers; treat medical, legal and
financial output as unreviewed. The Llama 3.2 base is under the Llama 3.2
Community License and its acceptable use policy; the Qwen3 base is Apache 2.0.
