#!/usr/bin/env python3
"""Add a weights-only model variant to MLC Chat's mlc-app-config.json.

The app reads <external files dir>/mlc-app-config.json INSTEAD of the bundled
asset when the file exists, so the output must keep every built-in record.
The new record borrows model_lib from a built-in record whose lib was compiled
for the same architecture, dimensions and quantization.

Usage: merge_app_config.py IN_JSON OUT_JSON MODEL_ID LIB_PREFIX
"""
import json
import sys

# The app falls back to downloading any missing file from model_url. Point it
# at a URL that cannot resolve, so a partial push fails loudly instead of
# silently mixing in shards from the censored base model.
NO_DOWNLOAD_URL = "https://huggingface.co/local-only/not-hosted"


def main():
    if len(sys.argv) != 5:
        print(__doc__)
        return 2
    src, dst, model_id, lib_prefix = sys.argv[1:]
    with open(src) as f:
        cfg = json.load(f)
    records = cfg.get("model_list", [])

    donors = [r for r in records if r.get("model_lib", "").startswith(lib_prefix)]
    if not donors:
        libs = ", ".join(r.get("model_lib", "?") for r in records)
        print(f"ERROR: no built-in model_lib starts with {lib_prefix}. APK has: {libs}")
        return 1
    donor = donors[0]

    records = [r for r in records if r.get("model_id") != model_id]
    records.append({
        "model_id": model_id,
        "model_lib": donor["model_lib"],
        "model_url": NO_DOWNLOAD_URL,
        "estimated_vram_bytes": donor.get("estimated_vram_bytes", 4679979417),
    })
    cfg["model_list"] = records

    with open(dst, "w") as f:
        json.dump(cfg, f, indent=2)
    print(f"added {model_id} using lib {donor['model_lib']} (donor {donor['model_id']})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
