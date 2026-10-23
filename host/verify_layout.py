#!/usr/bin/env python3
"""Check that converted MLC weights match the reference layout the APK lib was compiled for.

The prebuilt MLC Chat APK links a model library (for example
llama_q4f16_0_2d32572d8a4ab2af20a1f587ef6c8c63) whose kernels assume an exact
set of parameter names, shapes and dtypes. Swapping in a finetune is only safe
when every parameter matches. Byte offsets inside shards may differ; the
runtime resolves each parameter through the cache index, not by position.

Usage: verify_layout.py REFERENCE_CACHE_JSON CANDIDATE_MODEL_DIR
Exit code 0 means compatible, 1 means do not ship.
"""
import json
import os
import sys


def load_params(cache_path):
    with open(cache_path) as f:
        cache = json.load(f)
    params = {}
    shards = []
    for shard in cache["records"]:
        shards.append((shard["dataPath"], shard["nbytes"]))
        for rec in shard["records"]:
            params[rec["name"]] = (tuple(rec["shape"]), rec["dtype"], rec.get("format"))
    return params, shards


def main():
    if len(sys.argv) != 3:
        print(__doc__)
        return 2
    ref_cache, model_dir = sys.argv[1], sys.argv[2]
    cand_cache = os.path.join(model_dir, "ndarray-cache.json")
    if not os.path.exists(cand_cache):
        cand_cache = os.path.join(model_dir, "tensor-cache.json")

    ref, _ = load_params(ref_cache)
    cand, shards = load_params(cand_cache)

    errors = []
    for name in sorted(set(ref) - set(cand)):
        errors.append(f"missing param: {name}")
    for name in sorted(set(cand) - set(ref)):
        errors.append(f"unexpected param: {name}")
    for name in sorted(set(ref) & set(cand)):
        if ref[name] != cand[name]:
            errors.append(f"mismatch {name}: ref={ref[name]} cand={cand[name]}")

    for data_path, nbytes in shards:
        p = os.path.join(model_dir, data_path)
        if not os.path.exists(p):
            errors.append(f"shard missing on disk: {data_path}")
        elif os.path.getsize(p) != nbytes:
            errors.append(f"shard size {data_path}: index={nbytes} disk={os.path.getsize(p)}")

    if errors:
        print(f"FAIL: {len(errors)} problem(s)")
        for e in errors[:50]:
            print("  " + e)
        return 1
    print(f"OK: {len(cand)} params match reference, {len(shards)} shards present")
    return 0


if __name__ == "__main__":
    sys.exit(main())
