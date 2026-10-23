#!/usr/bin/env python3
"""Run `mlc_llm convert_weight` with group quantization done in numpy.

Why: on current MLC nightlies, relax.build() of the CPU quantize kernel
segfaults, so `convert_weight --device cpu` dies on the first quantized tensor.
Everything else in the pipeline (parameter naming, dtype casts, sharding,
tensor-cache.json) is still MLC's own code; only
GroupQuantize.quantize_weight is replaced with a numpy port of its tensor
expression (mlc_llm/quantization/group_quantization.py, _quantize and
pack_weight). Every op runs in the model dtype (float16) to reproduce TVM's
per-op rounding.

Validated against mlc-ai/Llama-3.2-3B-Instruct-q4f16_0-MLC by quantizing the
same base weights: all q_scale and non-quantized tensors are bit-identical;
99.87% of 4-bit codes are identical and the rest differ by one level at
near-.5 ties (the reference was likely produced on GPU with fast-math
division). Reconstruction RMSE matches the reference to 5 significant digits.

Usage: same arguments as `mlc_llm convert_weight`.
"""
import runpy
import sys

import numpy as np
import tvm
from mlc_llm.quantization.group_quantization import GroupQuantize


def _np_quantize_weight(self, weight, axis=-1, output_transpose=False):
    w = weight.numpy()
    dt = np.dtype(self.model_dtype)
    if w.dtype != dt:
        w = w.astype(dt)
    axis = axis if axis >= 0 else w.ndim + axis
    k = w.shape[axis]
    gs = self.group_size
    if k % gs or k % self.num_elem_per_storage:
        raise ValueError(f"axis length {k} not divisible by group size {gs}")
    if self.quantize_dtype != "int4" or self.storage_dtype != "uint32":
        raise NotImplementedError(f"{self.quantize_dtype}/{self.storage_dtype}")

    w = np.moveaxis(w, axis, -1)
    lead = w.shape[:-1]
    ng = k // gs
    max_int = dt.type(self.max_int_value)

    grouped = w.reshape(*lead, ng, gs)
    # TVM folds "x / const" into "x * fp16(1/const)"; dividing here instead
    # changes ~1/3 of scales by one ulp.
    recip = dt.type(1.0 / self.max_int_value)
    scale = (np.abs(grouped).max(axis=-1) * recip).astype(dt)     # (..., ng)
    q = grouped / scale[..., None]                                # fp16 op
    q = (q + max_int).astype(dt)                                  # fp16 op
    q = np.floor(q.astype(np.float32) + 0.5)                      # TVM round: half away (q >= 0)
    q = np.clip(q, dt.type(0), max_int * dt.type(2)).astype(np.uint32)
    q = q.reshape(*lead, k // self.num_elem_per_storage, self.num_elem_per_storage)
    shifts = (np.arange(self.num_elem_per_storage, dtype=np.uint32) * 4)
    packed = np.bitwise_or.reduce(q << shifts, axis=-1).astype(np.uint32)

    packed = np.moveaxis(packed, -1, axis)
    scale = np.moveaxis(scale, -1, axis)
    if output_transpose:
        if packed.ndim != 2:
            raise ValueError("transpose needs 2D weight")
        packed, scale = packed.T, scale.T
    dev = tvm.cpu()
    return [
        tvm.runtime.tensor(np.ascontiguousarray(packed), dev),
        tvm.runtime.tensor(np.ascontiguousarray(scale), dev),
    ]


GroupQuantize.quantize_weight = _np_quantize_weight

if __name__ == "__main__":
    sys.argv = ["mlc_llm", "convert_weight", *sys.argv[1:]]
    runpy.run_module("mlc_llm", run_name="__main__", alter_sys=True)
