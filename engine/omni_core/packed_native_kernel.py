"""Native arithmetic directly over packed learned signs and strength bytes.

CPU loading never compiles or downloads code. Release workers bundle the C++
library. Metal uses the OS shader compiler once per process and caches it.
"""
from __future__ import annotations

import ctypes
from functools import lru_cache
import hashlib
import json
from pathlib import Path
import sys

import torch


@lru_cache(maxsize=1)
def _library():
    directory = Path(__file__).with_name("native")
    suffix = ".dll" if sys.platform == "win32" else ".dylib" if sys.platform == "darwin" else ".so"
    path = directory / ("omni_packed_kernel" + suffix)
    if not path.is_file():
        return None
    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("abi") != 1 or manifest.get("file") != path.name or manifest.get("sha256") != hashlib.sha256(path.read_bytes()).hexdigest():
        raise RuntimeError("Native packed kernel integrity verification failed")
    source = Path(__file__).with_name("kernels") / "weighted_ternary.cpp"
    if source.is_file() and manifest.get("sourceSha256") != hashlib.sha256(source.read_bytes()).hexdigest():
        raise RuntimeError("Native packed CPU kernel must be rebuilt for the current source")
    library = ctypes.CDLL(str(path))
    library.omni_packed_kernel_abi.restype = ctypes.c_int
    if library.omni_packed_kernel_abi() != 1:
        raise RuntimeError("Native packed kernel ABI is incompatible")
    for name in ("omni_weighted_ternary_forward_i8", "omni_weighted_ternary_input_gradient_f32",
                 "omni_weighted_ternary_forward_f32", "omni_weighted_ternary_forward_f64"):
        function = getattr(library, name)
        function.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
            ctypes.c_int64, ctypes.c_int64, ctypes.c_int64, ctypes.c_void_p]
        function.restype = ctypes.c_int
    return library


def native_packed_kernel_available():
    return _library() is not None


@lru_cache(maxsize=1)
def _metal_library():
    if not torch.backends.mps.is_available() or not hasattr(torch.mps, "compile_shader"):
        return None
    return torch.mps.compile_shader(Path(__file__).with_name("kernels").joinpath("weighted_ternary.metal").read_text(encoding="utf-8"))


def _check(activity, signs, states, width, rows, *, transpose=False):
    if type(width) is not int or type(rows) is not int or width < 0 or rows < 0 or activity.ndim != 2 or activity.shape[1] != (rows if transpose else width):
        raise ValueError("Native packed activity shape is invalid")
    if signs.dtype != torch.uint8 or tuple(signs.shape) != (rows, (width + 3) // 4):
        raise ValueError("Native packed sign shape or type is invalid")
    if states is not None and (states.dtype != torch.uint8 or tuple(states.shape) != (rows, width)):
        raise ValueError("Native packed connection state is invalid")
    for tensor in (activity, signs, states):
        if tensor is not None and (tensor.device != activity.device or not tensor.is_contiguous()):
            raise ValueError("Native kernel requires contiguous buffers on one device")
    if activity.device.type not in {"cpu", "mps"}:
        raise NotImplementedError("Native packed kernel device is not available")


def _metal_arguments(activity, signs, states, width, rows, validate_reserved):
    # Dispatch tiling belongs to the caller. This backend cannot index a single
    # >32-bit tensor; it must fall back to admitted CPU/accelerator tiles.
    if max(activity.numel(), signs.numel(), rows * width, activity.shape[0] * max(rows, width)) > 0xffffffff:
        raise NotImplementedError("Metal packed invocation needs smaller indexing tiles")
    if validate_reserved:
        for shift in (0, 2, 4, 6):
            if bool(((signs >> shift) & 3).eq(3).any().item()):
                raise ValueError("Native packed sign contains a reserved code")
    shape = torch.tensor([value if value < 2**31 else value - 2**32 for value in
        (width, rows, int(states is not None))], dtype=torch.int32, device=activity.device)
    return states if states is not None else signs, shape


def _finish(code):
    if code == 1:
        raise ValueError("Native packed sign contains a reserved code")
    if code:
        raise ValueError("Native packed kernel rejected its dimensions")


def packed_forward_i8(activity, signs, states, width, rows, *, validate_reserved=True):
    _check(activity, signs, states, width, rows)
    if activity.dtype != torch.int8:
        raise ValueError("Native packed forward requires int8 activations")
    if activity.device.type == "mps":
        library = _metal_library()
        if library is None:
            raise NotImplementedError("Native packed Metal kernel is unavailable")
        state_buffer, shape = _metal_arguments(activity, signs, states, width, rows, validate_reserved)
        output = torch.empty((activity.shape[0], rows), dtype=torch.int64, device=activity.device)
        if output.numel():
            library.packed_forward_i8(output, activity, signs, state_buffer, shape,
                threads=output.numel() * 128, group_size=128)
        return output
    library = _library()
    if library is None:
        raise RuntimeError("Native packed CPU kernel is not built")
    output = torch.empty((activity.shape[0], rows), dtype=torch.int64, device="cpu")
    _finish(library.omni_weighted_ternary_forward_i8(activity.data_ptr(), signs.data_ptr(),
        states.data_ptr() if states is not None else None,
        activity.shape[0], width, rows, output.data_ptr()))
    return output


def packed_input_gradient_f32(gradient, signs, states, width, rows, *, validate_reserved=True):
    _check(gradient, signs, states, width, rows, transpose=True)
    if gradient.dtype != torch.float32:
        raise ValueError("Native packed input gradient requires float32 activity")
    if gradient.device.type == "mps":
        library = _metal_library()
        if library is None:
            raise NotImplementedError("Native packed Metal kernel is unavailable")
        state_buffer, shape = _metal_arguments(gradient, signs, states, width, rows, validate_reserved)
        output = torch.empty((gradient.shape[0], width), dtype=torch.float32, device=gradient.device)
        if output.numel():
            library.packed_input_gradient_f32(output, gradient, signs, state_buffer, shape, threads=output.numel())
        return output
    library = _library()
    if library is None:
        raise RuntimeError("Native packed CPU kernel is not built")
    output = torch.empty((gradient.shape[0], width), dtype=torch.float32, device="cpu")
    _finish(library.omni_weighted_ternary_input_gradient_f32(gradient.data_ptr(), signs.data_ptr(),
        states.data_ptr() if states is not None else None,
        gradient.shape[0], width, rows, output.data_ptr()))
    return output


def packed_forward_float(activity, signs, states, width, rows, *, validate_reserved=True):
    """Direct packed weighted forward without quantizing floating activity.

    Float64 stays float64 on CPU. Half/bfloat16 values promote exactly to
    float32 for arithmetic and return in their original dtype; no learned
    signs/gains are expanded into a floating matrix. The router's existing
    CPU-backed tile traversal decides placement and admission, not this ABI.
    """
    _check(activity, signs, states, width, rows)
    if activity.dtype not in {torch.float16, torch.bfloat16, torch.float32, torch.float64}:
        raise ValueError("Native packed floating forward requires floating activity")
    if not bool(torch.isfinite(activity).all()):
        raise ValueError("Native packed floating activity must be finite")
    original_dtype = activity.dtype
    compute_dtype = torch.float64 if original_dtype == torch.float64 else torch.float32
    values = activity.to(dtype=compute_dtype)
    if activity.device.type == "mps":
        if compute_dtype != torch.float32:
            raise NotImplementedError("Metal packed floating forward does not support float64")
        library = _metal_library()
        if library is None:
            raise NotImplementedError("Native packed Metal kernel is unavailable")
        state_buffer, shape = _metal_arguments(values, signs, states, width, rows, validate_reserved)
        output = torch.empty((values.shape[0], rows), dtype=torch.float32, device=values.device)
        if output.numel():
            library.packed_forward_f32(output, values, signs, state_buffer, shape, threads=output.numel())
        return output.to(dtype=original_dtype)
    library = _library()
    if library is None:
        raise RuntimeError("Native packed CPU kernel is not built")
    output = torch.empty((values.shape[0], rows), dtype=compute_dtype, device="cpu")
    function = library.omni_weighted_ternary_forward_f64 if compute_dtype == torch.float64 else library.omni_weighted_ternary_forward_f32
    _finish(function(values.data_ptr(), signs.data_ptr(), states.data_ptr() if states is not None else None,
                     values.shape[0], width, rows, output.data_ptr()))
    return output.to(dtype=original_dtype)
