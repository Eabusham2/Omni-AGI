"""Broadcast existing scalar/control authority, never average private weights.

Native projections remain uint8 and are handled by packed_collective. These
are only already-existing residual Parameters, optimizer controls, retention
state, and source-free counters; no FP projection master is introduced.
"""

import torch
import torch.distributed as dist


TRANSFER_BYTES = 4 * 1024 * 1024


def synchronize_control_state(controller, module, optimizer, brain):
    parameters = dict(module.named_parameters())
    context = controller.context

    def broadcast_tensor(value):
        if not value.is_contiguous():
            raise ValueError("native residual control must have a contiguous owner")
        flat = value.detach().reshape(-1).view(torch.uint8)
        device = context.device if context.backend == "nccl" else torch.device("cpu")
        for offset in range(0, flat.numel(), TRANSFER_BYTES):
            amount = min(TRANSFER_BYTES, flat.numel() - offset)
            def prepare():
                controller.reserve(ram_bytes=amount * 2)
                return flat[offset:offset + amount].to(device).contiguous() if context.is_rank_zero else torch.empty(amount, dtype=torch.uint8, device=device)
            block = controller._phase(prepare)
            if context.distributed:
                dist.broadcast(block, src=0)
            controller._phase(lambda: flat[offset:offset + amount].copy_(block.to(flat.device)))

    def schema(value, parameter_device=None):
        if isinstance(value, torch.Tensor):
            if value.is_floating_point() or value.is_complex():
                flat = value.detach().reshape(-1)
                step = max(1, TRANSFER_BYTES // value.element_size())
                for offset in range(0, flat.numel(), step):
                    controller.reserve(ram_bytes=min(step, flat.numel() - offset))
                    if not bool(torch.isfinite(flat[offset:offset + step]).all()):
                        raise RuntimeError("canonical residual control is non-finite")
            return {"tensor": {"shape": list(value.shape), "dtype": str(value.dtype).replace("torch.", ""),
                "device": "cpu" if value.device.type == "cpu" else "parameter"}}
        if isinstance(value, dict):
            return {"mapping": [[key, schema(child, parameter_device)] for key, child in sorted(value.items())]}
        if isinstance(value, (list, tuple)):
            return {"sequence": [schema(child, parameter_device) for child in value], "tuple": isinstance(value, tuple)}
        if value is None or isinstance(value, (str, bool, int, float)):
            return {"scalar": value}
        raise ValueError("optimizer residual state contains an unsupported owner")

    def install(spec, local, parameter_device=None):
        if "tensor" in spec:
            declaration = spec["tensor"]
            dtype = getattr(torch, declaration["dtype"])
            device = torch.device("cpu") if declaration["device"] == "cpu" else parameter_device
            if device is None:
                raise ValueError("residual control tensor has no native device owner")
            shape = tuple(declaration["shape"])
            def prepare():
                nonlocal local
                if not isinstance(local, torch.Tensor) or tuple(local.shape) != shape or local.dtype != dtype or local.device != device:
                    byte_count = torch.empty((), dtype=dtype).element_size()
                    for width in shape: byte_count *= width
                    controller.reserve(ram_bytes=byte_count)
                    local = torch.empty(shape, dtype=dtype, device=device)
                return local
            local = controller._phase(prepare)
            broadcast_tensor(local)
            return local
        if "mapping" in spec:
            local = local if isinstance(local, dict) else {}
            return {key: install(child, local.get(key), parameter_device) for key, child in spec["mapping"]}
        if "sequence" in spec:
            values = [install(child, local[index] if isinstance(local, (list, tuple)) and index < len(local) else None, parameter_device)
                for index, child in enumerate(spec["sequence"])]
            return tuple(values) if spec["tuple"] else values
        return spec["scalar"]

    def declaration():
        return {"parameters": [[name, list(value.shape), str(value.dtype)] for name, value in sorted(parameters.items())],
            "optimizer": [[name, schema(optimizer.state.get(value, {}), value.device)] for name, value in sorted(parameters.items())],
            "anchors": schema(brain.slow_anchors), "importance": schema(brain.slow_importance),
            "counters": schema(brain.counters)}
    proposed = controller._phase(lambda: declaration() if context.is_rank_zero else None)
    spec = controller._objects(proposed)[0]
    expected = [[name, list(value.shape), str(value.dtype)] for name, value in sorted(parameters.items())]
    controller._phase(lambda: None if expected == spec["parameters"] else (_ for _ in ()).throw(ValueError("residual native parameter topology differs across ranks")))
    with torch.no_grad():
        for name, value in sorted(parameters.items()):
            broadcast_tensor(value)
        for name, state_spec in spec["optimizer"]:
            value = parameters[name]
            optimizer.state[value] = install(state_spec, optimizer.state.get(value), value.device)
        brain.slow_anchors = install(spec["anchors"], brain.slow_anchors)
        brain.slow_importance = install(spec["importance"], brain.slow_importance)
        brain.counters = install(spec["counters"], brain.counters)
