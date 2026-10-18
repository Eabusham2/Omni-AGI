"""Exact supported process RNG frontier, separate from the private sampler."""
import torch


def capture_execution_rng(device):
    device = torch.device(device)
    value = {"cpu": torch.get_rng_state().clone(), "device": str(device), "accelerator": None}
    if device.type == "cuda":
        value["accelerator"] = torch.cuda.get_rng_state(device).clone()
    elif device.type == "mps" and callable(getattr(torch.mps, "get_rng_state", None)):
        value["accelerator"] = torch.mps.get_rng_state().clone()
    return value


def restore_execution_rng(value, device):
    device = torch.device(device)
    if not isinstance(value, dict) or set(value) != {"cpu", "device", "accelerator"} or value["device"] != str(device):
        raise ValueError("execution RNG has a different supported backend")
    cpu, accelerator = value["cpu"], value["accelerator"]
    if (not isinstance(cpu, torch.Tensor) or cpu.dtype != torch.uint8 or cpu.device.type != "cpu"
            or cpu.ndim != 1 or cpu.shape != torch.get_rng_state().shape):
        raise ValueError("execution CPU RNG frontier is invalid")
    if device.type in {"cuda", "mps"}:
        if not isinstance(accelerator, torch.Tensor) or accelerator.dtype != torch.uint8 or accelerator.device.type != "cpu" or accelerator.ndim != 1:
            raise ValueError("execution accelerator RNG frontier is invalid")
        expected = torch.cuda.get_rng_state(device) if device.type == "cuda" else torch.mps.get_rng_state()
        if accelerator.shape != expected.shape:
            raise ValueError("execution accelerator RNG geometry is invalid")
    elif accelerator is not None:
        raise ValueError("CPU execution contains an unrelated accelerator RNG")
    torch.set_rng_state(cpu)
    if device.type == "cuda":
        torch.cuda.set_rng_state(accelerator, device)
    elif device.type == "mps":
        torch.mps.set_rng_state(accelerator)
