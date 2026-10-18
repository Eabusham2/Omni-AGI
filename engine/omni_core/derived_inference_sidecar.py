"""Diagnostics for derived inference copies, never neural-state authority."""
from copy import deepcopy
from typing import Mapping


def _binding(brain):
    mutable = getattr(brain, "mutable_state_manifest", None) or {}
    substrate = getattr(getattr(brain, "memory", None), "persistence_manifest", None) or {}
    connected = getattr(brain, "_connected_graph_states", None) or {}
    return {"mutableGeneration": mutable.get("activeGeneration"),
            "substrateGeneration": substrate.get("activeGeneration"),
            "connectedGraphs": connected.get("sha256")}


def record_sidecar_validation(brain, *, manifest_sha256, parameter_checksum, refresh_required, reason):
    """Record a completed explicit check; callers must verify bytes first.

    Load records only in memory. It must not rewrite paused metadata/RNG or
    publish a new neural generation merely to refresh a derived export.
    """
    status = {"format": "omni-derived-inference-sidecar-status", "formatVersion": 1,
              "status": "refresh-required" if refresh_required else "verified-committed-copy",
              "refreshRequired": bool(refresh_required), "reason": reason,
              "existingSidecarFullyVerified": manifest_sha256 is not None,
              "manifestContentSha256": manifest_sha256, "parameterChecksum": parameter_checksum,
              "binding": _binding(brain), "runtimeConsumesSidecar": False,
              "scope": "committed-generation-copy; not an in-flight weight-readiness proof",
              "explicitExportMustRepack": True}
    brain._packed_inference_sidecar_status = status
    if refresh_required:
        brain.packed_ternary_manifest = None
    return deepcopy(status)


def sidecar_status(brain):
    saved = getattr(brain, "_packed_inference_sidecar_status", None)
    if not isinstance(saved, Mapping):
        return {"format": "omni-derived-inference-sidecar-status", "formatVersion": 1,
                "status": "not-verified", "refreshRequired": True,
                "existingSidecarFullyVerified": False, "runtimeConsumesSidecar": False,
                "scope": "committed-generation-copy; not an in-flight weight-readiness proof",
                "explicitExportMustRepack": True, "binding": _binding(brain)}
    status = deepcopy(dict(saved))
    if status.get("binding") != _binding(brain):
        status.update(status="refresh-required", refreshRequired=True, reason="committed-neural-or-graph-generation-changed")
    return status


def current_sidecar_descriptor(brain):
    if sidecar_status(brain)["refreshRequired"]:
        return None
    value = getattr(brain, "packed_ternary_manifest", None)
    return deepcopy(dict(value)) if isinstance(value, Mapping) else None
