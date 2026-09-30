"""Native read-only evaluation over explicitly registered real data.

No model construction, training/backward, tool execution, generated benchmark,
or chat readiness quiz. Every registered record/window is consumed or the
evaluation fails closed. Large JSON tool records require physical admission.
"""

import json
import math
from pathlib import Path

import torch
from torch.nn import functional as F

from .architecture_migration import preserve_runtime_rng
from .registered_geometry_holdouts import load_registered_geometry_holdouts


def _rows(path, policy):
    with path.open("rb") as source:
        while True:
            begin, size = source.tell(), 0
            while True:
                piece = source.readline(65536)
                size += len(piece)
                if not piece or piece.endswith(b"\n"): break
            if not size: return
            if policy.status(estimated_ram_bytes=size * 16).get("memoryPressure"):
                raise RuntimeError("geometry JSON record decoding paused at the physical memory reserve")
            source.seek(begin); payload = source.read(size)
            if payload.strip(): yield json.loads(payload)


@torch.no_grad()
def evaluate_isolated_geometry_holdouts(brain, *, excluded_text_sha256, expected_benchmark_sha256=None,
    registered_manifest=None, cancelled=None):
    from .native_action_protocol import validate_structural_value
    from .evolution import _utf8_fingerprint
    manifest = registered_manifest or load_registered_geometry_holdouts(brain.engine_path, cancelled=cancelled)
    if expected_benchmark_sha256 is not None and manifest["benchmarkSha256"] != expected_benchmark_sha256:
        raise ValueError("native holdout benchmark changed")
    excluded = set(excluded_text_sha256)
    before = brain.parameter_checksum()
    roots = tuple(brain._trainable_modules())
    modes = [root.training for root in roots]
    metrics, peak, accelerator_peak = {}, 0, 0
    def check():
        nonlocal peak, accelerator_peak
        if cancelled is not None and cancelled(): raise InterruptedError("native geometry evaluation cancelled")
        state = brain.resource_policy.status()
        peak = max(peak, int(state.get("admissionResidentMemoryBytes", state.get("processMemoryBytes", 0)) or 0))
        accelerator_peak = max(accelerator_peak, int(state.get("acceleratorAllocatedMemoryBytes", 0) or 0))
        if state.get("memoryPressure") or state.get("diskPressure"):
            raise RuntimeError("native geometry evaluation paused at the selected resource envelope")
    def idea(text):
        if not isinstance(text, str) or not text.strip() or "\x00" in text: raise ValueError("holdout needs literal valid text")
        if _utf8_fingerprint(text)["sha256"] in excluded: raise ValueError("held-out input overlaps candidate training")
        return brain._idea_model_vector(brain.memory.vector_for_text(text)).detach().reshape(1, -1).to(brain.device)
    try:
        for root in roots: root.eval()
        with preserve_runtime_rng(brain.device):
            for category, data in manifest["categories"].items():
                total, elements, records = 0., 0, 0
                for item in data["files"]:
                    check(); path = Path(manifest["root"]) / item["path"]
                    if category in {"token", "tool"}:
                        for row in _rows(path, brain.resource_policy):
                            check()
                            if category == "token":
                                if not isinstance(row, dict) or set(row) != {"text"}: raise ValueError("token holdout must contain exact text records")
                                text = row["text"]; vector = idea(text)
                                for ids in brain.tokenizer.window_tensors(text, brain.device,
                                    max_length=min(brain.config.max_seq_len, brain._runtime_training_max_seq_len), add_bos=True, add_eos=True):
                                    check()
                                    loss = brain.decoder(ids, memory_bias=brain.idea_adapter(vector), labels=ids)["loss"]
                                    count = int(ids.shape[1]) - 1
                                    total += float(loss.item()) * count; elements += count
                            else:
                                if not isinstance(row, dict) or set(row) != {"context", "schemas", "expectedIndex", "arguments"}: raise ValueError("tool holdout schema is invalid")
                                schemas, index = row["schemas"], row["expectedIndex"]
                                if not isinstance(schemas, list) or not schemas or type(index) is not int or not 0 <= index < len(schemas): raise ValueError("tool holdout target is invalid")
                                if any(not isinstance(schema, dict) or len(schema.get("actions", [])) != 1 for schema in schemas): raise ValueError("holdout candidate schemas need exactly one typed action")
                                state = idea(row["context"])
                                head, arguments = brain.decoder.tool_route_head, brain.decoder.action_argument_head
                                features = torch.stack([head.encode(schema["id"] + " " + schema["actions"][0]) for schema in schemas]).to(brain.device)
                                logits = head.forward_internal(state, candidate_features=features)
                                selected = schemas[index]; action = selected["actions"][0]
                                structural = brain._tool_action_input_schema(selected, action)
                                if structural is None or not validate_structural_value(row["arguments"], structural): raise ValueError("heldout arguments violate the typed schema")
                                encoded = arguments.schema_features(selected["id"], action, selected)
                                if encoded is None: raise ValueError("heldout action is disabled or unavailable")
                                loss = F.cross_entropy(logits, torch.tensor([index + 1], device=brain.device)) + arguments.supervised_loss(state, encoded[None].to(brain.device), row["arguments"])
                                total += float(loss.item()); elements += 1
                            records += 1
                    else:
                        kind = item.get("kind")
                        if kind not in {"image", "audio", "video"} or not getattr(brain.config, kind + "_enabled", False): raise ValueError("registered modality pack/data is unavailable")
                        state = idea(item["conditionText"])
                        windows = [(brain._decode_image(str(path)), 1)] if kind == "image" else (
                            brain._iter_audio_windows(str(path)) if kind == "audio" else brain._iter_video_windows(str(path)))
                        consumed = 0
                        try:
                            for target, units in windows:
                                check()
                                torch.random.default_generator.manual_seed(int(item["sha256"][:8], 16) + consumed)
                                if brain.device.type == "cuda": torch.cuda.default_generators[brain.device.index or 0].manual_seed(int(item["sha256"][:8], 16) + consumed)
                                output = getattr(brain.modalities, kind)(target, state)["reconstruction"]
                                if kind == "audio": output, target = output[..., :units], target[..., :units]
                                if kind == "video": output, target = output[:, :, :units], target[:, :, :units]
                                total += float(F.mse_loss(output, target, reduction="sum").item()); elements += target.numel(); consumed += units
                        finally:
                            if hasattr(windows, "close"): windows.close()
                        if consumed < 1: raise ValueError("registered media contained no valid decoded units")
                        records += 1
                if records != data["examples"] or elements < 1 or not math.isfinite(total):
                    raise ValueError("native heldout traversal was incomplete/empty/nonfinite")
                metrics[category] = {"sourceSha256": data["sourceSha256"], "loss": total / elements,
                    "examples": records, "heldOut": True, "trainingOverlapCount": 0}
        check()
        if brain.parameter_checksum() != before: raise RuntimeError("read-only heldout evaluator mutated native parameters")
        if peak < 1: raise RuntimeError("native heldout resource measurement is unavailable")
        return {"format": "omni-native-geometry-holdouts", "formatVersion": 1, "benchmarkSha256": manifest["benchmarkSha256"],
            **metrics, "resources": {"withinSelectedEnvelope": True, "peakManagedMemoryBytes": peak, "peakAcceleratorMemoryBytes": accelerator_peak}}
    finally:
        for root, mode in zip(roots, modes): root.train(mode)
