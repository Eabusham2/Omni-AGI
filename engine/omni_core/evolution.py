"""Transactional neural evolution candidates for OmniCortex.

Candidates never train against the live ``AdaptiveBrain`` object.  A complete
safe-tensor checkpoint is copied into an isolated model directory, trained
there, and evaluated against anchors stored outside that writable overlay.
Promotion is a recoverable three-file transaction guarded by both parameter
and complete neural-state checksums.
"""

import hashlib
import json
import math
import os
import shutil
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import torch
from torch.nn import functional as F

from .offload import copy_mutable_state_snapshot
from .architecture_migration import (
    assert_architecture_quiescent,
    file_tensor_inventory,
    growth_dimensions,
    isolated_checkpoint_resident_bytes,
    normalize_architecture_change,
    preserve_runtime_rng,
    reseal_native_descriptor,
    verify_preserved_tensor_prefixes,
)
from .bounded_tensor_io import BoundedTensorFile
from .evolution_anchors import RetentionAnchorFile, save_retention_anchors
from .persistence import (
    atomic_write_json,
    copy_substrate_snapshot,
    read_json,
    snapshot_files,
    snapshot_required_bytes,
)


EVALUATOR_VERSION = "omni-neural-evaluator-1.0"
CAPABILITY_PROBES: Tuple[str, ...] = (
    "A cause can precede an effect while evidence can revise a hypothesis.",
    "A tool action has typed arguments, a visible result, and a recoverable error.",
    "An image, a sound, and a moving scene can share one internal idea.",
    "An improvement is accepted only after capability and retention checks.",
)
UNSAFE_TENSOR_SUFFIXES = {
    ".ckpt",
    ".joblib",
    ".pkl",
    ".pickle",
    ".pt",
    ".pth",
}


def _iso_now() -> str:
    import time

    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            block = handle.read(1024 * 1024)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def _json_sha256(value: Any) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _benchmark_payload(manifest: Mapping[str, Any]) -> Dict[str, Any]:
    payload = {
        "evaluator": EVALUATOR_VERSION,
        "capabilityProbes": list(CAPABILITY_PROBES),
        "parentStateChecksum": manifest["parentStateChecksum"],
        "anchorTensorSha256": manifest["anchorTensorSha256"],
        "architecture": manifest["architecture"],
        "architectureMutation": manifest.get("architectureMutation"),
    }
    # Old immutable baselines keep their exact original hash; new candidates
    # additionally bind complete metadata/context/lineage against tampering.
    if "parentMetadataSha256" in manifest:
        payload["parentMetadataSha256"] = manifest["parentMetadataSha256"]
    return payload


def _validate_json(value: Any, label: str) -> Any:
    try:
        serialized = json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
    except (TypeError, ValueError) as error:
        raise ValueError("%s must contain JSON-safe finite values" % label) from error
    return json.loads(serialized)


def _bundle_checksum(engine_path: Path) -> str:
    digest = hashlib.sha256()
    for filename, prefix in (("core.safetensors", "core:"), ("plasticity.safetensors", "plasticity:")):
        reader = BoundedTensorFile(engine_path / filename)
        for key, spec in reader.specs.items():
            digest.update((prefix + key).encode("utf-8"))
            digest.update(str(spec.shape).encode("ascii"))
            digest.update(str(spec.dtype).encode("ascii"))
            for chunk in reader.chunks(key):
                digest.update(chunk.reshape(-1).view(torch.uint8).numpy().tobytes())
    digest.update(b"substrate:")
    digest.update(_substrate_content_checksum(engine_path).encode("ascii"))
    return digest.hexdigest()


def _substrate_content_checksum(engine_path: Path) -> str:
    metadata = read_json(engine_path / "brain.json")
    substrate = metadata.get("substrate", {})
    if not isinstance(substrate, Mapping):
        return ""
    persistence = substrate.get("persistence", {})
    if not isinstance(persistence, Mapping):
        # Early internal stable-v1 checkpoints are represented in the
        # plasticity tensor checksum above.
        return ""
    checksum = str(persistence.get("contentSha256", ""))
    if checksum and (
        len(checksum) != 64
        or any(character not in "0123456789abcdef" for character in checksum)
    ):
        raise ValueError("substrate content checksum is invalid")
    return checksum


def _architecture_signature(engine_path: Path) -> Dict[str, Any]:
    metadata = read_json(engine_path / "brain.json")
    tensors = {
        prefix + name: spec
        for filename, prefix in (("core.safetensors", "core:"), ("plasticity.safetensors", "plasticity:"))
        for name, spec in file_tensor_inventory(engine_path / filename).items()
    }
    return {
        "fixedTensors": {
            name: {
                "shape": tensor["shape"],
                "dtype": tensor["dtype"],
            }
            for name, tensor in sorted(tensors.items())
            # Expert inventories are counted separately. All decoder blocks
            # and router controls have exact declared migration geometries.
            if (
                (
                    name.startswith("core:")
                    and not name.startswith("core:decoder.experts.")
                    and not name.startswith(
                        "core:decoder.expert_prototypes."
                    )
                )
                or name.startswith("plasticity:router.")
            )
        },
        "expertCount": int(metadata.get("expert_count", 0)),
        "expertFormat": "ternary-residual-expert-v1",
        "layers": int(metadata.get("config", {}).get("n_layers", 0)),
        "routerNeurons": int(metadata.get("config", {}).get("router_neurons", 0)),
        "expertRoutingBaselineCount": int(metadata.get("config", {}).get("expert_routing_baseline_count", -1)),
        "headGeometry": {
            key: metadata.get("config", {}).get(key)
            for key in ("d_model", "d_ff", "n_heads", "idea_dim", "vsa_dim")
        },
        "protectedGeometry": {
            key: metadata.get("config", {}).get(key)
            for key in (
                "vocab_size", "dropout", "modality_channels", "image_size",
                "audio_samples", "video_frames", "liquid_mode",
                "working_memory_slots", "ternary_weights", "spiking_dynamics",
                "stdp_plasticity", "liquid_dynamics", "vector_symbolic_memory",
            )
        },
    }


def _normalize_architecture_change(
    value: Optional[Mapping[str, Any]],
) -> Optional[Dict[str, Any]]:
    return normalize_architecture_change(value)


def _architecture_compatible(
    baseline: Mapping[str, Any],
    candidate: Mapping[str, Any],
    mutation: Optional[Mapping[str, Any]],
) -> bool:
    if baseline.get("headGeometry") != candidate.get("headGeometry"):
        return False
    if baseline.get("protectedGeometry") != candidate.get("protectedGeometry"):
        return False
    if (
        baseline.get("expertFormat") != "ternary-residual-expert-v1"
        or candidate.get("expertFormat") != "ternary-residual-expert-v1"
    ):
        return False
    before = int(baseline.get("expertCount", -1))
    after = int(candidate.get("expertCount", -1))
    old_routing = int(baseline.get("expertRoutingBaselineCount", -1))
    new_routing = int(candidate.get("expertRoutingBaselineCount", -1))
    if mutation is None:
        return after == before and old_routing == new_routing and baseline.get("fixedTensors") == candidate.get("fixedTensors")
    kind = mutation.get("mutation")
    old, new = dict(baseline.get("fixedTensors", {})), dict(candidate.get("fixedTensors", {}))
    if kind == "grow-experts":
        expected_routing = before if old_routing < 0 else old_routing
        return old == new and new_routing == expected_routing and after == before + int(mutation.get("addExperts", 0))
    if after != before or old_routing != new_routing:
        return False
    if kind == "grow-depth":
        old_layers = int(baseline.get("layers", 0))
        count = int(mutation.get("addLayers", 0))
        template = {name[len("core:decoder.blocks.0."):]: value for name, value in old.items() if name.startswith("core:decoder.blocks.0.")}
        if not template or int(candidate.get("layers", 0)) != old_layers + count or baseline.get("routerNeurons") != candidate.get("routerNeurons"):
            return False
        expected = dict(old)
        for index in range(old_layers, old_layers + count):
            expected.update({"core:decoder.blocks.%d.%s" % (index, name): value for name, value in template.items()})
        return expected == new
    if kind in {"grow-router", "grow-regions"}:
        old_count = int(baseline.get("routerNeurons", 0))
        addition = int(mutation.get("addNeurons", 0)) if kind == "grow-router" else int(mutation.get("addRegions", 0)) * int(mutation.get("neuronsPerRegion", 0))
        new_count = old_count + addition
        if int(candidate.get("routerNeurons", 0)) != new_count or baseline.get("layers") != candidate.get("layers") or set(old) != set(new):
            return False
        for name, value in old.items():
            if not name.startswith("plasticity:router."):
                if new[name] != value:
                    return False
                continue
            if new[name]["dtype"] != value["dtype"]:
                return False
            shape = list(value["shape"])
            if name.endswith("input_projection._packed_forward_weight"):
                shape[0] = new_count
            elif name.endswith("input_projection._packed_forward_bias"):
                shape[1] = (new_count + 3) // 4
            elif name.endswith("output_projection._packed_forward_weight") or name.endswith("synapses._packed_weights"):
                shape[1] = (new_count + 3) // 4
                if name.endswith("synapses._packed_weights"):
                    shape[0] = new_count
            elif name.endswith("region_ends"):
                regions = int(mutation.get("addRegions", 1))
                if new[name]["shape"] != [shape[0] + regions]:
                    return False
                continue
            elif name.endswith(("synapses.eligibility_accumulator", "synapses.stability", "synapses.uses")):
                shape = [new_count, new_count]
            elif name.endswith(("input_projection._row_stability", "population.membrane", "population.spike_count", "synapses.pre_trace", "synapses.post_trace")):
                shape = [new_count]
            if new[name]["shape"] != shape:
                return False
        return True
    return False


def _tensor_resources(engine_path: Path) -> Dict[str, int]:
    specs = [spec for filename in ("core.safetensors", "plasticity.safetensors") for spec in BoundedTensorFile(engine_path / filename).specs.values()]
    shard_count, shard_bytes = 0, 0
    if (engine_path / "substrate").is_dir():
        for path in (engine_path / "substrate").rglob("*"):
            if path.is_file():
                shard_count += 1
                shard_bytes += int(path.stat().st_size)
    return {
        "tensorCount": len(specs),
        "elementCount": sum(spec.numel for spec in specs),
        "tensorBytes": sum(spec.byte_count for spec in specs),
        "checkpointBytes": sum(
            int((engine_path / filename).stat().st_size)
            for filename in ("core.safetensors", "plasticity.safetensors")
        )
        + shard_bytes,
        "substrateShardFiles": shard_count,
        "substrateShardBytes": shard_bytes,
    }


def _diff_checksum(baseline_path: Path, candidate_path: Path) -> Tuple[str, float]:
    digest = hashlib.sha256()
    squared_norm = 0.0
    for filename, prefix in (("core.safetensors", "core:"), ("plasticity.safetensors", "plasticity:")):
        left_reader, right_reader = BoundedTensorFile(baseline_path / filename), BoundedTensorFile(candidate_path / filename)
        for key in sorted(set(left_reader.specs).union(right_reader.specs)):
            left, right = left_reader.specs.get(key), right_reader.specs.get(key)
            digest.update((prefix + key).encode("utf-8"))
            if left is None or right is None:
                reader, tag = (right_reader, b"added") if left is None else (left_reader, b"removed")
                digest.update(tag)
                for chunk in reader.chunks(key):
                    digest.update(chunk.reshape(-1).view(torch.uint8).numpy().tobytes())
                    squared_norm += float(chunk.float().pow(2).sum().item())
            elif left.shape != right.shape or left.dtype != right.dtype:
                digest.update(b"reshaped")
                digest.update(str(left.shape).encode("ascii"))
                digest.update(str(right.shape).encode("ascii"))
                for chunk in right_reader.chunks(key):
                    digest.update(chunk.reshape(-1).view(torch.uint8).numpy().tobytes())
                    squared_norm += float(chunk.float().pow(2).sum().item())
                for chunk in left_reader.chunks(key):
                    squared_norm += float(chunk.float().pow(2).sum().item())
            else:
                for left_chunk, right_chunk in zip(left_reader.chunks(key), right_reader.chunks(key)):
                    delta = right_chunk.float() - left_chunk.float()
                    digest.update(delta.numpy().tobytes())
                    squared_norm += float(delta.pow(2).sum().item())
    baseline_substrate = _substrate_content_checksum(baseline_path)
    candidate_substrate = _substrate_content_checksum(candidate_path)
    digest.update(b"substrate:")
    digest.update(baseline_substrate.encode("ascii"))
    digest.update(candidate_substrate.encode("ascii"))
    if baseline_substrate != candidate_substrate:
        squared_norm += 1.0
    return digest.hexdigest(), math.sqrt(max(0.0, squared_norm))


def _copy_atomic(source: Path, destination: Path) -> None:
    temporary = destination.with_name(destination.name + ".evolution.tmp")
    shutil.copy2(str(source), str(temporary))
    os.replace(str(temporary), str(destination))


class NeuralEvolutionManager:
    """Create, evaluate, promote, reject, and roll back neural candidates."""

    def __init__(self, brain: Any):
        self.brain = brain
        self.engine_path = brain.engine_path
        self.candidates_path = self.engine_path / "candidates"
        self.baselines_path = self.engine_path / "evolution-baselines"

    def _candidate_dir(self, candidate_id: str) -> Path:
        candidate_id = str(candidate_id)
        if (
            not candidate_id
            or candidate_id in {".", ".."}
            or any(character not in "0123456789abcdef" for character in candidate_id)
        ):
            raise ValueError("candidateId is invalid")
        path = (self.candidates_path / candidate_id).resolve()
        try:
            path.relative_to(self.candidates_path.resolve())
        except ValueError as error:
            raise ValueError("candidateId escapes the candidates directory") from error
        if not path.is_dir():
            raise ValueError("neural candidate does not exist")
        return path

    def _record(self, candidate_id: str) -> Tuple[Path, Dict[str, Any]]:
        directory = self._candidate_dir(candidate_id)
        record = read_json(directory / "candidate.json")
        if record.get("kind") != "neural-evolution":
            raise ValueError("candidate is not a neural evolution candidate")
        return directory, record

    def _model_path(self, candidate_dir: Path) -> Path:
        model = candidate_dir / "model"
        engine = model / "engine"
        for filename in ("brain.json", "core.safetensors", "plasticity.safetensors"):
            if not (engine / filename).is_file():
                raise ValueError("candidate model is missing %s" % filename)
        return model

    def _admit_isolated_load(self, engine_path: Path) -> int:
        resident = isolated_checkpoint_resident_bytes(engine_path)
        reading = self.brain.resource_policy.status(estimated_ram_bytes=resident)
        if reading.get("memoryPressure"):
            self.brain.core_pager.cool_to_budget(max(0, int(self.brain.core_pager.status()["cpuHeapBytes"]) - resident))
            reading = self.brain.resource_policy.status(estimated_ram_bytes=resident)
        if reading.get("memoryPressure"):
            raise RuntimeError("isolated evolution checkpoint load paused at the shared control-state reserve")
        return resident

    @staticmethod
    def _objective_loss(brain: Any, texts: Sequence[str]) -> float:
        if not texts:
            return 0.0
        losses = [
            brain._evaluate_experience(text, brain.memory.vector_for_text(text))
            for text in texts
        ]
        return sum(losses) / float(len(losses))

    @staticmethod
    def _latent_loss(brain: Any, anchors: RetentionAnchorFile) -> float:
        if anchors.numel() == 0:
            return 0.0
        width = int(anchors.shape[1])
        pager = getattr(brain.decoder, "working_attention_pager", None)
        compute = int(getattr(pager, "device_tile_budget_bytes", 1_048_576))
        # Include hidden/normalization/output and transfer lifetimes, not just
        # the input row. Admission never cuts the corpus to fit a reservation.
        batch_rows = min(32, compute // max(1, width * 4 * 32))
        if batch_rows < 1:
            raise RuntimeError("retention evaluation paused: one idea row exceeds compute reservation")
        brain.idea_adapter.eval()
        total, elements = 0.0, 0
        with torch.no_grad():
            for batch in anchors.batches(max_rows=batch_rows, byte_budget=max(width * 4, min(1_048_576, compute // 32)), policy=getattr(brain, "resource_policy", None)):
                values = batch.to(brain.device)
                total += float(F.mse_loss(brain.idea_adapter(values), values, reduction="sum").item())
                elements += values.numel()
        return total / elements

    @classmethod
    def _capability_loss(cls, brain: Any) -> float:
        return cls._objective_loss(brain, CAPABILITY_PROBES)

    @staticmethod
    def _latent_anchors(brain: Any, texts: Sequence[str], path: Path, candidate_id: str) -> RetentionAnchorFile:
        def rows():
            yield from brain.replay
            for text in texts:
                yield brain._idea_model_vector(brain.memory.vector_for_text(text))
        return save_retention_anchors(
            path, rows(), count=len(brain.replay) + len(texts),
            width=int(brain.config.idea_dim), candidate_id=candidate_id,
            policy=getattr(brain, "resource_policy", None),
        )

    def _baseline_paths(self, candidate_id: str) -> Tuple[Path, Path]:
        return (
            self.baselines_path / (candidate_id + ".json"),
            self.baselines_path / (candidate_id + ".safetensors"),
        )

    def _load_baseline(
        self, candidate_id: str, record: Mapping[str, Any]
    ) -> Tuple[Dict[str, Any], RetentionAnchorFile]:
        manifest_path, tensors_path = self._baseline_paths(candidate_id)
        if not manifest_path.is_file() or not tensors_path.is_file():
            raise ValueError("candidate immutable evaluation baseline is missing")
        if _file_sha256(manifest_path) != record.get("baselineManifestSha256"):
            raise ValueError("candidate immutable baseline manifest failed verification")
        manifest = read_json(manifest_path)
        if _file_sha256(tensors_path) != manifest.get("anchorTensorSha256"):
            raise ValueError("candidate immutable baseline tensors failed verification")
        anchors = RetentionAnchorFile(tensors_path, expected_width=int(self.brain.config.idea_dim))
        expected_benchmark = _json_sha256(_benchmark_payload(manifest))
        if expected_benchmark != manifest.get("benchmarkSha256"):
            raise ValueError("candidate immutable benchmark hash failed verification")
        return manifest, anchors

    def propose(
        self,
        *,
        texts: Optional[Sequence[str]] = None,
        source_ids: Optional[Sequence[str]] = None,
        epochs: int = 1,
        learning_rate: Optional[float] = None,
        latent_replay: bool = False,
        objectives: Optional[Sequence[str]] = None,
        provenance: Optional[Mapping[str, Any]] = None,
        architecture_change: Optional[Mapping[str, Any]] = None,
        progress: Optional[Any] = None,
    ) -> Dict[str, Any]:
        assert_architecture_quiescent(self.brain)
        architecture_mutation = _normalize_architecture_change(
            architecture_change
        )
        candidate_type = (
            "architecture" if architecture_mutation is not None else "neural"
        )
        epochs = int(epochs)
        if epochs < 1:
            raise ValueError("epochs must be positive")
        clean_texts = [
            text.replace("\x00", "")
            for text in (texts or [])
            if isinstance(text, str) and text.strip()
        ]
        selected_sources = [
            str(value) for value in (source_ids or []) if str(value)
        ]
        objective_names = [
            str(value)
            for value in (
                objectives
                or (
                    ["latent-replay", "retention", "capability"]
                    if latent_replay and not clean_texts
                    else ["language-prediction", "retention", "capability"]
                )
            )
        ]
        safe_provenance = _validate_json(dict(provenance or {}), "provenance")

        # Save first so the parent checksum and rollback point describe the
        # complete live neural state, including fast synapses and replay.
        self.brain.save()
        parent_parameter_checksum = self.brain.parameter_checksum()
        parent_state_checksum = _bundle_checksum(self.engine_path)
        parent_metadata_sha256 = _file_sha256(self.engine_path / "brain.json")
        candidate_id, candidate_dir = self.brain._begin_candidate(
            "neural-evolution"
        )
        self.brain._record_candidate(
            candidate_dir,
            status="training",
            candidateType=candidate_type,
            parentParameterChecksum=parent_parameter_checksum,
            parentStateChecksum=parent_state_checksum,
            parentMetadataSha256=parent_metadata_sha256,
            objectives=objective_names,
            provenance=safe_provenance,
            sourceIds=selected_sources,
            epochs=epochs,
            learningRate=learning_rate,
            latentReplay=bool(latent_replay),
            architectureCandidate={
                "supported": architecture_mutation is not None,
                "mutation": architecture_mutation,
                "reason": (
                    architecture_mutation["compatibilityBoundary"]
                    if architecture_mutation is not None
                    else (
                        "No architecture mutation requested; all architecture "
                        "shapes must match the immutable baseline."
                    )
                ),
            },
        )

        model_path = candidate_dir / "model"
        model_engine = model_path / "engine"
        candidate = None
        candidate_rng = preserve_runtime_rng(self.brain.device)
        candidate_rng.__enter__()
        try:
            stable = candidate_dir / "stable"
            self.brain.resource_policy.require_disk(snapshot_required_bytes(stable), "isolated evolution working copy")
            resident_admission = self._admit_isolated_load(stable)
            self.brain._record_candidate(candidate_dir, isolatedResidentAdmissionBytes=resident_admission)
            snapshot_files(stable, model_engine)
            # Late import avoids an AdaptiveBrain/evolution import cycle.
            from .brain import AdaptiveBrain

            candidate = AdaptiveBrain.load(
                model_path, expected_brain_id=self.brain.brain_id
            )
            source_lookup = {
                str(source.get("id")): source
                for source in candidate.training_sources
                if isinstance(source, Mapping)
            }
            unavailable_sources: List[str] = []
            for source_id in selected_sources:
                source = source_lookup.get(source_id)
                retained = source.get("raw_text") if source else None
                if isinstance(retained, str) and retained.strip():
                    clean_texts.append(retained)
                else:
                    unavailable_sources.append(source_id)
            if unavailable_sources and not latent_replay:
                raise ValueError(
                    "selected sources have no retained text; enable latentReplay: %s"
                    % ", ".join(unavailable_sources)
                )
            # Stable de-duplication preserves caller order.
            clean_texts = list(dict.fromkeys(clean_texts))
            if (
                not clean_texts
                and not latent_replay
                and architecture_mutation is None
            ):
                raise ValueError(
                    "candidate requires texts, retained sourceIds, or latentReplay"
                )
            self.baselines_path.mkdir(parents=True, exist_ok=True)
            baseline_manifest_path, baseline_tensor_path = self._baseline_paths(candidate_id)
            anchors = self._latent_anchors(candidate, clean_texts, baseline_tensor_path, candidate_id)
            if (
                latent_replay
                and anchors.numel() == 0
                and architecture_mutation is None
            ):
                raise ValueError("latent replay is unavailable because replay is empty")

            baseline_objective = (
                self._objective_loss(candidate, clean_texts)
                if clean_texts
                else self._latent_loss(candidate, anchors)
            )
            baseline_capability = self._capability_loss(candidate)
            baseline_retention = self._latent_loss(candidate, anchors)
            baseline_resources = _tensor_resources(candidate_dir / "stable")
            architecture = _architecture_signature(candidate_dir / "stable")
            anchor_sha = _file_sha256(baseline_tensor_path)
            baseline_manifest = {
                "format": "omni-neural-evolution-baseline",
                "version": 1,
                "candidateId": candidate_id,
                "brainId": self.brain.brain_id,
                "createdAt": _iso_now(),
                "evaluator": EVALUATOR_VERSION,
                "parentParameterChecksum": parent_parameter_checksum,
                "parentStateChecksum": parent_state_checksum,
                "parentMetadataSha256": parent_metadata_sha256,
                "objectives": objective_names,
                "objectiveTextFingerprints": [
                    {
                        "sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
                        "utf8Bytes": len(text.encode("utf-8")),
                    }
                    for text in clean_texts
                ],
                "sourceIds": selected_sources,
                "baselineObjectiveLoss": baseline_objective,
                "baselineCapabilityLoss": baseline_capability,
                "baselineRetentionLoss": baseline_retention,
                "retentionAnchors": int(anchors.shape[0]),
                "resources": baseline_resources,
                "architecture": architecture,
                "architectureMutation": architecture_mutation,
                "anchorTensorSha256": anchor_sha,
                "provenanceSha256": _json_sha256(safe_provenance),
            }
            baseline_manifest["benchmarkSha256"] = _json_sha256(_benchmark_payload(baseline_manifest))
            atomic_write_json(baseline_manifest_path, baseline_manifest)
            baseline_manifest_sha = _file_sha256(baseline_manifest_path)
            self.brain._record_candidate(
                candidate_dir,
                baselineManifestSha256=baseline_manifest_sha,
                benchmarkSha256=baseline_manifest["benchmarkSha256"],
            )

            if progress is not None:
                progress(0.15, "Immutable neural baseline captured")
            architecture_result = None
            if architecture_mutation is not None:
                assert_architecture_quiescent(candidate)
                candidate.core_pager.flush()
                kind = architecture_mutation["mutation"]
                d, ff, old_neurons = int(candidate.config.d_model), int(candidate.config.d_ff), int(candidate.config.router_neurons)
                old_layers = int(candidate.config.n_layers)
                new_layers, new_neurons = growth_dimensions(candidate.config, architecture_mutation)
                if kind == "grow-experts":
                    additions = int(architecture_mutation["addExperts"])
                    hidden = max(16, ff // 2)
                    packed_bytes = (additions * (2 * d + 3 * d * hidden) + 3) // 4
                    nonweight_transient = additions * 64 * 1024
                elif kind == "grow-depth":
                    additions = int(architecture_mutation["addLayers"])
                    packed_bytes = additions * ((4 * d * d + 3 * d * ff + 3) // 4 + 2 * ((d + 3) // 4))
                    nonweight_transient = additions * (5 * d + 2 * ff + 64 * 1024)
                else:
                    additions = new_neurons - old_neurons
                    packed_bytes = (2 * d * additions + new_neurons * new_neurons - old_neurons * old_neurons + 3) // 4
                    # Router timing/stability/use buffers are not packed
                    # weights. Charge complete new buffers while the old
                    # instance still exists, including transfer/load margin.
                    nonweight_transient = 2 * (10 * new_neurons * new_neurons + 16 * new_neurons + 32) + 64 * 1024
                estimated_bytes = packed_bytes * 3 + nonweight_transient + max(1_048_576, 256 * max(d, ff, new_neurons))
                if not candidate._allow_substrate_growth(estimated_bytes):
                    raise ValueError(
                        "architecture growth paused at the host resource reserve"
                    )
                # Packed owner paging does not cover STDP timing/activity or
                # other ordinary controls. Admit their complete simultaneous
                # transfer before allocating a larger accelerator population.
                admission = getattr(candidate.core_pager, "reserve_admission", None)
                if candidate.device.type != "cpu" and callable(admission):
                    admission(max(4096, nonweight_transient), candidate.device)
                expert_count_before = int(candidate.decoder.expert_count)
                with preserve_runtime_rng(candidate.device):
                    if kind == "grow-experts":
                        if int(candidate.config.expert_routing_baseline_count) < 0:
                            candidate.config.expert_routing_baseline_count = expert_count_before
                        for _ in range(additions):
                            index = candidate.decoder.grow_expert()
                            candidate.decoder.experts[index].network.down.fill_ternary_(0)
                    elif kind == "grow-depth":
                        candidate.decoder.grow_depth(additions)
                    else:
                        region_sizes = (int(architecture_mutation["neuronsPerRegion"]),) * int(architecture_mutation["addRegions"]) if kind == "grow-regions" else None
                        candidate.router.grow_neurons(additions, region_sizes=region_sizes)
                candidate.config.n_layers = new_layers
                candidate.config.router_neurons = new_neurons
                candidate.config.native_architecture = reseal_native_descriptor(candidate.config, architecture_mutation)
                candidate.config.validate()
                candidate.core_pager.bind_names((("decoder.", candidate.decoder), ("router.", candidate.router)))
                runtime = candidate._resource_readings()
                activity = candidate._working_attention_status()
                heap = candidate.core_pager.status().get("cpuHeapBytes", 0)
                candidate._native_residency_baseline_bytes = max(
                    int(candidate._native_residency_baseline_bytes),
                    max(0, int(runtime.get("processMemoryBytes") or 0) - int(heap) - int(activity.get("residentBytes", 0))),
                )
                candidate.core_pager.refresh_budget(force=True)
                candidate._configure_working_attention_resources()
                candidate._replace_optimizer(learning_rate)
                candidate._sync_stability_state()
                candidate.save()
                preservation = {
                    filename: verify_preserved_tensor_prefixes(candidate_dir / "stable" / filename, model_engine / filename)
                    for filename in ("core.safetensors", "plasticity.safetensors")
                }
                architecture_result = {
                    **architecture_mutation,
                    "expertCountBefore": expert_count_before,
                    "expertCountAfter": int(candidate.decoder.expert_count),
                    "estimatedGrowthBytes": estimated_bytes,
                    "layersBefore": old_layers, "layersAfter": new_layers,
                    "routerNeuronsBefore": old_neurons, "routerNeuronsAfter": new_neurons,
                    "preservedBeforeTraining": preservation,
                    "normalizationAndHeadGeometryChanged": False,
                    "qualityVerified": False,
                    "resourceReadings": candidate._resource_readings(),
                }
                if progress is not None:
                    progress(
                        0.2,
                        "Compatible native architecture inserted with old-state byte proof",
                    )
            if clean_texts:
                def scaled_progress(
                    value: float,
                    message: str,
                    data: Optional[Dict[str, Any]] = None,
                ) -> None:
                    assert progress is not None
                    progress(
                        0.15 + 0.7 * float(value),
                        message,
                        dict(data) if isinstance(data, Mapping) else None,
                    )

                training = candidate.train(
                    texts=clean_texts,
                    epochs=epochs,
                    learning_rate=learning_rate,
                    progress=(
                        (
                            scaled_progress
                        )
                        if progress is not None
                        else None
                    ),
                )
            elif latent_replay and anchors.numel() > 0:
                def scaled_replay_progress(
                    value: float,
                    message: str,
                    data: Optional[Dict[str, Any]] = None,
                ) -> None:
                    assert progress is not None
                    progress(
                        0.15 + 0.7 * float(value),
                        message,
                        dict(data) if isinstance(data, Mapping) else None,
                    )

                training = candidate._train_evolution_replay_candidate(
                    steps=epochs,
                    progress=(
                        (
                            scaled_replay_progress
                        )
                        if progress is not None
                        else None
                    ),
                )
            else:
                training = {
                    "promoted": True,
                    "mode": "function-preserving-architecture-insertion",
                    "steps": 0,
                    "reason": (
                        "Compatible zero-residual/dormant-region capacity inserted "
                        "with old-state byte preservation; useful improvement remains unverified."
                    ),
                }
            candidate.save()
            final_objective = (
                self._objective_loss(candidate, clean_texts)
                if clean_texts
                else self._latent_loss(candidate, anchors)
            )
            final_capability = self._capability_loss(candidate)
            final_retention = self._latent_loss(candidate, anchors)
            candidate_state_checksum = _bundle_checksum(model_engine)
            candidate_metadata_sha256 = _file_sha256(model_engine / "brain.json")
            candidate_parameter_checksum = candidate.parameter_checksum()
            diff_sha, diff_norm = _diff_checksum(
                candidate_dir / "stable", model_engine
            )
            resources = _tensor_resources(model_engine)
            unsafe_files = [
                str(path.relative_to(candidate_dir))
                for path in candidate_dir.rglob("*")
                if path.is_file() and path.suffix.lower() in UNSAFE_TENSOR_SUFFIXES
            ]
            changed = candidate_state_checksum != parent_state_checksum
            training_promoted = bool(training.get("promoted", False))
            status = "ready" if training_promoted and changed and not unsafe_files else "rejected"
            rejection = ""
            if not training_promoted:
                rejection = str(
                    training.get("rejection", "isolated training did not pass")
                )
            elif not changed:
                rejection = "candidate produced no neural-state change"
            elif unsafe_files:
                rejection = "candidate contains unsafe executable tensor files"
            self.brain._record_candidate(
                candidate_dir,
                status=status,
                reason=rejection,
                readyAt=_iso_now() if status == "ready" else None,
                rejectedAt=_iso_now() if status == "rejected" else None,
                candidateParameterChecksum=candidate_parameter_checksum,
                candidateStateChecksum=candidate_state_checksum,
                candidateMetadataSha256=candidate_metadata_sha256,
                candidateDiffSha256=diff_sha,
                candidateDeltaNorm=diff_norm,
                candidateDeltaNormBasis="encoded-checkpoint-state-diff-not-learning-magnitude",
                training=training,
                preliminaryMetrics={
                    "baselineObjectiveLoss": baseline_objective,
                    "candidateObjectiveLoss": final_objective,
                    "baselineCapabilityLoss": baseline_capability,
                    "candidateCapabilityLoss": final_capability,
                    "baselineRetentionLoss": baseline_retention,
                    "candidateRetentionLoss": final_retention,
                },
                resources=resources,
                architectureMutation=architecture_result,
                unsafeTensorFiles=unsafe_files,
                modelFiles={
                    "metadata": "model/engine/brain.json",
                    "core": "model/engine/core.safetensors",
                    "plasticity": "model/engine/plasticity.safetensors",
                },
                completedAt=_iso_now(),
            )
            result = read_json(candidate_dir / "candidate.json")
            self.brain.events.append(
                (
                    "architecture-evolution-candidate"
                    if candidate_type == "architecture"
                    else "neural-evolution-candidate"
                ),
                {
                    "candidateId": candidate_id,
                    "candidateType": candidate_type,
                    "status": status,
                    "parentStateChecksum": parent_state_checksum,
                    "candidateStateChecksum": candidate_state_checksum,
                    "candidateDiffSha256": diff_sha,
                    "objectives": objective_names,
                    "resources": resources,
                    "architectureMutation": architecture_result,
                    "reason": rejection,
                },
            )
            if progress is not None:
                progress(1.0, "Isolated neural candidate ready")
            return result
        except Exception as error:
            self.brain._record_candidate(
                candidate_dir,
                status="rejected",
                reason="candidate proposal failed: %s" % error,
                rejectedAt=_iso_now(),
            )
            raise
        finally:
            try:
                if candidate is not None:
                    candidate.close()
            finally:
                candidate_rng.__exit__(None, None, None)

    def evaluate(self, candidate_id: str) -> Dict[str, Any]:
        candidate_dir, record = self._record(candidate_id)
        if record.get("status") not in {"ready", "evaluated"}:
            raise ValueError(
                "candidate cannot be evaluated from status %s"
                % record.get("status")
            )
        baseline, anchors = self._load_baseline(candidate_id, record)
        model_path = self._model_path(candidate_dir)
        from .brain import AdaptiveBrain
        self._admit_isolated_load(model_path / "engine")
        candidate = None
        candidate_rng = preserve_runtime_rng(self.brain.device)
        candidate_rng.__enter__()
        try:
            candidate = AdaptiveBrain.load(model_path, expected_brain_id=self.brain.brain_id)
            model_engine = model_path / "engine"
            state_checksum = _bundle_checksum(model_engine)
            integrity_passed = state_checksum == record.get(
                "candidateStateChecksum"
            )
            if record.get("candidateMetadataSha256") is not None:
                integrity_passed = integrity_passed and _file_sha256(model_engine / "brain.json") == record["candidateMetadataSha256"]
            candidate_architecture = _architecture_signature(model_engine)
            architecture_mutation = baseline.get("architectureMutation")
            architecture_passed = _architecture_compatible(
                baseline["architecture"],
                candidate_architecture,
                architecture_mutation,
            )
            resources = _tensor_resources(model_engine)
            baseline_bytes = int(baseline["resources"]["tensorBytes"])
            growth_bytes = max(0, resources["tensorBytes"] - baseline_bytes)
            resource_readings = candidate._resource_readings()
            disk_free = resource_readings.get("diskFreeBytes")
            memory_free = resource_readings.get("availableMemoryBytes")
            resource_passed = not (
                (
                    isinstance(disk_free, int)
                    and disk_free < max(512 * 1024 * 1024, growth_bytes * 8)
                )
                or (
                    isinstance(memory_free, int)
                    and memory_free < max(384 * 1024 * 1024, growth_bytes * 4)
                )
            )
            resources["growthBytes"] = growth_bytes
            resources["host"] = resource_readings
            audit = candidate._ternary_audit()
            ternary_passed = (
                float(audit.get("coverage", 0.0)) == 1.0
                and not audit.get("violations")
            )
            capability_loss = self._capability_loss(candidate)
            retention_loss = self._latent_loss(candidate, anchors)
            preliminary = record.get("preliminaryMetrics", {})
            baseline_objective = float(baseline["baselineObjectiveLoss"])
            objective_loss = float(
                preliminary.get("candidateObjectiveLoss", math.inf)
            )
            objective_passed = (
                math.isfinite(objective_loss)
                and objective_loss <= baseline_objective * 1.05 + 1e-6
            )
            capability_passed = (
                math.isfinite(capability_loss)
                and capability_loss
                <= float(baseline["baselineCapabilityLoss"]) * 1.25 + 1e-6
            )
            retention_passed = (
                math.isfinite(retention_loss)
                and retention_loss
                <= float(baseline["baselineRetentionLoss"]) * 1.10 + 1e-6
            )
            changed = state_checksum != baseline["parentStateChecksum"]
            checks = {
                "integrity": integrity_passed,
                "architectureCompatible": architecture_passed,
                "resources": resource_passed,
                "ternaryCoverage": ternary_passed,
                "objectiveNonRegression": objective_passed,
                "capabilityRetention": capability_passed,
                "neuralRetention": retention_passed,
                "changed": changed,
            }
            passed = all(checks.values())
            failures = [name for name, value in checks.items() if not value]
            metrics = {
                "baselineObjectiveLoss": baseline_objective,
                "candidateObjectiveLoss": objective_loss,
                "baselineCapabilityLoss": float(
                    baseline["baselineCapabilityLoss"]
                ),
                "candidateCapabilityLoss": capability_loss,
                "baselineRetentionLoss": float(
                    baseline["baselineRetentionLoss"]
                ),
                "candidateRetentionLoss": retention_loss,
                "candidateDeltaNorm": float(
                    record.get("candidateDeltaNorm", 0.0)
                ),
                "expertCountBefore": int(
                    baseline["architecture"].get("expertCount", 0)
                ),
                "expertCountAfter": int(
                    candidate_architecture.get("expertCount", 0)
                ),
            }
            evaluation = {
                "evaluator": EVALUATOR_VERSION,
                "evaluatedAt": _iso_now(),
                "benchmarkSha256": baseline["benchmarkSha256"],
                "passed": passed,
                "checks": checks,
                "failures": failures,
                "metrics": metrics,
                "resources": resources,
                "ternaryAudit": audit,
                "architectureMutation": record.get(
                    "architectureMutation"
                ),
                "architecture": candidate_architecture,
                "candidateDeltaNormBasis": record.get("candidateDeltaNormBasis", "legacy-checkpoint-state-diff"),
            }
            evaluation["evaluationSha256"] = _json_sha256(evaluation)
            status = "evaluated" if passed else "rejected"
            self.brain._record_candidate(
                candidate_dir,
                status=status,
                evaluation=evaluation,
                reason=(
                    ""
                    if passed
                    else "candidate failed: %s" % ", ".join(failures)
                ),
                evaluatedAt=_iso_now(),
                rejectedAt=_iso_now() if not passed else None,
            )
            self.brain.events.append(
                "neural-evolution-evaluated",
                {
                    "candidateId": candidate_id,
                    "passed": passed,
                    "checks": checks,
                    "evaluationSha256": evaluation["evaluationSha256"],
                },
            )
            return {
                "brainId": self.brain.brain_id,
                "candidateId": candidate_id,
                **evaluation,
                "status": status,
            }
        finally:
            try:
                if candidate is not None:
                    candidate.close()
            finally:
                candidate_rng.__exit__(None, None, None)

    def list(self) -> Dict[str, Any]:
        candidates: List[Dict[str, Any]] = []
        if self.candidates_path.is_dir():
            for path in self.candidates_path.iterdir():
                record_path = path / "candidate.json"
                if not path.is_dir() or not record_path.is_file():
                    continue
                try:
                    record = read_json(record_path)
                except (OSError, ValueError, json.JSONDecodeError):
                    continue
                if record.get("kind") == "neural-evolution":
                    candidates.append(record)
        candidates.sort(
            key=lambda value: str(value.get("createdAt", "")), reverse=True
        )
        return {"brainId": self.brain.brain_id, "candidates": candidates}

    def reject(self, candidate_id: str, reason: str = "") -> Dict[str, Any]:
        candidate_dir, record = self._record(candidate_id)
        if record.get("status") == "promoted":
            raise ValueError("promoted candidates must be rolled back")
        if record.get("status") == "rolled-back":
            raise ValueError("rolled-back candidate is already terminal")
        final_reason = str(reason).strip() or "rejected by operator"
        self.brain._record_candidate(
            candidate_dir,
            status="rejected",
            reason=final_reason,
            rejectedAt=_iso_now(),
        )
        self.brain.events.append(
            "neural-evolution-rejected",
            {"candidateId": candidate_id, "reason": final_reason},
        )
        return {
            "brainId": self.brain.brain_id,
            "candidateId": candidate_id,
            "status": "rejected",
            "reason": final_reason,
        }

    def promote(self, candidate_id: str) -> Dict[str, Any]:
        assert_architecture_quiescent(self.brain)
        candidate_dir, record = self._record(candidate_id)
        if record.get("status") == "ready":
            self.evaluate(candidate_id)
            candidate_dir, record = self._record(candidate_id)
        if record.get("status") != "evaluated":
            raise ValueError(
                "candidate must pass evaluation before promotion"
            )
        evaluation = record.get("evaluation", {})
        if not evaluation.get("passed"):
            raise ValueError("candidate evaluation did not pass")
        live_parameter_checksum = self.brain.parameter_checksum()
        live_state_checksum = _bundle_checksum(self.engine_path)
        if (
            live_parameter_checksum != record.get("parentParameterChecksum")
            or live_state_checksum != record.get("parentStateChecksum")
            or (record.get("parentMetadataSha256") is not None
                and _file_sha256(self.engine_path / "brain.json") != record["parentMetadataSha256"])
        ):
            self.brain._record_candidate(
                candidate_dir,
                status="stale",
                reason="live baseline checksum changed after proposal",
                staleAt=_iso_now(),
                observedLiveParameterChecksum=live_parameter_checksum,
                observedLiveStateChecksum=live_state_checksum,
            )
            raise ValueError("candidate baseline is stale; live brain has changed")
        model_engine = self._model_path(candidate_dir) / "engine"
        candidate_state_checksum = _bundle_checksum(model_engine)
        if candidate_state_checksum != record.get("candidateStateChecksum"):
            raise ValueError("candidate checkpoint failed checksum verification")
        if record.get("candidateMetadataSha256") is not None and _file_sha256(model_engine / "brain.json") != record["candidateMetadataSha256"]:
            raise ValueError("candidate metadata/context/lineage failed checksum verification")
        if _architecture_signature(model_engine) != evaluation.get("architecture"):
            raise ValueError("candidate architecture changed after immutable evaluation")
        self.brain._record_candidate(
            candidate_dir,
            status="promoting",
            operation="promote",
            promotionStartedAt=_iso_now(),
            rollbackPoint="stable/",
        )
        try:
            for filename in ("core.safetensors", "plasticity.safetensors"):
                _copy_atomic(model_engine / filename, self.engine_path / filename)
            copy_substrate_snapshot(model_engine, self.engine_path)
            copy_mutable_state_snapshot(model_engine, self.engine_path)
            _copy_atomic(
                model_engine / "brain.json",
                self.engine_path / "brain.json",
            )
            promoted_checksum = _bundle_checksum(self.engine_path)
            if promoted_checksum != candidate_state_checksum:
                raise RuntimeError("promoted checkpoint checksum mismatch")
        except Exception:
            self.brain._restore_candidate_checkpoint(candidate_dir)
            self.brain._record_candidate(
                candidate_dir,
                status="rejected",
                reason="promotion failed; rollback point restored",
                rejectedAt=_iso_now(),
            )
            raise
        self.brain._record_candidate(
            candidate_dir,
            status="promoted",
            promotedAt=_iso_now(),
            promotedStateChecksum=promoted_checksum,
            promotedMetadataSha256=_file_sha256(self.engine_path / "brain.json"),
            rollbackAvailable=True,
        )
        candidate_type = str(record.get("candidateType", "neural"))
        self.brain.events.append(
            (
                "architecture-evolution-promoted"
                if candidate_type == "architecture"
                else "neural-evolution-promoted"
            ),
            {
                "candidateId": candidate_id,
                "candidateType": candidate_type,
                "parentStateChecksum": record["parentStateChecksum"],
                "promotedStateChecksum": promoted_checksum,
                "candidateDiffSha256": record["candidateDiffSha256"],
                "evaluationSha256": evaluation.get("evaluationSha256"),
                "rollbackPoint": "stable/",
                "architectureMutation": record.get(
                    "architectureMutation"
                ),
            },
        )
        return {
            "brainId": self.brain.brain_id,
            "candidateId": candidate_id,
            "promoted": True,
            "parameterChecksumBefore": record["parentParameterChecksum"],
            "stateChecksumBefore": record["parentStateChecksum"],
            "stateChecksumAfter": promoted_checksum,
            "rollbackAvailable": True,
            "reloadRequired": True,
            "candidateType": candidate_type,
            "architectureMutation": record.get(
                "architectureMutation"
            ),
        }

    def rollback(self, candidate_id: str, force: bool = False) -> Dict[str, Any]:
        assert_architecture_quiescent(self.brain)
        candidate_dir, record = self._record(candidate_id)
        if record.get("status") != "promoted":
            raise ValueError("only a promoted candidate can be rolled back")
        current_checksum = _bundle_checksum(self.engine_path)
        expected = str(record.get("promotedStateChecksum", ""))
        metadata_changed = record.get("promotedMetadataSha256") is not None and _file_sha256(self.engine_path / "brain.json") != record["promotedMetadataSha256"]
        if (current_checksum != expected or metadata_changed) and not force:
            raise ValueError(
                "live brain changed after promotion; explicit force is required"
            )
        self.brain._record_candidate(
            candidate_dir,
            # Existing crash recovery treats this phase as a recoverable
            # checkpoint transaction and restores ``stable/``.
            status="promoting",
            operation="rollback",
            rollbackStartedAt=_iso_now(),
        )
        self.brain._restore_candidate_checkpoint(candidate_dir)
        restored_checksum = _bundle_checksum(self.engine_path)
        if restored_checksum != record.get("parentStateChecksum"):
            raise RuntimeError("rollback point checksum mismatch")
        if record.get("parentMetadataSha256") is not None and _file_sha256(self.engine_path / "brain.json") != record["parentMetadataSha256"]:
            raise RuntimeError("rollback point metadata/context/lineage checksum mismatch")
        self.brain._record_candidate(
            candidate_dir,
            status="rolled-back",
            rolledBackAt=_iso_now(),
            rollbackForced=bool(force),
            restoredStateChecksum=restored_checksum,
            rollbackAvailable=False,
        )
        self.brain.events.append(
            (
                "architecture-evolution-rolled-back"
                if record.get("candidateType") == "architecture"
                else "neural-evolution-rolled-back"
            ),
            {
                "candidateId": candidate_id,
                "candidateType": record.get("candidateType", "neural"),
                "forced": bool(force),
                "stateChecksumBefore": current_checksum,
                "stateChecksumAfter": restored_checksum,
            },
        )
        return {
            "brainId": self.brain.brain_id,
            "candidateId": candidate_id,
            "rolledBack": True,
            "stateChecksumBefore": current_checksum,
            "stateChecksumAfter": restored_checksum,
            "reloadRequired": True,
        }
