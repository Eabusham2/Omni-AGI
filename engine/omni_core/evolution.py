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

from .persistence import (
    atomic_save_tensors,
    atomic_write_json,
    load_tensors,
    read_json,
    snapshot_files,
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


def _bundle_tensors(engine_path: Path) -> Dict[str, torch.Tensor]:
    tensors: Dict[str, torch.Tensor] = {}
    for filename, prefix in (
        ("core.safetensors", "core:"),
        ("plasticity.safetensors", "plasticity:"),
    ):
        for name, tensor in load_tensors(engine_path / filename, device="cpu").items():
            tensors[prefix + name] = tensor.detach().cpu().contiguous()
    return tensors


def _bundle_checksum(engine_path: Path) -> str:
    digest = hashlib.sha256()
    for name, tensor in sorted(_bundle_tensors(engine_path).items()):
        digest.update(name.encode("utf-8"))
        digest.update(str(tuple(tensor.shape)).encode("ascii"))
        digest.update(str(tensor.dtype).encode("ascii"))
        digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def _architecture_signature(engine_path: Path) -> Dict[str, Any]:
    metadata = read_json(engine_path / "brain.json")
    tensors = _bundle_tensors(engine_path)
    return {
        "fixedTensors": {
            name: {
                "shape": list(tensor.shape),
                "dtype": str(tensor.dtype),
            }
            for name, tensor in sorted(tensors.items())
            # Expert modules are the one stable-v1 architecture extension that
            # can be instantiated from metadata before strict tensor loading.
            # Every other core/router shape remains immutable.
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
    }


def _normalize_architecture_change(
    value: Optional[Mapping[str, Any]],
) -> Optional[Dict[str, Any]]:
    if not value:
        return None
    mutation = str(value.get("mutation", ""))
    if mutation != "grow-experts":
        raise ValueError(
            "stable v1 supports only the compatible grow-experts architecture mutation"
        )
    unexpected = set(value).difference({"mutation", "addExperts"})
    if unexpected:
        raise ValueError(
            "unsupported architecture mutation fields: %s"
            % ", ".join(sorted(str(name) for name in unexpected))
        )
    raw_count = value.get("addExperts", 1)
    if isinstance(raw_count, bool):
        raise ValueError("architecture addExperts must be a positive integer")
    try:
        count = int(raw_count)
    except (TypeError, ValueError) as error:
        raise ValueError(
            "architecture addExperts must be a positive integer"
        ) from error
    if count < 1 or count != raw_count:
        raise ValueError("architecture addExperts must be a positive integer")
    return {
        "mutation": "grow-experts",
        "addExperts": count,
        "compatibilityBoundary": (
            "Adds only load-aware ternary residual experts and prototypes; "
            "decoder width, depth, attention, router, modality, and existing "
            "tensor shapes remain immutable."
        ),
    }


def _architecture_compatible(
    baseline: Mapping[str, Any],
    candidate: Mapping[str, Any],
    mutation: Optional[Mapping[str, Any]],
) -> bool:
    if baseline.get("fixedTensors") != candidate.get("fixedTensors"):
        return False
    if (
        baseline.get("expertFormat") != "ternary-residual-expert-v1"
        or candidate.get("expertFormat") != "ternary-residual-expert-v1"
    ):
        return False
    before = int(baseline.get("expertCount", -1))
    after = int(candidate.get("expertCount", -1))
    if mutation is None:
        return after == before
    return (
        mutation.get("mutation") == "grow-experts"
        and after == before + int(mutation.get("addExperts", 0))
    )


def _tensor_resources(engine_path: Path) -> Dict[str, int]:
    tensors = _bundle_tensors(engine_path)
    return {
        "tensorCount": len(tensors),
        "elementCount": sum(int(tensor.numel()) for tensor in tensors.values()),
        "tensorBytes": sum(
            int(tensor.numel() * tensor.element_size())
            for tensor in tensors.values()
        ),
        "checkpointBytes": sum(
            int((engine_path / filename).stat().st_size)
            for filename in ("core.safetensors", "plasticity.safetensors")
        ),
    }


def _diff_checksum(baseline_path: Path, candidate_path: Path) -> Tuple[str, float]:
    baseline = _bundle_tensors(baseline_path)
    candidate = _bundle_tensors(candidate_path)
    digest = hashlib.sha256()
    squared_norm = 0.0
    for name in sorted(set(baseline).union(candidate)):
        left = baseline.get(name)
        right = candidate.get(name)
        digest.update(name.encode("utf-8"))
        if left is None:
            assert right is not None
            digest.update(b"added")
            digest.update(right.contiguous().numpy().tobytes())
            squared_norm += float(right.float().pow(2).sum().item())
        elif right is None:
            digest.update(b"removed")
            digest.update(left.contiguous().numpy().tobytes())
            squared_norm += float(left.float().pow(2).sum().item())
        elif tuple(left.shape) != tuple(right.shape) or left.dtype != right.dtype:
            # Dynamic substrate/replay state may legitimately change shape.
            # Architecture compatibility is checked separately.
            digest.update(b"reshaped")
            digest.update(str(tuple(left.shape)).encode("ascii"))
            digest.update(str(tuple(right.shape)).encode("ascii"))
            digest.update(right.contiguous().numpy().tobytes())
            squared_norm += float(left.float().pow(2).sum().item())
            squared_norm += float(right.float().pow(2).sum().item())
        else:
            delta = right.float() - left.float()
            digest.update(delta.contiguous().numpy().tobytes())
            squared_norm += float(delta.pow(2).sum().item())
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
    def _latent_loss(brain: Any, anchors: torch.Tensor) -> float:
        if anchors.numel() == 0:
            return 0.0
        values = anchors.to(brain.device)
        brain.idea_adapter.eval()
        with torch.no_grad():
            return float(F.mse_loss(brain.idea_adapter(values), values).item())

    @classmethod
    def _capability_loss(cls, brain: Any) -> float:
        return cls._objective_loss(brain, CAPABILITY_PROBES)

    @staticmethod
    def _latent_anchors(brain: Any, texts: Sequence[str]) -> torch.Tensor:
        anchors: List[torch.Tensor] = [
            value.detach().cpu().reshape(-1) for value in brain.replay
        ]
        for text in texts:
            anchors.append(
                brain._idea_model_vector(
                    brain.memory.vector_for_text(text)
                ).detach().cpu().reshape(-1)
            )
        if not anchors:
            return torch.empty((0, int(brain.config.idea_dim)), dtype=torch.float32)
        return torch.stack(anchors).float()

    def _baseline_paths(self, candidate_id: str) -> Tuple[Path, Path]:
        return (
            self.baselines_path / (candidate_id + ".json"),
            self.baselines_path / (candidate_id + ".safetensors"),
        )

    def _load_baseline(
        self, candidate_id: str, record: Mapping[str, Any]
    ) -> Tuple[Dict[str, Any], torch.Tensor]:
        manifest_path, tensors_path = self._baseline_paths(candidate_id)
        if not manifest_path.is_file() or not tensors_path.is_file():
            raise ValueError("candidate immutable evaluation baseline is missing")
        if _file_sha256(manifest_path) != record.get("baselineManifestSha256"):
            raise ValueError("candidate immutable baseline manifest failed verification")
        manifest = read_json(manifest_path)
        if _file_sha256(tensors_path) != manifest.get("anchorTensorSha256"):
            raise ValueError("candidate immutable baseline tensors failed verification")
        tensors = load_tensors(tensors_path, device="cpu")
        anchors = tensors.get("retention_anchors")
        if anchors is None:
            raise ValueError("candidate retention anchors are missing")
        expected_benchmark = _json_sha256(
            {
                "evaluator": EVALUATOR_VERSION,
                "capabilityProbes": list(CAPABILITY_PROBES),
                "parentStateChecksum": manifest["parentStateChecksum"],
                "anchorTensorSha256": manifest["anchorTensorSha256"],
                "architecture": manifest["architecture"],
                "architectureMutation": manifest.get(
                    "architectureMutation"
                ),
            }
        )
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
        candidate_id, candidate_dir = self.brain._begin_candidate(
            "neural-evolution"
        )
        self.brain._record_candidate(
            candidate_dir,
            status="training",
            candidateType=candidate_type,
            parentParameterChecksum=parent_parameter_checksum,
            parentStateChecksum=parent_state_checksum,
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
        snapshot_files(candidate_dir / "stable", model_engine)
        candidate = None
        try:
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
            anchors = self._latent_anchors(candidate, clean_texts)
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
            self.baselines_path.mkdir(parents=True, exist_ok=True)
            baseline_manifest_path, baseline_tensor_path = self._baseline_paths(
                candidate_id
            )
            atomic_save_tensors(
                baseline_tensor_path,
                {"retention_anchors": anchors},
                metadata={
                    "format": "omni-evolution-evaluation-anchors",
                    "candidate_id": candidate_id,
                },
            )
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
            baseline_manifest["benchmarkSha256"] = _json_sha256(
                {
                    "evaluator": EVALUATOR_VERSION,
                    "capabilityProbes": list(CAPABILITY_PROBES),
                    "parentStateChecksum": parent_state_checksum,
                    "anchorTensorSha256": anchor_sha,
                    "architecture": architecture,
                    "architectureMutation": architecture_mutation,
                }
            )
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
                additions = int(architecture_mutation["addExperts"])
                hidden = max(16, int(candidate.config.d_ff) // 2)
                # Master weights, optimizer moments, and checkpoint copies all
                # consume space even though inference forwards are ternary.
                estimated_parameters = additions * (
                    2 * int(candidate.config.d_model)
                    + 3 * int(candidate.config.d_model) * hidden
                )
                estimated_bytes = estimated_parameters * 12
                if not candidate._allow_substrate_growth(estimated_bytes):
                    raise ValueError(
                        "architecture growth paused at the host resource reserve"
                    )
                expert_count_before = int(candidate.decoder.expert_count)
                for _ in range(additions):
                    index = candidate.decoder.grow_expert()
                    # A newly inserted architecture is function-preserving
                    # before isolated training. Its residual path begins at
                    # zero and can then learn inside the candidate overlay.
                    candidate.decoder.experts[index].network.down.weight.data.zero_()
                candidate._optimizer = candidate._new_optimizer(
                    learning_rate
                )
                candidate._sync_stability_state()
                architecture_result = {
                    **architecture_mutation,
                    "expertCountBefore": expert_count_before,
                    "expertCountAfter": int(candidate.decoder.expert_count),
                    "estimatedGrowthBytes": estimated_bytes,
                    "resourceReadings": candidate._resource_readings(),
                }
                if progress is not None:
                    progress(
                        0.2,
                        "Compatible ternary residual expert architecture created",
                    )
            if clean_texts:
                training = candidate.train(
                    texts=clean_texts,
                    epochs=epochs,
                    learning_rate=learning_rate,
                    progress=(
                        (
                            lambda value, message: progress(
                                0.15 + 0.7 * float(value), message
                            )
                        )
                        if progress is not None
                        else None
                    ),
                )
            elif latent_replay and anchors.numel() > 0:
                training = candidate.consolidate(
                    steps=epochs,
                    progress=(
                        (
                            lambda value, message: progress(
                                0.15 + 0.7 * float(value), message
                            )
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
                        "The zero-residual compatible expert adds capacity "
                        "without changing blank-brain outputs before future learning."
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
                candidateDiffSha256=diff_sha,
                candidateDeltaNorm=diff_norm,
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
            if candidate is not None:
                candidate.events.close()

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

        candidate = AdaptiveBrain.load(
            model_path, expected_brain_id=self.brain.brain_id
        )
        try:
            model_engine = model_path / "engine"
            state_checksum = _bundle_checksum(model_engine)
            integrity_passed = state_checksum == record.get(
                "candidateStateChecksum"
            )
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
            candidate.events.close()

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
        self.brain._record_candidate(
            candidate_dir,
            status="promoting",
            operation="promote",
            promotionStartedAt=_iso_now(),
            rollbackPoint="stable/",
        )
        try:
            for filename in (
                "core.safetensors",
                "plasticity.safetensors",
                "brain.json",
            ):
                _copy_atomic(model_engine / filename, self.engine_path / filename)
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
        candidate_dir, record = self._record(candidate_id)
        if record.get("status") != "promoted":
            raise ValueError("only a promoted candidate can be rolled back")
        current_checksum = _bundle_checksum(self.engine_path)
        expected = str(record.get("promotedStateChecksum", ""))
        if current_checksum != expected and not force:
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
