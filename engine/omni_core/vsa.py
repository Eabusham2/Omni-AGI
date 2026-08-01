"""Distributed neural substrate with VSA binding and sparse ternary synapses.

The stable OmniCortex format does not keep an authoritative "idea database"
beside the neural state.  Concepts, experiences, and higher-order ideas are
represented as neuron assemblies connected by plastic synapses.  The
``concepts``/``ideas``/``relations`` properties at the bottom of the class are
compatibility views over that same substrate for the desktop inspector.
"""

import hashlib
import json
import math
import os
import re
import tempfile
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import torch
from torch.nn import functional as F

from .persistence import (
    atomic_save_tensors,
    atomic_write_bytes,
    atomic_write_json,
    load_tensors,
    read_json,
)


_WORD = re.compile(r"[A-Za-z0-9_][A-Za-z0-9_+\-'.]{1,95}")
_SEGMENT = re.compile(r"(?<=[.!?])\s+|\n+")
_SUBSTRATE_STORE_FORMAT = "omni-substrate-shards"
_SUBSTRATE_STORE_VERSION = 1
_SYNAPSE_TENSOR_FIELDS = (
    "latent_weight",
    "effective_weight",
    "eligibility",
    "plasticity",
    "uses",
    "stability",
    "last_updated_at",
)


def _now() -> float:
    return time.time()


class SubstrateResourcePause(RuntimeError):
    """Raised when the host reserve asks structural growth to pause."""


class HypervectorSpace:
    """Deterministic bipolar hypervectors with bind, bundle, and permutation."""

    def __init__(self, dimensions: int = 256, seed: int = 7):
        if dimensions < 16:
            raise ValueError("hypervector dimensions must be at least 16")
        self.dimensions = int(dimensions)
        self.seed = int(seed)

    def symbol(self, name: str) -> torch.Tensor:
        digest = hashlib.sha256(
            ("%d:%s" % (self.seed, name)).encode("utf-8")
        ).digest()
        generator = torch.Generator(device="cpu")
        generator.manual_seed(int.from_bytes(digest[:8], "little") & 0x7FFFFFFF)
        values = torch.randint(
            0, 2, (self.dimensions,), generator=generator, dtype=torch.float32
        )
        return values.mul(2.0).sub(1.0)

    @staticmethod
    def bind(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
        return left * right

    @staticmethod
    def bundle(vectors: Sequence[torch.Tensor]) -> torch.Tensor:
        if not vectors:
            raise ValueError("cannot bundle an empty vector sequence")
        summed = torch.stack(list(vectors)).sum(dim=0)
        bundled = torch.sign(summed)
        bundled[bundled == 0] = 1.0
        return bundled

    @staticmethod
    def weighted_bundle(
        vectors: Sequence[torch.Tensor], weights: Sequence[float]
    ) -> torch.Tensor:
        if not vectors or len(vectors) != len(weights):
            raise ValueError("weighted bundle needs equally sized non-empty inputs")
        stacked = torch.stack([vector.float() for vector in vectors])
        scale = torch.tensor(weights, dtype=stacked.dtype).reshape(-1, 1)
        result = torch.sign((stacked * scale).sum(dim=0))
        result[result == 0] = 1.0
        return result

    @staticmethod
    def permute(vector: torch.Tensor, steps: int = 1) -> torch.Tensor:
        if vector.ndim == 0:
            raise ValueError("cannot permute a scalar hypervector")
        return torch.roll(vector, shifts=int(steps), dims=-1)

    @staticmethod
    def inverse_permute(vector: torch.Tensor, steps: int = 1) -> torch.Tensor:
        return torch.roll(vector, shifts=-int(steps), dims=-1)

    @staticmethod
    def similarity(left: torch.Tensor, right: torch.Tensor) -> float:
        return float(
            F.cosine_similarity(
                left.float().reshape(1, -1),
                right.float().reshape(1, -1),
            ).item()
        )


class NeuralSubstrate:
    """One growable substrate for neurons, assemblies, and ternary synapses.

    Structural growth is governed by a host-resource callback rather than a
    user-set or implementation-defined cardinality.
    """

    SCHEMA = "neural-substrate-1"

    def __init__(
        self,
        dimensions: int = 256,
        seed: int = 7,
        growth_guard: Optional[Callable[[int], bool]] = None,
    ):
        self.space = HypervectorSpace(dimensions, seed)
        self.neurons: Dict[str, Dict[str, Any]] = {}
        self.neuron_vectors: Dict[str, torch.Tensor] = {}
        self.assemblies: List[Dict[str, Any]] = []
        self.assembly_vectors: Dict[str, torch.Tensor] = {}
        self.synapses: Dict[str, Dict[str, Any]] = {}
        self.growth_events = 0
        self.growth_pauses = 0
        self.growth_guard = growth_guard
        self.persistence_manifest: Optional[Dict[str, Any]] = None
        self._last_recall_audit: Dict[str, Any] = {
            "rule": "effective-ternary-recurrent-settling",
            "exactTernaryContribution": True,
            "latentMagnitudeUsed": False,
            "settledRounds": 0,
            "inhibitoryEdges": 0,
            "inhibitorySignals": 0,
            "suppressedAssemblies": 0,
        }

    @property
    def concepts(self) -> Dict[str, Dict[str, Any]]:
        """Inspector view over substrate neurons (not separate storage)."""

        return self.neurons

    @property
    def concept_vectors(self) -> Dict[str, torch.Tensor]:
        return self.neuron_vectors

    @property
    def ideas(self) -> List[Dict[str, Any]]:
        """Inspector view over distributed assemblies."""

        return self.assemblies

    @property
    def idea_vectors(self) -> Dict[str, torch.Tensor]:
        return self.assembly_vectors

    @property
    def relations(self) -> Dict[str, Dict[str, Any]]:
        """Inspector view over the authoritative synapse store."""

        return self.synapses

    @property
    def capacity_expansions(self) -> int:
        return self.growth_events

    @staticmethod
    def extract_concepts(text: str, limit: Optional[int] = None) -> List[str]:
        """Extract all ordered atomic and compositional units.

        Stable OmniCortex has no fixed 48-item cutoff.  ``limit`` remains an
        explicit diagnostic escape hatch, but ordinary learning never passes
        it.
        """

        words = [
            match.group(0).lower().strip(".'") for match in _WORD.finditer(text)
        ]
        words = [word for word in words if len(word) >= 2]
        counts = Counter(words)
        ordered_atoms = sorted(
            counts, key=lambda word: (words.index(word), -counts[word])
        )
        units: List[str] = []
        seen = set()

        def add(unit: str) -> bool:
            if unit in seen:
                return False
            seen.add(unit)
            units.append(unit)
            return limit is not None and len(units) >= max(1, int(limit))

        for word in ordered_atoms:
            if add(word):
                return units
        for width in (2, 3):
            for index in range(0, max(0, len(words) - width + 1)):
                unit = "::".join(words[index : index + width])
                if len(set(words[index : index + width])) == 1:
                    continue
                if add(unit):
                    return units
        return units

    @staticmethod
    def _segments(text: str) -> List[str]:
        clean = text.replace("\x00", "").strip()
        if not clean:
            return []
        segments = [part.strip() for part in _SEGMENT.split(clean) if part.strip()]
        return segments or [clean]

    def _check_growth(self, estimated_bytes: int) -> None:
        if self.growth_guard is None:
            return
        if not self.growth_guard(max(1, int(estimated_bytes))):
            self.growth_pauses += 1
            raise SubstrateResourcePause(
                "neural substrate growth paused at the host resource reserve"
            )

    def _ensure_neuron(
        self, label: str, timestamp: float, region: str = "semantic"
    ) -> str:
        neuron_id = hashlib.sha256(
            ("%s:%s" % (region, label)).encode("utf-8")
        ).hexdigest()[:24]
        record = self.neurons.get(neuron_id)
        if record is None:
            record = {
                "id": neuron_id,
                "neuron_id": neuron_id,
                "label": label,
                "region": region,
                "activation": 0.0,
                "importance": 0.1,
                "uncertainty": 0.5,
                "exposures": 0,
                "created_at": timestamp,
                "last_activated_at": timestamp,
                "aliases": [],
            }
            self.neurons[neuron_id] = record
            self.neuron_vectors[neuron_id] = self.space.symbol(
                "%s-neuron:%s" % (region, label)
            )
            self.growth_events += 1
        record["activation"] = min(
            1.0, float(record["activation"]) * 0.68 + 0.32
        )
        record["importance"] = min(
            1.0, float(record["importance"]) + 1.0 / (10.0 + record["exposures"])
        )
        record["exposures"] += 1
        record["last_activated_at"] = timestamp
        return neuron_id

    def vector_for_labels(self, labels: Sequence[str]) -> torch.Tensor:
        if not labels:
            return self.space.symbol("empty-assembly")
        vectors = []
        for index, label in enumerate(labels):
            neuron_id = hashlib.sha256(
                ("semantic:%s" % label).encode("utf-8")
            ).hexdigest()[:24]
            neuron = self.neuron_vectors.get(
                neuron_id, self.space.symbol("semantic-neuron:" + label)
            )
            role = self.space.symbol("position:%d" % index)
            vectors.append(
                self.space.permute(self.space.bind(neuron, role), steps=index + 1)
            )
        return self.space.bundle(vectors)

    def vector_for_text(self, text: str) -> torch.Tensor:
        return self.vector_for_labels(self.extract_concepts(text))

    @staticmethod
    def _ternary(latent: float) -> int:
        return 1 if latent >= 0.25 else (-1 if latent <= -0.25 else 0)

    def _strengthen_synapse(
        self,
        source: str,
        target: str,
        timestamp: float,
        *,
        kind: str,
        amount: float,
    ) -> None:
        if source == target:
            return
        synapse_id = "%s>%s:%s" % (source, target, kind)
        synapse = self.synapses.get(synapse_id)
        if synapse is None:
            synapse = {
                "id": synapse_id,
                "source_id": source,
                "target_id": target,
                "kind": kind,
                "latent_weight": 0.0,
                "effective_weight": 0,
                "eligibility": 0.0,
                "plasticity": 1.0,
                "uses": 0,
                "stability": 0.0,
                "last_updated_at": timestamp,
            }
            self.synapses[synapse_id] = synapse
            self.growth_events += 1
        synapse["uses"] += 1
        synapse["eligibility"] = min(
            1.0, float(synapse.get("eligibility", 0.0)) * 0.8 + abs(amount)
        )
        synapse["stability"] = min(
            20.0, float(synapse.get("stability", 0.0)) + 0.01
        )
        local_rate = float(synapse.get("plasticity", 1.0)) / (
            1.0 + float(synapse["stability"])
        )
        latent = max(
            -1.0,
            min(1.0, float(synapse["latent_weight"]) + amount * local_rate),
        )
        synapse["latent_weight"] = latent
        synapse["effective_weight"] = self._ternary(latent)
        synapse["last_updated_at"] = timestamp

    def _wire_local_structure(
        self, neuron_ids: Sequence[str], timestamp: float
    ) -> None:
        # Local sparse wiring scales linearly with experience length.  It
        # preserves order and co-activation without constructing an O(n²)
        # clique for a long document.
        for index, source in enumerate(neuron_ids):
            for distance, target in enumerate(
                neuron_ids[index + 1 : index + 9], start=1
            ):
                self._strengthen_synapse(
                    source,
                    target,
                    timestamp,
                    kind="co-activates",
                    amount=0.09 / float(distance),
                )
                self._strengthen_synapse(
                    target,
                    source,
                    timestamp,
                    kind="associates",
                    amount=0.045 / float(distance),
                )

    def _store_assembly(
        self,
        text: str,
        *,
        kind: str,
        source: str,
        source_label: str,
        retain_source_text: bool,
        importance: float,
        timestamp: float,
        child_ids: Optional[Sequence[str]] = None,
    ) -> Tuple[Dict[str, Any], torch.Tensor, bool]:
        labels = self.extract_concepts(text)
        if not labels:
            labels = ["empty-experience"]
        estimated = (
            len(labels) * (self.space.dimensions * 4 + 640)
            + len(labels) * 16 * 320
            + self.space.dimensions * 4
        )
        self._check_growth(estimated)
        neuron_ids = [
            self._ensure_neuron(label, timestamp, "semantic") for label in labels
        ]
        vector = self.vector_for_labels(labels)
        fingerprint = hashlib.sha256(text.encode("utf-8")).hexdigest()
        existing = next(
            (
                assembly
                for assembly in self.assemblies
                if assembly["fingerprint"] == fingerprint
            ),
            None,
        )
        created = existing is None
        if existing is None:
            assembly_id = hashlib.sha256(
                ("assembly:" + fingerprint).encode("ascii")
            ).hexdigest()[:24]
            record: Dict[str, Any] = {
                "id": assembly_id,
                "assembly_neuron_id": assembly_id,
                "fingerprint": fingerprint,
                "neuron_ids": neuron_ids,
                "concept_ids": neuron_ids,
                "child_assembly_ids": list(child_ids or []),
                "kind": kind,
                "source": source,
                "confidence": 0.5,
                "importance": max(0.0, min(float(importance), 1.0)),
                "rehearsals": 1,
                "created_at": timestamp,
                "last_recalled_at": None,
                "source_label": source_label,
            }
            if retain_source_text:
                record["source_text"] = text
            self.assemblies.append(record)
            self.assembly_vectors[assembly_id] = vector
            self.neurons[assembly_id] = {
                "id": assembly_id,
                "neuron_id": assembly_id,
                "label": source_label or kind,
                "region": "assembly",
                "activation": 0.3,
                "importance": record["importance"],
                "uncertainty": 0.5,
                "exposures": 1,
                "created_at": timestamp,
                "last_activated_at": timestamp,
                "aliases": [],
            }
            self.neuron_vectors[assembly_id] = vector
            self.growth_events += 1
            for member in neuron_ids:
                self._strengthen_synapse(
                    assembly_id,
                    member,
                    timestamp,
                    kind="contains",
                    amount=0.5,
                )
                self._strengthen_synapse(
                    member,
                    assembly_id,
                    timestamp,
                    kind="participates",
                    amount=0.35,
                )
            for child_id in child_ids or []:
                self._strengthen_synapse(
                    assembly_id,
                    child_id,
                    timestamp,
                    kind="composes",
                    amount=0.55,
                )
        else:
            record = existing
            record["rehearsals"] += 1
            record["last_recalled_at"] = timestamp
            record["importance"] = min(
                1.0, float(record["importance"]) + 0.03
            )
            assembly_id = str(record["id"])
            node = self.neurons.get(assembly_id)
            if node is not None:
                node["activation"] = min(
                    1.0, float(node["activation"]) * 0.7 + 0.3
                )
                node["exposures"] += 1
                node["last_activated_at"] = timestamp
        self._wire_local_structure(neuron_ids, timestamp)
        return record, vector, created

    def learn(
        self,
        text: str,
        kind: str = "knowledge",
        source: str = "conversation",
        source_label: str = "",
        retain_source_text: bool = False,
        importance: float = 0.5,
    ) -> Dict[str, Any]:
        timestamp = _now()
        segments = self._segments(text)
        if not segments:
            segments = ["empty-experience"]
        child_ids: List[str] = []
        created_count = 0
        nearest = -1.0
        for segment in segments:
            record, vector, created = self._store_assembly(
                segment,
                kind=kind,
                source=source,
                source_label=source_label,
                retain_source_text=retain_source_text,
                importance=importance,
                timestamp=timestamp,
            )
            child_ids.append(str(record["id"]))
            created_count += int(created)
            comparisons = [
                candidate
                for assembly_id, candidate in self.assembly_vectors.items()
                if assembly_id != record["id"]
            ]
            if comparisons:
                nearest = max(
                    nearest,
                    max(self.space.similarity(vector, item) for item in comparisons),
                )

        if len(segments) > 1:
            parent, parent_vector, parent_created = self._store_assembly(
                text,
                kind=kind,
                source=source,
                source_label=source_label,
                retain_source_text=retain_source_text,
                importance=min(1.0, importance + 0.08),
                timestamp=timestamp,
                child_ids=child_ids,
            )
            primary = parent
            primary_vector = parent_vector
            created_count += int(parent_created)
        else:
            primary = next(
                item for item in self.assemblies if item["id"] == child_ids[0]
            )
            primary_vector = self.assembly_vectors[str(primary["id"])]

        novelty = max(0.0, min(1.0, 1.0 - max(0.0, nearest)))
        return {
            "idea_id": primary["id"],
            "assembly_id": primary["id"],
            "vector": primary_vector,
            "novelty": novelty,
            "concept_ids": list(primary["neuron_ids"]),
            "neuron_ids": list(primary["neuron_ids"]),
            "labels": [
                self.neurons[item]["label"]
                for item in primary["neuron_ids"]
                if item in self.neurons
            ],
            "assemblies_created": created_count,
        }

    def recall_vector(
        self,
        cue: torch.Tensor,
        limit: Optional[int] = None,
        *,
        workspace_slots: Optional[int] = None,
    ) -> Tuple[torch.Tensor, List[Dict[str, Any]]]:
        """Recall by similarity followed by recurrent spreading activation.

        Ordinary callers leave ``limit`` unset.  All assemblies above the
        adaptive activation floor participate in the bundled neural signal;
        working-memory size affects salience, not permanent addressability.
        """

        if not self.assembly_vectors:
            return cue, []
        scored = sorted(
            (
                (self.space.similarity(cue, vector), assembly_id, vector)
                for assembly_id, vector in self.assembly_vectors.items()
            ),
            reverse=True,
        )
        positive = [item for item in scored if item[0] > 0]
        if not positive:
            return cue, []
        best = positive[0][0]
        slots = max(1, int(workspace_slots or max(8, math.sqrt(len(positive)))))
        adaptive_floor = max(0.01, best / (2.0 + math.log2(slots + 1.0)))
        selected = [item for item in positive if item[0] >= adaptive_floor]
        if limit is not None:
            selected = selected[: max(1, int(limit))]

        seeds: Dict[str, float] = {
            assembly_id: float(score) for score, assembly_id, _ in selected
        }
        adjacency: Dict[str, List[Tuple[str, int]]] = defaultdict(list)
        incoming: Dict[str, int] = defaultdict(int)
        inhibitory_edges = 0
        eligible_edges = 0
        for synapse in self.synapses.values():
            effective = int(synapse.get("effective_weight", 0))
            if effective == 0:
                continue
            if effective not in {-1, 1}:
                raise ValueError(
                    "live substrate synapses must have exact ternary weights"
                )
            source = str(synapse["source_id"])
            target = str(synapse["target_id"])
            if source not in self.neurons or target not in self.neurons:
                continue
            eligible_edges += 1
            inhibitory_edges += int(effective < 0)
            adjacency[str(synapse["source_id"])].append(
                (target, effective)
            )
            incoming[target] += 1

        # This synchronous recurrent system is a contraction: each target's
        # signed drive is normalized by its incoming degree and damped by 0.52.
        # It therefore settles without a fixed hop ceiling. Crucially, the
        # contribution of each live edge is exactly source_activity * {-1,+1};
        # the latent master magnitude is learning state and never scales the
        # forward signal.
        activation = dict(seeds)
        admitted = set(seeds)
        propagation_rounds = 0
        inhibitory_signals = 0
        convergence_delta = 0.0
        while activation:
            workspace_pressure = max(
                1.0, len(activation) / max(1.0, float(slots))
            )
            pressure_floor = max(
                1e-5,
                adaptive_floor
                * (
                    0.02
                    + 0.03 * math.log2(workspace_pressure + 1.0)
                ),
            )
            drives: Dict[str, float] = defaultdict(float)
            round_inhibitory = 0
            for source, source_activation in activation.items():
                if abs(source_activation) <= 1e-12:
                    continue
                for target, weight in adjacency.get(source, []):
                    contribution = source_activation * float(weight)
                    drives[target] += contribution
                    round_inhibitory += int(contribution < 0.0)

            candidates = admitted.union(seeds).union(drives)
            settled: Dict[str, float] = {}
            for target in candidates:
                recurrent = (
                    0.52
                    * drives.get(target, 0.0)
                    / float(max(1, incoming.get(target, 0)))
                )
                value = max(
                    -1.0,
                    min(1.0, seeds.get(target, 0.0) + recurrent),
                )
                if target in admitted or abs(value) >= pressure_floor:
                    admitted.add(target)
                    settled[target] = value

            convergence_delta = max(
                (
                    abs(settled.get(key, 0.0) - activation.get(key, 0.0))
                    for key in set(settled).union(activation)
                ),
                default=0.0,
            )
            inhibitory_signals = round_inhibitory
            activation = settled
            propagation_rounds += 1
            if convergence_delta <= max(1e-7, pressure_floor * 0.001):
                break

        by_id = {assembly["id"]: assembly for assembly in self.assemblies}
        active = sorted(activation.items(), key=lambda item: item[1], reverse=True)
        vectors = [cue]
        weights = [1.0]
        recalled = []
        timestamp = _now()
        for assembly_id, score in active:
            vector = self.assembly_vectors.get(assembly_id)
            if vector is None or score <= 0:
                continue
            vectors.append(vector)
            weights.append(max(0.01, float(score)))
            assembly = by_id.get(assembly_id)
            if assembly is None:
                continue
            assembly["last_recalled_at"] = timestamp
            node = self.neurons.get(assembly_id)
            if node is not None:
                node["activation"] = min(
                    1.0, float(node["activation"]) * 0.6 + min(0.4, score)
                )
                node["last_activated_at"] = timestamp
            recalled.append(
                {
                    "idea_id": assembly_id,
                    "assembly_id": assembly_id,
                    "score": float(score),
                    "concept_ids": list(assembly["neuron_ids"]),
                    "neuron_ids": list(assembly["neuron_ids"]),
                }
            )
        signal = self.space.weighted_bundle(vectors, weights)
        # Exposed as inspection metadata only; it is derived from this recall
        # operation and is never an authoritative memory record.
        self._last_recall_rounds = propagation_rounds
        suppressed = sum(
            1
            for assembly_id, seed in seeds.items()
            if seed >= adaptive_floor
            and activation.get(assembly_id, 0.0) < adaptive_floor
        )
        self._last_recall_audit = {
            "rule": "effective-ternary-recurrent-settling",
            "exactTernaryContribution": True,
            "latentMagnitudeUsed": False,
            "eligibleEdges": eligible_edges,
            "inhibitoryEdges": inhibitory_edges,
            "inhibitorySignals": inhibitory_signals,
            "suppressedAssemblies": suppressed,
            "settledRounds": propagation_rounds,
            "convergenceDelta": convergence_delta,
            "damping": 0.52,
            "fanInNormalization": True,
            "activeNeuralNodes": len(activation),
            "activationByAssembly": {
                key: float(value)
                for key, value in sorted(activation.items())
                if key in self.assembly_vectors
            },
        }
        return signal, recalled

    def decay(self, amount: float = 0.002) -> None:
        amount = max(0.0, min(float(amount), 1.0))
        for neuron in self.neurons.values():
            neuron["activation"] *= 1.0 - amount
            neuron["uncertainty"] = min(
                1.0,
                float(neuron["uncertainty"])
                + amount / (1.0 + neuron["exposures"]),
            )
        for synapse in self.synapses.values():
            stability = float(synapse.get("stability", 0.0))
            synapse["latent_weight"] *= (
                1.0
                - amount
                * (1.0 - 0.8 * min(1.0, stability))
                / (1.0 + synapse["uses"])
            )
            synapse["eligibility"] = float(
                synapse.get("eligibility", 0.0)
            ) * (1.0 - amount)
            synapse["stability"] = max(0.0, stability - amount * 0.05)
            synapse["effective_weight"] = self._ternary(
                float(synapse["latent_weight"])
            )

    def metadata(self, include_records: bool = True) -> Dict[str, Any]:
        metadata: Dict[str, Any] = {
            "schema": self.SCHEMA,
            "dimensions": self.space.dimensions,
            "seed": self.space.seed,
            "cardinality_limit": None,
            "authoritative_memory": "neurons-assemblies-ternary-synapses",
            "growth_events": self.growth_events,
            "growth_pauses": self.growth_pauses,
        }
        if self.persistence_manifest is not None:
            metadata["persistence"] = dict(self.persistence_manifest)
        if include_records:
            metadata.update(
                {
                    "neurons": list(self.neurons.values()),
                    "assemblies": self.assemblies,
                    "synapses": list(self.synapses.values()),
                    "neuron_vector_ids": list(self.neuron_vectors),
                    "assembly_vector_ids": list(self.assembly_vectors),
                }
            )
        return metadata

    @staticmethod
    def _canonical_json(value: Any) -> bytes:
        return json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")

    @staticmethod
    def _bucket(kind: str, record_id: str) -> str:
        return hashlib.sha256(
            ("%s:%s" % (kind, record_id)).encode("utf-8")
        ).hexdigest()[:1]

    @staticmethod
    def _safe_store_path(root: Path, relative: str) -> Path:
        if (
            not relative
            or relative.startswith(("/", "\\"))
            or "\\" in relative
            or any(part in {"", ".", ".."} for part in relative.split("/"))
        ):
            raise ValueError("substrate shard manifest contains an unsafe path")
        resolved = (root / relative).resolve()
        try:
            resolved.relative_to(root.resolve())
        except ValueError as error:
            raise ValueError("substrate shard path escapes its store") from error
        return resolved

    @staticmethod
    def _file_sha256(path: Path) -> str:
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            while True:
                block = handle.read(1024 * 1024)
                if not block:
                    break
                digest.update(block)
        return digest.hexdigest()

    def _store_json_blob(
        self, root: Path, value: Dict[str, Any]
    ) -> Dict[str, Any]:
        payload = self._canonical_json(value)
        checksum = hashlib.sha256(payload).hexdigest()
        relative = "blobs/%s.json" % checksum
        path = self._safe_store_path(root, relative)
        if not path.exists():
            self._check_growth(len(payload) + 4096)
            atomic_write_bytes(path, payload)
        return {
            "path": relative,
            "sha256": checksum,
            "bytes": len(payload),
        }

    def _store_tensor_blob(
        self,
        root: Path,
        tensors: Dict[str, torch.Tensor],
        *,
        reusable: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        state_digest = hashlib.sha256()
        for name, value in sorted(tensors.items()):
            contiguous = value.detach().cpu().contiguous()
            state_digest.update(name.encode("utf-8"))
            state_digest.update(str(tuple(contiguous.shape)).encode("ascii"))
            state_digest.update(str(contiguous.dtype).encode("ascii"))
            state_digest.update(contiguous.numpy().tobytes())
        state_checksum = state_digest.hexdigest()
        if (
            isinstance(reusable, dict)
            and reusable.get("stateSha256") == state_checksum
        ):
            reused_path = self._safe_store_path(
                root, str(reusable.get("path", ""))
            )
            if (
                reused_path.is_file()
                and self._file_sha256(reused_path)
                == str(reusable.get("sha256", ""))
                and reused_path.stat().st_size == int(reusable.get("bytes", -1))
            ):
                return dict(reusable)
        estimated = 4096 + sum(
            int(value.numel() * value.element_size())
            for value in tensors.values()
        )
        self._check_growth(estimated)
        blobs = root / "blobs"
        blobs.mkdir(parents=True, exist_ok=True)
        descriptor, temporary_name = tempfile.mkstemp(
            prefix="substrate.", suffix=".safetensors.tmp", dir=str(blobs)
        )
        os.close(descriptor)
        os.unlink(temporary_name)
        temporary = Path(temporary_name)
        try:
            atomic_save_tensors(
                temporary,
                tensors,
                metadata={
                    "format": _SUBSTRATE_STORE_FORMAT,
                    "formatVersion": str(_SUBSTRATE_STORE_VERSION),
                },
            )
            checksum = self._file_sha256(temporary)
            relative = "blobs/%s.safetensors" % checksum
            destination = self._safe_store_path(root, relative)
            size = temporary.stat().st_size
            if destination.exists():
                temporary.unlink()
            else:
                os.replace(str(temporary), str(destination))
            return {
                "path": relative,
                "sha256": checksum,
                "bytes": size,
                "stateSha256": state_checksum,
            }
        finally:
            if temporary.exists():
                temporary.unlink()

    @staticmethod
    def _groups(
        kind: str,
        records: Iterable[Tuple[str, Any]],
        records_per_shard: int,
    ) -> Iterable[Tuple[str, int, List[Tuple[str, Any]]]]:
        buckets: Dict[str, List[Tuple[str, Any]]] = defaultdict(list)
        for record_id, record in records:
            buckets[NeuralSubstrate._bucket(kind, str(record_id))].append(
                (str(record_id), record)
            )
        for bucket in sorted(buckets):
            ordered = sorted(buckets[bucket], key=lambda item: item[0])
            for offset in range(0, len(ordered), records_per_shard):
                yield bucket, offset // records_per_shard, ordered[
                    offset : offset + records_per_shard
                ]

    def save_sharded(
        self,
        root: Path,
        *,
        records_per_shard: int = 512,
    ) -> Dict[str, Any]:
        """Write a deterministic content-addressed substrate generation.

        Only bounded shards are materialized. Existing content blobs are
        reused, so a growth update rewrites the affected hash bucket rather
        than the complete sparse substrate.
        """

        if records_per_shard < 1:
            raise ValueError("records_per_shard must be positive")
        store = Path(root).resolve()
        (store / "blobs").mkdir(parents=True, exist_ok=True)
        (store / "generations").mkdir(parents=True, exist_ok=True)
        shards: List[Dict[str, Any]] = []
        prior_shards: Dict[Tuple[str, str, int], Dict[str, Any]] = {}
        if self.persistence_manifest is not None:
            try:
                prior_path = self._safe_store_path(
                    store,
                    str(self.persistence_manifest.get("generationManifest", "")),
                )
                prior_generation = read_json(prior_path)
                prior_shards = {
                    (
                        str(item.get("kind", "")),
                        str(item.get("bucket", "")),
                        int(item.get("part", -1)),
                    ): item
                    for item in prior_generation.get("shards", [])
                    if isinstance(item, dict)
                }
            except (FileNotFoundError, OSError, ValueError, TypeError):
                # A complete new generation is still safe. Any malformed prior
                # pointer will be rejected on load rather than trusted for reuse.
                prior_shards = {}

        for kind, values, vectors in (
            ("neurons", self.neurons.items(), self.neuron_vectors),
            (
                "assemblies",
                (
                    (
                        str(item["id"]),
                        {**item, "__persistence_ordinal": index},
                    )
                    for index, item in enumerate(self.assemblies)
                ),
                self.assembly_vectors,
            ),
        ):
            for bucket, part, group in self._groups(
                kind, values, records_per_shard
            ):
                ids = [record_id for record_id, _record in group]
                json_blob = self._store_json_blob(
                    store,
                    {
                        "kind": kind,
                        "ids": ids,
                        "records": [record for _record_id, record in group],
                        "vectorIds": [
                            record_id for record_id in ids if record_id in vectors
                        ],
                    },
                )
                vector_ids = [
                    record_id for record_id in ids if record_id in vectors
                ]
                tensor_blob = (
                    self._store_tensor_blob(
                        store,
                        {
                            "vectors": torch.stack(
                                [
                                    vectors[record_id].detach().cpu()
                                    for record_id in vector_ids
                                ]
                            )
                        },
                        reusable=prior_shards.get(
                            (kind, bucket, part), {}
                        ).get("tensors"),
                    )
                    if vector_ids
                    else None
                )
                shards.append(
                    {
                        "kind": kind,
                        "bucket": bucket,
                        "part": part,
                        "count": len(group),
                        "records": json_blob,
                        "tensors": tensor_blob,
                    }
                )

        for bucket, part, group in self._groups(
            "synapses", self.synapses.items(), records_per_shard
        ):
            structures = []
            for record_id, record in group:
                structures.append(
                    {
                        key: value
                        for key, value in record.items()
                        if key not in _SYNAPSE_TENSOR_FIELDS
                    }
                )
                if structures[-1].get("id") != record_id:
                    raise ValueError("synapse mapping key does not match record id")
            json_blob = self._store_json_blob(
                store,
                {
                    "kind": "synapses",
                    "ids": [record_id for record_id, _record in group],
                    "records": structures,
                },
            )
            tensors = {
                "latent_weight": torch.tensor(
                    [
                        float(record.get("latent_weight", 0.0))
                        for _record_id, record in group
                    ],
                    dtype=torch.float64,
                ),
                "effective_weight": torch.tensor(
                    [
                        int(record.get("effective_weight", 0))
                        for _record_id, record in group
                    ],
                    dtype=torch.int8,
                ),
                "eligibility": torch.tensor(
                    [
                        float(record.get("eligibility", 0.0))
                        for _record_id, record in group
                    ],
                    dtype=torch.float64,
                ),
                "plasticity": torch.tensor(
                    [
                        float(record.get("plasticity", 1.0))
                        for _record_id, record in group
                    ],
                    dtype=torch.float64,
                ),
                "uses": torch.tensor(
                    [
                        int(record.get("uses", 0))
                        for _record_id, record in group
                    ],
                    dtype=torch.int64,
                ),
                "stability": torch.tensor(
                    [
                        float(record.get("stability", 0.0))
                        for _record_id, record in group
                    ],
                    dtype=torch.float64,
                ),
                "last_updated_at": torch.tensor(
                    [
                        float(record.get("last_updated_at", 0.0))
                        for _record_id, record in group
                    ],
                    dtype=torch.float64,
                ),
            }
            reusable_synapse = prior_shards.get(
                ("synapses", bucket, part), {}
            ).get("tensors")
            tensor_blob = self._store_tensor_blob(
                store,
                tensors,
                reusable=reusable_synapse,
            )
            shards.append(
                {
                    "kind": "synapses",
                    "bucket": bucket,
                    "part": part,
                    "count": len(group),
                    "records": json_blob,
                    "tensors": tensor_blob,
                }
            )

        generation_body = {
            "format": _SUBSTRATE_STORE_FORMAT,
            "formatVersion": _SUBSTRATE_STORE_VERSION,
            "schema": self.SCHEMA,
            "dimensions": self.space.dimensions,
            "seed": self.space.seed,
            "growthEvents": self.growth_events,
            "growthPauses": self.growth_pauses,
            "recordsPerShard": int(records_per_shard),
            "counts": {
                "neurons": len(self.neurons),
                "assemblies": len(self.assemblies),
                "synapses": len(self.synapses),
            },
            "shards": sorted(
                shards,
                key=lambda item: (
                    str(item["kind"]),
                    str(item["bucket"]),
                    int(item["part"]),
                ),
            ),
        }
        content_checksum = hashlib.sha256(
            self._canonical_json(generation_body)
        ).hexdigest()
        generation = {
            **generation_body,
            "contentSha256": content_checksum,
        }
        generation_relative = "generations/%s/manifest.json" % content_checksum
        generation_path = self._safe_store_path(store, generation_relative)
        generation_bytes = self._canonical_json(generation)
        generation_sha = hashlib.sha256(generation_bytes).hexdigest()
        if not generation_path.exists():
            self._check_growth(len(generation_bytes) + 4096)
            atomic_write_bytes(generation_path, generation_bytes)
        pointer = {
            "format": _SUBSTRATE_STORE_FORMAT,
            "formatVersion": _SUBSTRATE_STORE_VERSION,
            "activeGeneration": content_checksum,
            "generationManifest": generation_relative,
            "generationManifestSha256": generation_sha,
            "counts": dict(generation_body["counts"]),
            "shardCount": len(shards),
            "contentSha256": content_checksum,
        }
        atomic_write_json(store / "manifest.json", pointer)
        self.persistence_manifest = pointer
        return dict(pointer)

    @classmethod
    def load_sharded(
        cls,
        root: Path,
        metadata: Dict[str, Any],
        *,
        growth_guard: Optional[Callable[[int], bool]] = None,
    ) -> "NeuralSubstrate":
        store = Path(root).resolve()
        persistence = metadata.get("persistence")
        pointer = (
            dict(persistence)
            if isinstance(persistence, dict)
            else read_json(store / "manifest.json")
        )
        if (
            pointer.get("format") != _SUBSTRATE_STORE_FORMAT
            or int(pointer.get("formatVersion", 0))
            != _SUBSTRATE_STORE_VERSION
        ):
            raise ValueError("unsupported neural substrate shard format")
        generation_path = cls._safe_store_path(
            store, str(pointer.get("generationManifest", ""))
        )
        active_generation = str(pointer.get("activeGeneration", ""))
        if (
            len(active_generation) != 64
            or any(
                character not in "0123456789abcdef"
                for character in active_generation
            )
            or str(pointer.get("generationManifest", ""))
            != "generations/%s/manifest.json" % active_generation
        ):
            raise ValueError("substrate generation identity is invalid")
        generation_bytes = generation_path.read_bytes()
        if hashlib.sha256(generation_bytes).hexdigest() != str(
            pointer.get("generationManifestSha256", "")
        ):
            raise ValueError("substrate generation manifest checksum mismatch")
        generation = json.loads(generation_bytes.decode("utf-8"))
        if (
            not isinstance(generation, dict)
            or generation.get("format") != _SUBSTRATE_STORE_FORMAT
            or int(generation.get("formatVersion", 0))
            != _SUBSTRATE_STORE_VERSION
            or generation.get("schema") != cls.SCHEMA
        ):
            raise ValueError("substrate generation manifest is incompatible")
        content_body = {
            key: value
            for key, value in generation.items()
            if key != "contentSha256"
        }
        content_checksum = hashlib.sha256(
            cls._canonical_json(content_body)
        ).hexdigest()
        if (
            content_checksum != str(generation.get("contentSha256", ""))
            or content_checksum != str(pointer.get("activeGeneration", ""))
            or content_checksum != str(pointer.get("contentSha256", ""))
        ):
            raise ValueError("substrate generation content checksum mismatch")
        if (
            int(generation["dimensions"]) != int(metadata.get("dimensions", -1))
            or int(generation["seed"]) != int(metadata.get("seed", -1))
            or generation.get("counts") != pointer.get("counts")
        ):
            raise ValueError("substrate generation does not match engine metadata")
        substrate = cls(
            dimensions=int(generation["dimensions"]),
            seed=int(generation["seed"]),
            growth_guard=growth_guard,
        )
        substrate.growth_events = int(generation.get("growthEvents", 0))
        substrate.growth_pauses = int(generation.get("growthPauses", 0))

        for shard in generation.get("shards", []):
            if not isinstance(shard, dict):
                raise ValueError("substrate shard entry is invalid")
            record_spec = shard.get("records")
            if not isinstance(record_spec, dict):
                raise ValueError("substrate record shard is invalid")
            record_path = cls._safe_store_path(
                store, str(record_spec.get("path", ""))
            )
            if str(record_spec.get("path", "")) != "blobs/%s.json" % str(
                record_spec.get("sha256", "")
            ):
                raise ValueError("substrate record shard identity is invalid")
            if (
                cls._file_sha256(record_path)
                != str(record_spec.get("sha256", ""))
                or record_path.stat().st_size != int(record_spec.get("bytes", -1))
            ):
                raise ValueError("substrate record shard checksum mismatch")
            payload = json.loads(record_path.read_text("utf-8"))
            kind = str(shard.get("kind", ""))
            records = payload.get("records", [])
            if len(records) != int(shard.get("count", -1)):
                raise ValueError("substrate record shard count mismatch")

            tensor_values: Dict[str, torch.Tensor] = {}
            tensor_spec = shard.get("tensors")
            if tensor_spec is not None:
                if not isinstance(tensor_spec, dict):
                    raise ValueError("substrate tensor shard is invalid")
                tensor_path = cls._safe_store_path(
                    store, str(tensor_spec.get("path", ""))
                )
                if str(
                    tensor_spec.get("path", "")
                ) != "blobs/%s.safetensors" % str(
                    tensor_spec.get("sha256", "")
                ):
                    raise ValueError("substrate tensor shard identity is invalid")
                if (
                    cls._file_sha256(tensor_path)
                    != str(tensor_spec.get("sha256", ""))
                    or tensor_path.stat().st_size
                    != int(tensor_spec.get("bytes", -1))
                ):
                    raise ValueError("substrate tensor shard checksum mismatch")
                tensor_values = load_tensors(tensor_path, device="cpu")

            if kind == "neurons":
                ids = [str(item) for item in payload.get("ids", [])]
                if ids != [str(item.get("id", "")) for item in records]:
                    raise ValueError("neuron shard identifiers do not match")
                substrate.neurons.update(
                    {record_id: dict(record) for record_id, record in zip(ids, records)}
                )
                vector_ids = [
                    str(item) for item in payload.get("vectorIds", [])
                ]
                vectors = tensor_values.get("vectors")
                if vectors is not None:
                    if vectors.shape[0] != len(vector_ids):
                        raise ValueError("neuron vector shard count mismatch")
                    substrate.neuron_vectors.update(
                        {
                            record_id: vectors[index].detach().cpu()
                            for index, record_id in enumerate(vector_ids)
                        }
                    )
            elif kind == "assemblies":
                ids = [str(item) for item in payload.get("ids", [])]
                if ids != [str(item.get("id", "")) for item in records]:
                    raise ValueError("assembly shard identifiers do not match")
                substrate.assemblies.extend(dict(record) for record in records)
                vector_ids = [
                    str(item) for item in payload.get("vectorIds", [])
                ]
                vectors = tensor_values.get("vectors")
                if vectors is not None:
                    if vectors.shape[0] != len(vector_ids):
                        raise ValueError("assembly vector shard count mismatch")
                    substrate.assembly_vectors.update(
                        {
                            record_id: vectors[index].detach().cpu()
                            for index, record_id in enumerate(vector_ids)
                        }
                    )
            elif kind == "synapses":
                ids = [str(item) for item in payload.get("ids", [])]
                if ids != [str(item.get("id", "")) for item in records]:
                    raise ValueError("synapse shard identifiers do not match")
                for field in _SYNAPSE_TENSOR_FIELDS:
                    values = tensor_values.get(field)
                    if values is None or values.numel() != len(ids):
                        raise ValueError(
                            "synapse tensor shard is missing " + field
                        )
                for index, (record_id, record) in enumerate(zip(ids, records)):
                    restored = dict(record)
                    restored.update(
                        {
                            "latent_weight": float(
                                tensor_values["latent_weight"][index].item()
                            ),
                            "effective_weight": int(
                                tensor_values["effective_weight"][index].item()
                            ),
                            "eligibility": float(
                                tensor_values["eligibility"][index].item()
                            ),
                            "plasticity": float(
                                tensor_values["plasticity"][index].item()
                            ),
                            "uses": int(tensor_values["uses"][index].item()),
                            "stability": float(
                                tensor_values["stability"][index].item()
                            ),
                            "last_updated_at": float(
                                tensor_values["last_updated_at"][index].item()
                            ),
                        }
                    )
                    if restored["effective_weight"] not in {-1, 0, 1}:
                        raise ValueError(
                            "substrate shard contains a non-ternary live synapse"
                        )
                    substrate.synapses[record_id] = restored
            else:
                raise ValueError("unknown substrate shard kind")

        expected_counts = generation.get("counts", {})
        observed_counts = {
            "neurons": len(substrate.neurons),
            "assemblies": len(substrate.assemblies),
            "synapses": len(substrate.synapses),
        }
        if observed_counts != expected_counts:
            raise ValueError("substrate shard generation count mismatch")
        substrate.assemblies.sort(
            key=lambda item: int(item.get("__persistence_ordinal", 0))
        )
        for item in substrate.assemblies:
            item.pop("__persistence_ordinal", None)
        substrate.persistence_manifest = dict(pointer)
        return substrate

    def tensor_state(self, prefix: str = "substrate.") -> Dict[str, torch.Tensor]:
        tensors: Dict[str, torch.Tensor] = {}
        if self.neuron_vectors:
            tensors[prefix + "neuron_vectors"] = torch.stack(
                [self.neuron_vectors[key] for key in self.neuron_vectors]
            )
        if self.assembly_vectors:
            tensors[prefix + "assembly_vectors"] = torch.stack(
                [self.assembly_vectors[key] for key in self.assembly_vectors]
            )
        return tensors

    @classmethod
    def from_state(
        cls,
        metadata: Dict[str, Any],
        tensors: Dict[str, torch.Tensor],
        prefix: str = "substrate.",
    ) -> "NeuralSubstrate":
        if metadata.get("schema") != cls.SCHEMA:
            raise ValueError(
                "incompatible beta neural memory; stable v1 requires a new brain"
            )
        substrate = cls(
            dimensions=int(metadata["dimensions"]),
            seed=int(metadata["seed"]),
        )
        substrate.neurons = {
            item["id"]: dict(item) for item in metadata.get("neurons", [])
        }
        substrate.assemblies = [
            dict(item) for item in metadata.get("assemblies", [])
        ]
        substrate.synapses = {
            item["id"]: dict(item) for item in metadata.get("synapses", [])
        }
        substrate.growth_events = int(metadata.get("growth_events", 0))
        substrate.growth_pauses = int(metadata.get("growth_pauses", 0))
        neuron_ids = metadata.get("neuron_vector_ids", [])
        neuron_tensor = tensors.get(prefix + "neuron_vectors")
        if neuron_tensor is not None:
            substrate.neuron_vectors = {
                neuron_id: neuron_tensor[index].detach().cpu()
                for index, neuron_id in enumerate(neuron_ids)
            }
        assembly_ids = metadata.get("assembly_vector_ids", [])
        assembly_tensor = tensors.get(prefix + "assembly_vectors")
        if assembly_tensor is not None:
            substrate.assembly_vectors = {
                assembly_id: assembly_tensor[index].detach().cpu()
                for index, assembly_id in enumerate(assembly_ids)
            }
        return substrate


# Source compatibility only.  The stable engine imports ``NeuralSubstrate``;
# external beta checkpoints remain intentionally incompatible.
ConceptMemory = NeuralSubstrate
