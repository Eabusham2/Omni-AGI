"""Distributed neural substrate with VSA binding and sparse ternary synapses.

The stable OmniCortex format does not keep an authoritative "idea database"
beside the neural state.  Concepts, experiences, and higher-order ideas are
represented as neuron assemblies connected by plastic synapses.  The
``concepts``/``ideas``/``relations`` properties at the bottom of the class are
compatibility views over that same substrate for the desktop inspector.
"""

import hashlib
import math
import re
import time
from collections import Counter, defaultdict
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import torch
from torch.nn import functional as F


_WORD = re.compile(r"[A-Za-z0-9_][A-Za-z0-9_+\-'.]{1,95}")
_SEGMENT = re.compile(r"(?<=[.!?])\s+|\n+")


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

    ``max_*`` arguments are accepted only so older callers fail softly while
    moving to the stable interface.  They are deliberately ignored: structural
    growth is governed by a host-resource callback rather than cardinality.
    """

    SCHEMA = "neural-substrate-1"

    def __init__(
        self,
        dimensions: int = 256,
        seed: int = 7,
        max_concepts: Optional[int] = None,
        max_ideas: Optional[int] = None,
        max_relations: Optional[int] = None,
        growth_guard: Optional[Callable[[int], bool]] = None,
    ):
        del max_concepts, max_ideas, max_relations
        self.space = HypervectorSpace(dimensions, seed)
        self.neurons: Dict[str, Dict[str, Any]] = {}
        self.neuron_vectors: Dict[str, torch.Tensor] = {}
        self.assemblies: List[Dict[str, Any]] = []
        self.assembly_vectors: Dict[str, torch.Tensor] = {}
        self.synapses: Dict[str, Dict[str, Any]] = {}
        self.growth_events = 0
        self.growth_pauses = 0
        self.growth_guard = growth_guard

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

        activation: Dict[str, float] = {
            assembly_id: float(score) for score, assembly_id, _ in selected
        }
        adjacency: Dict[str, List[Tuple[str, float]]] = defaultdict(list)
        for synapse in self.synapses.values():
            effective = int(synapse.get("effective_weight", 0))
            if effective == 0:
                continue
            adjacency[str(synapse["source_id"])].append(
                (
                    str(synapse["target_id"]),
                    effective
                    * min(1.0, abs(float(synapse.get("latent_weight", 0.0)))),
                )
            )
        frontier = dict(activation)
        propagation_rounds = 0
        while frontier:
            propagated: Dict[str, float] = {}
            # There is deliberately no model-defined hop ceiling here.
            # Recurrent activity settles when damping and interference leave
            # no materially stronger activation.  As active assemblies exceed
            # the global workspace, the admission floor rises continuously,
            # modelling working-memory pressure without making the substrate
            # permanently address-limited.
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
            changed = False
            for source, source_activation in frontier.items():
                for target, weight in adjacency.get(source, []):
                    if target not in self.assembly_vectors:
                        continue
                    signal = source_activation * weight * 0.52
                    if signal <= pressure_floor:
                        continue
                    previous = max(
                        activation.get(target, 0.0),
                        propagated.get(target, 0.0),
                    )
                    # A recurrent edge is scheduled only when it materially
                    # changes the settled state. Cycles therefore converge
                    # naturally as the damped signal falls below interference.
                    if signal <= previous + max(1e-6, pressure_floor * 0.02):
                        continue
                    propagated[target] = signal
                    changed = True
            if not changed:
                break
            for target, signal in propagated.items():
                activation[target] = max(activation.get(target, 0.0), signal)
            frontier = propagated
            propagation_rounds += 1

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

    def metadata(self) -> Dict[str, Any]:
        return {
            "schema": self.SCHEMA,
            "dimensions": self.space.dimensions,
            "seed": self.space.seed,
            "cardinality_limit": None,
            "authoritative_memory": "neurons-assemblies-ternary-synapses",
            "growth_events": self.growth_events,
            "growth_pauses": self.growth_pauses,
            "neurons": list(self.neurons.values()),
            "assemblies": self.assemblies,
            "synapses": list(self.synapses.values()),
            "neuron_vector_ids": list(self.neuron_vectors),
            "assembly_vector_ids": list(self.assembly_vectors),
        }

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
