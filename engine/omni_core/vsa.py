"""Vector-symbolic concept and idea memory."""

import hashlib
import re
import time
import uuid
from collections import Counter
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import torch
from torch.nn import functional as F


_WORD = re.compile(r"[A-Za-z0-9_][A-Za-z0-9_+\-'.]{1,47}")


def _now() -> float:
    return time.time()


class HypervectorSpace:
    """Deterministic bipolar hypervectors with bind, bundle, and permutation."""

    def __init__(self, dimensions: int = 256, seed: int = 7):
        if dimensions < 16:
            raise ValueError("hypervector dimensions must be at least 16")
        self.dimensions = dimensions
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
    def permute(vector: torch.Tensor, steps: int = 1) -> torch.Tensor:
        """Encode sequence/role position with an exactly invertible rotation."""

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


class ConceptMemory:
    """Stores compositional ideas and sparse typed relations.

    Long-term recall returns only hypervectors.  Raw source text is retained
    solely for the explicitly selected ``total-recall`` recipe.
    """

    def __init__(
        self,
        dimensions: int = 256,
        seed: int = 7,
        max_concepts: int = 50000,
        max_ideas: int = 10000,
        max_relations: int = 1000000,
    ):
        self.space = HypervectorSpace(dimensions, seed)
        self.max_concepts = int(max_concepts)
        self.max_ideas = int(max_ideas)
        self.max_relations = int(max_relations)
        self.concepts: Dict[str, Dict[str, Any]] = {}
        self.concept_vectors: Dict[str, torch.Tensor] = {}
        self.ideas: List[Dict[str, Any]] = []
        self.idea_vectors: Dict[str, torch.Tensor] = {}
        self.relations: Dict[str, Dict[str, Any]] = {}
        self.capacity_expansions = 0

    def expand_capacities(
        self,
        *,
        concepts: Optional[int] = None,
        ideas: Optional[int] = None,
        relations: Optional[int] = None,
    ) -> Dict[str, int]:
        """Raise sparse-memory ceilings without changing existing identities.

        The caller owns resource policy.  This method never shrinks a memory
        and therefore cannot silently evict state while expanding it.
        """

        before = {
            "concepts": self.max_concepts,
            "ideas": self.max_ideas,
            "relations": self.max_relations,
        }
        if concepts is not None:
            self.max_concepts = max(self.max_concepts, int(concepts))
        if ideas is not None:
            self.max_ideas = max(self.max_ideas, int(ideas))
        if relations is not None:
            self.max_relations = max(self.max_relations, int(relations))
        after = {
            "concepts": self.max_concepts,
            "ideas": self.max_ideas,
            "relations": self.max_relations,
        }
        if after != before:
            self.capacity_expansions += 1
        return after

    @staticmethod
    def extract_concepts(text: str, limit: int = 48) -> List[str]:
        words = [match.group(0).lower().strip(".'") for match in _WORD.finditer(text)]
        words = [word for word in words if len(word) >= 2]
        counts = Counter(words)
        ranked = sorted(counts, key=lambda word: (-counts[word], words.index(word)))
        atoms = ranked[:limit]
        pairs = []
        for left, right in zip(words, words[1:]):
            if left != right:
                pairs.append("%s::%s" % (left, right))
        for pair in pairs:
            if pair not in atoms:
                atoms.append(pair)
            if len(atoms) >= limit:
                break
        return atoms

    def _ensure_concept(self, label: str, timestamp: float) -> str:
        concept_id = hashlib.sha256(label.encode("utf-8")).hexdigest()[:20]
        record = self.concepts.get(concept_id)
        if record is None:
            if len(self.concepts) >= self.max_concepts:
                # Reuse the least important concept rather than exceed the
                # configured resource boundary.
                concept_id = min(
                    self.concepts,
                    key=lambda item: (
                        self.concepts[item]["importance"],
                        self.concepts[item]["last_activated_at"],
                    ),
                )
                return concept_id
            record = {
                "id": concept_id,
                "label": label,
                "activation": 0.0,
                "importance": 0.1,
                "uncertainty": 0.5,
                "exposures": 0,
                "created_at": timestamp,
                "last_activated_at": timestamp,
                "aliases": [],
            }
            self.concepts[concept_id] = record
            self.concept_vectors[concept_id] = self.space.symbol("concept:" + label)
        record["activation"] = min(1.0, float(record["activation"]) * 0.7 + 0.3)
        record["importance"] = min(
            1.0, float(record["importance"]) + 1.0 / (10.0 + record["exposures"])
        )
        record["exposures"] += 1
        record["last_activated_at"] = timestamp
        return concept_id

    def vector_for_labels(self, labels: Sequence[str]) -> torch.Tensor:
        if not labels:
            return self.space.symbol("empty-idea")
        vectors = []
        for index, label in enumerate(labels):
            concept_id = hashlib.sha256(label.encode("utf-8")).hexdigest()[:20]
            concept = self.concept_vectors.get(
                concept_id, self.space.symbol("concept:" + label)
            )
            role = self.space.symbol("position:%d" % min(index, 15))
            bound = self.space.bind(concept, role)
            vectors.append(self.space.permute(bound, steps=index + 1))
        return self.space.bundle(vectors)

    def vector_for_text(self, text: str) -> torch.Tensor:
        return self.vector_for_labels(self.extract_concepts(text))

    def _add_relations(self, concept_ids: Sequence[str], timestamp: float) -> None:
        for index, source in enumerate(concept_ids):
            for target in concept_ids[index + 1 : min(index + 5, len(concept_ids))]:
                if source == target:
                    continue
                pair = sorted((source, target))
                relation_id = "%s>%s:co-occurs" % (pair[0], pair[1])
                relation = self.relations.get(relation_id)
                if relation is None:
                    if len(self.relations) >= self.max_relations:
                        evicted_id = min(
                            self.relations,
                            key=lambda key: (
                                self.relations[key].get("stability", 0.0),
                                self.relations[key]["uses"],
                                abs(self.relations[key]["latent_weight"]),
                            ),
                        )
                        self.relations.pop(evicted_id, None)
                    relation = {
                        "id": relation_id,
                        "source_id": pair[0],
                        "target_id": pair[1],
                        "kind": "co-occurs",
                        "latent_weight": 0.0,
                        "effective_weight": 0,
                        "uses": 0,
                        "stability": 0.0,
                        "last_updated_at": timestamp,
                    }
                    self.relations[relation_id] = relation
                relation["uses"] += 1
                relation["stability"] = min(
                    1.0, float(relation.get("stability", 0.0)) + 0.01
                )
                relation["latent_weight"] = min(
                    1.0, float(relation["latent_weight"]) + 0.08
                )
                latent = float(relation["latent_weight"])
                relation["effective_weight"] = 1 if latent >= 0.33 else 0
                relation["last_updated_at"] = timestamp

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
        labels = self.extract_concepts(text)
        if not labels:
            labels = ["empty-experience"]
        concept_ids = [self._ensure_concept(label, timestamp) for label in labels]
        idea_vector = self.vector_for_labels(labels)
        nearest = -1.0
        if self.idea_vectors:
            nearest = max(
                self.space.similarity(idea_vector, vector)
                for vector in self.idea_vectors.values()
            )
        novelty = max(0.0, min(1.0, 1.0 - max(0.0, nearest)))
        fingerprint = hashlib.sha256(text.encode("utf-8")).hexdigest()
        duplicate = next(
            (idea for idea in self.ideas if idea["fingerprint"] == fingerprint),
            None,
        )
        if duplicate is not None:
            duplicate["rehearsals"] += 1
            duplicate["last_recalled_at"] = timestamp
            duplicate["importance"] = min(
                1.0, float(duplicate["importance"]) + 0.03
            )
            idea_id = duplicate["id"]
        else:
            idea_id = uuid.uuid4().hex
            record: Dict[str, Any] = {
                "id": idea_id,
                "fingerprint": fingerprint,
                "concept_ids": concept_ids,
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
            if len(self.ideas) >= self.max_ideas:
                evicted = min(
                    self.ideas,
                    key=lambda item: (
                        item["importance"],
                        item["rehearsals"],
                        item["created_at"],
                    ),
                )
                self.ideas.remove(evicted)
                self.idea_vectors.pop(evicted["id"], None)
            self.ideas.append(record)
            self.idea_vectors[idea_id] = idea_vector
        self._add_relations(concept_ids, timestamp)
        return {
            "idea_id": idea_id,
            "vector": idea_vector,
            "novelty": novelty,
            "concept_ids": concept_ids,
            "labels": labels,
        }

    def recall_vector(
        self, cue: torch.Tensor, limit: int = 4
    ) -> Tuple[torch.Tensor, List[Dict[str, Any]]]:
        if not self.idea_vectors:
            return cue, []
        scored = sorted(
            (
                (self.space.similarity(cue, vector), idea_id, vector)
                for idea_id, vector in self.idea_vectors.items()
            ),
            reverse=True,
        )[: max(1, int(limit))]
        vectors = [cue]
        recalled = []
        by_id = {idea["id"]: idea for idea in self.ideas}
        for score, idea_id, vector in scored:
            if score <= 0:
                continue
            vectors.extend([vector] * max(1, int(round(score * 3))))
            idea = by_id.get(idea_id)
            if idea is not None:
                idea["last_recalled_at"] = _now()
                recalled.append(
                    {
                        "idea_id": idea_id,
                        "score": score,
                        "concept_ids": list(idea["concept_ids"]),
                    }
                )
        return self.space.bundle(vectors), recalled

    def decay(self, amount: float = 0.002) -> None:
        amount = max(0.0, min(float(amount), 1.0))
        for concept in self.concepts.values():
            concept["activation"] *= 1.0 - amount
            concept["uncertainty"] = min(
                1.0,
                float(concept["uncertainty"])
                + amount / (1.0 + concept["exposures"]),
            )
        for relation in self.relations.values():
            stability = float(relation.get("stability", 0.0))
            relation["latent_weight"] *= 1.0 - amount * (1.0 - 0.8 * stability) / (
                1.0 + relation["uses"]
            )
            relation["stability"] = max(0.0, stability - amount * 0.05)
            latent = float(relation["latent_weight"])
            relation["effective_weight"] = (
                1 if latent >= 0.33 else (-1 if latent <= -0.33 else 0)
            )

    def metadata(self) -> Dict[str, Any]:
        return {
            "dimensions": self.space.dimensions,
            "seed": self.space.seed,
            "max_concepts": self.max_concepts,
            "max_ideas": self.max_ideas,
            "max_relations": self.max_relations,
            "capacity_expansions": self.capacity_expansions,
            "concepts": list(self.concepts.values()),
            "ideas": self.ideas,
            "relations": list(self.relations.values()),
            "concept_vector_ids": list(self.concept_vectors),
            "idea_vector_ids": list(self.idea_vectors),
        }

    def tensor_state(self, prefix: str = "memory.") -> Dict[str, torch.Tensor]:
        tensors: Dict[str, torch.Tensor] = {}
        if self.concept_vectors:
            tensors[prefix + "concept_vectors"] = torch.stack(
                [self.concept_vectors[key] for key in self.concept_vectors]
            )
        if self.idea_vectors:
            tensors[prefix + "idea_vectors"] = torch.stack(
                [self.idea_vectors[key] for key in self.idea_vectors]
            )
        return tensors

    @classmethod
    def from_state(
        cls,
        metadata: Dict[str, Any],
        tensors: Dict[str, torch.Tensor],
        prefix: str = "memory.",
    ) -> "ConceptMemory":
        memory = cls(
            dimensions=int(metadata["dimensions"]),
            seed=int(metadata["seed"]),
            max_concepts=int(metadata.get("max_concepts", 50000)),
            max_ideas=int(metadata.get("max_ideas", 10000)),
            max_relations=int(metadata.get("max_relations", 1000000)),
        )
        memory.concepts = {
            item["id"]: dict(item) for item in metadata.get("concepts", [])
        }
        memory.ideas = [dict(item) for item in metadata.get("ideas", [])]
        memory.relations = {
            item["id"]: dict(item) for item in metadata.get("relations", [])
        }
        memory.capacity_expansions = int(metadata.get("capacity_expansions", 0))
        concept_ids = metadata.get("concept_vector_ids", [])
        concept_tensor = tensors.get(prefix + "concept_vectors")
        if concept_tensor is not None:
            memory.concept_vectors = {
                concept_id: concept_tensor[index].detach().cpu()
                for index, concept_id in enumerate(concept_ids)
            }
        idea_ids = metadata.get("idea_vector_ids", [])
        idea_tensor = tensors.get(prefix + "idea_vectors")
        if idea_tensor is not None:
            memory.idea_vectors = {
                idea_id: idea_tensor[index].detach().cpu()
                for index, idea_id in enumerate(idea_ids)
            }
        return memory
