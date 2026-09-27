"""Continuous, reversible settling for OmniCortex neural experience.

This module does not store a second semantic memory.  The authoritative
knowledge remains in ``NeuralSubstrate`` neurons, assemblies, and synapses.
It tracks transient neural afterimages and continuously remeasures how
accessible and plastic learned assemblies are under the current activity.
There are deliberately no one-way ``fading``/``working``/``lasting`` stages.
"""

from __future__ import annotations

import math
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, ClassVar, Dict, List, Mapping, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F

from .paged_assembly_view import PagedAssemblyView
from .paged_neuron_metadata import PagedNeuronMetadata


def _iso_now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _bounded(value: Any, default: float = 0.0) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        number = default
    if not math.isfinite(number):
        number = default
    return max(0.0, min(1.0, number))


@dataclass
class OrganicMemoryLifecycle:
    """Transient, continuously scored state derived from the neural substrate.

    ``afterimage_vectors`` are disk-persisted neural activity traces.  Their
    metadata contains no source text or token IDs.  An afterimage can warm,
    cool, disappear after becoming influence-free, and be recreated by later
    use.  None of those changes deletes or permanently classifies its learned
    substrate assembly.
    """

    SCHEMA = "organic-memory-lifecycle-2"
    EPISODE_MERGE_COSINE: ClassVar[float] = 0.985

    @classmethod
    def _same_episode_vector(
        cls, left: torch.Tensor, right: torch.Tensor
    ) -> bool:
        first = left.float().reshape(-1)
        second = right.float().reshape(-1)
        if first.shape != second.shape:
            return False
        similarity = float(
            F.cosine_similarity(
                first.reshape(1, -1), second.reshape(1, -1)
            ).item()
        )
        return math.isfinite(similarity) and similarity >= cls.EPISODE_MERGE_COSINE

    cycle: int = 0
    afterimage_vectors: List[torch.Tensor] = field(default_factory=list)
    afterimage_items: List[Dict[str, Any]] = field(default_factory=list)
    active_focus: List[Dict[str, Any]] = field(default_factory=list)
    settled_experiences: int = 0
    settled_rest_cycles: int = 0
    expired_afterimages: int = 0
    reinforcement_events: int = 0
    last_settled_at: str = ""
    last_source: str = ""
    # Operational migration cursor only; neither knowledge nor persisted
    # memory state. Stable writers never introduce v1 memory_stage labels.
    _legacy_clean_memory: Any = field(default=None, init=False, repr=False, compare=False)
    _legacy_clean_assemblies: Any = field(default=None, init=False, repr=False, compare=False)
    _legacy_clean_neurons: Any = field(default=None, init=False, repr=False, compare=False)
    _legacy_clean_count: int = field(default=0, init=False, repr=False, compare=False)

    # Temporary aliases keep older callers/loaders operational while the
    # persisted and inspection schema uses the accurate "afterimage" name.
    @property
    def scratch_vectors(self) -> List[torch.Tensor]:
        return self.afterimage_vectors

    @scratch_vectors.setter
    def scratch_vectors(self, values: List[torch.Tensor]) -> None:
        self.afterimage_vectors = values

    @property
    def scratch_items(self) -> List[Dict[str, Any]]:
        return self.afterimage_items

    @scratch_items.setter
    def scratch_items(self, values: List[Dict[str, Any]]) -> None:
        self.afterimage_items = values

    @staticmethod
    def _record_for(memory: Any, assembly_id: str) -> Optional[Dict[str, Any]]:
        indexed = getattr(memory, "assembly_by_id", None)
        if isinstance(indexed, Mapping):
            return indexed.get(assembly_id)
        return next(
            (
                record
                for record in memory.assemblies
                if str(record.get("id", "")) == assembly_id
            ),
            None,
        )

    @staticmethod
    def _synapses_for(memory: Any, assembly_id: str) -> Sequence[Dict[str, Any]]:
        connected_records = getattr(memory.synapses, "connected_records", None)
        if callable(connected_records):
            return connected_records((assembly_id,))
        return tuple(
            synapse
            for synapse in memory.synapses.values()
            if str(synapse.get("source_id", "")) == assembly_id
            or str(synapse.get("target_id", "")) == assembly_id
        )

    @staticmethod
    def discard_legacy_stage_labels(memory: Any) -> int:
        """Remove inert v1 categories while preserving every learned record."""

        removed = 0
        if not isinstance(memory.assemblies, PagedAssemblyView):
            for record in memory.assemblies:
                if "memory_stage" in record:
                    record.pop("memory_stage", None)
                    removed += 1
        # Current paged schemas reject the removed label at admission; a
        # read-only page cannot carry it and needs no corpus scan/mutation.
        if not isinstance(memory.neurons, PagedNeuronMetadata):
            for node in memory.neurons.values():
                if "memory_stage" in node:
                    node.pop("memory_stage", None)
                    removed += 1
        return removed

    def _discard_legacy_stage_labels_for_settle(self, memory: Any) -> None:
        """Migrate old labels once, then inspect only newly appended assemblies.

        Loaded legacy records are fully cleaned on their first settling cycle.
        Stable substrate writers create new neurons without the removed label;
        replacing either backing collection forces another full migration.
        """

        records = memory.assemblies
        neurons = memory.neurons
        if isinstance(records, PagedAssemblyView):
            self._legacy_clean_memory = memory
            self._legacy_clean_assemblies = records
            self._legacy_clean_neurons = neurons
            self._legacy_clean_count = len(records)
            return
        if (
            self._legacy_clean_memory is memory
            and self._legacy_clean_assemblies is records
            and self._legacy_clean_neurons is neurons
            and len(records) >= self._legacy_clean_count
        ):
            for record in records[self._legacy_clean_count :]:
                record.pop("memory_stage", None)
        else:
            self.discard_legacy_stage_labels(memory)
        self._legacy_clean_memory = memory
        self._legacy_clean_assemblies = records
        self._legacy_clean_neurons = neurons
        self._legacy_clean_count = len(records)

    def _update_focus(self, memory: Any, assembly_id: str, salience: float) -> None:
        assembly_nodes = []
        # Outside the legacy raw-activation epoch, effective_activation is
        # exactly zero for every neuron absent from the active attention set.
        # A learned substrate may hold millions of cold assemblies, so read
        # only the active assembly IDs plus this cycle's explicitly salient
        # assembly. Keep the full traversal for legacy checkpoints whose raw
        # activations are still authoritative, and for foreign/mock memories
        # that do not expose the indexed substrate contract.
        indexed_records = None
        active_ids = None
        if not bool(getattr(memory, "attention_legacy_raw_active", True)):
            candidate_index = getattr(memory, "assembly_by_id", None)
            candidate_active = getattr(memory, "attention_active_neuron_ids", None)
            if (
                isinstance(candidate_index, Mapping)
                and isinstance(candidate_active, (set, frozenset))
                and len(candidate_index) == len(memory.assemblies)
            ):
                indexed_records = candidate_index
                active_ids = candidate_active
        candidate_ids = (
            (identifier for identifier in active_ids | {assembly_id} if identifier in indexed_records)
            if indexed_records is not None and active_ids is not None
            else (str(record.get("id", "")) for record in memory.assemblies)
        )
        for candidate_id in candidate_ids:
            node = memory.neurons.get(candidate_id, {})
            activation = _bounded(memory.effective_activation(node), 0.0)
            if candidate_id == assembly_id:
                activation = max(activation, salience)
            if activation > 0.01:
                assembly_nodes.append((activation, candidate_id))
        if not assembly_nodes:
            self.active_focus = []
            return
        peak = max(value[0] for value in assembly_nodes)
        adaptive_floor = max(0.04, peak * 0.42)
        self.active_focus = [
            {
                "assemblyId": candidate_id,
                "activation": activation,
                "currentlyFiring": activation >= adaptive_floor,
            }
            for activation, candidate_id in sorted(assembly_nodes, reverse=True)
            if activation >= adaptive_floor
        ]

    def _measure_signals(
        self,
        memory: Any,
        assembly_id: str,
        *,
        salience_boost: float = 0.0,
        spike_rate: float = 0.0,
        record_index: Optional[Mapping[str, Dict[str, Any]]] = None,
        synapse_index: Optional[
            Mapping[str, Sequence[Dict[str, Any]]]
        ] = None,
        focus_index: Optional[Mapping[str, Mapping[str, Any]]] = None,
    ) -> Tuple[Dict[str, float], int, Sequence[Dict[str, Any]]]:
        """Measure one assembly against the current substrate and focus."""

        record = (
            record_index.get(assembly_id)
            if record_index is not None
            else self._record_for(memory, assembly_id)
        )
        rehearsals = max(1, int((record or {}).get("rehearsals", 1)))
        repetition = min(1.0, math.log1p(rehearsals) / math.log(6.0))
        related_synapses = (
            synapse_index.get(assembly_id, ())
            if synapse_index is not None
            else self._synapses_for(memory, assembly_id)
        )
        stability = min(
            1.0,
            sum(_bounded(item.get("stability"), 0.0) for item in related_synapses)
            / float(max(1, len(related_synapses))),
        )

        focus = (
            focus_index.get(assembly_id, {})
            if focus_index is not None
            else next(
                (
                    item
                    for item in self.active_focus
                    if str(item.get("assemblyId", "")) == assembly_id
                ),
                {},
            )
        )
        focus_activation = _bounded(focus.get("activation"), 0.0)
        node = memory.neurons.get(assembly_id, {})
        activation = _bounded(
            0.52 * _bounded(memory.effective_activation(node), 0.0)
            + 0.28 * focus_activation
            + 0.12 * _bounded(salience_boost)
            + 0.08 * _bounded(spike_rate)
        )
        synaptic_reuse = (
            sum(
                min(
                    1.0,
                    math.log1p(max(0, int(item.get("uses", 0))))
                    / math.log(8.0),
                )
                for item in related_synapses
            )
            / float(max(1, len(related_synapses)))
        )
        reuse = _bounded(0.65 * repetition + 0.35 * synaptic_reuse)

        live_directions = {
            (str(item.get("source_id", "")), str(item.get("target_id", "")))
            for item in related_synapses
            if int(item.get("effective_weight", 0)) != 0
        }
        recurrent_edges = sum(
            1
            for source_id, target_id in live_directions
            if (target_id, source_id) in live_directions
        )
        recurrent_support = recurrent_edges / float(max(1, len(live_directions)))
        related_ids = {
            endpoint
            for item in related_synapses
            for endpoint in (
                str(item.get("source_id", "")),
                str(item.get("target_id", "")),
            )
            if endpoint and endpoint != assembly_id
        }
        focus_values = (
            focus_index.values() if focus_index is not None else self.active_focus
        )
        related_activity = [
            _bounded(item.get("activation"), 0.0)
            for item in focus_values
            if str(item.get("assemblyId", "")) in related_ids
            and bool(item.get("currentlyFiring", False))
        ]
        coactive_support = (
            sum(related_activity) / float(len(related_activity))
            if related_activity
            else 0.0
        )
        recurrence = _bounded(
            0.65 * repetition
            + 0.20 * recurrent_support
            + 0.15 * coactive_support
        )

        competing = [
            _bounded(item.get("activation"), 0.0)
            for item in self.active_focus
            if str(item.get("assemblyId", "")) != assembly_id
            and str(item.get("assemblyId", "")) not in related_ids
            and bool(item.get("currentlyFiring", False))
        ]
        competitor_pressure = (
            sum(competing)
            / float(max(1.0, math.sqrt(max(1, len(competing))) * 2.0))
            if competing
            else 0.0
        )
        recall_audit = getattr(memory, "_last_recall_audit", {})
        if not isinstance(recall_audit, Mapping):
            recall_audit = {}
        inhibitory_ratio = _bounded(
            float(recall_audit.get("inhibitorySignals", 0))
            / float(max(1, int(recall_audit.get("eligibleEdges", 0))))
        )
        suppressed_ratio = _bounded(
            float(recall_audit.get("suppressedAssemblies", 0))
            / float(max(1, int(recall_audit.get("activeNeuralNodes", 0))))
        )
        interference = _bounded(
            0.55 * _bounded(competitor_pressure)
            + 0.25 * inhibitory_ratio
            + 0.20 * suppressed_ratio
        )
        return (
            {
                "activation": activation,
                "recurrence": recurrence,
                "reuse": reuse,
                "salience": _bounded(salience_boost),
                "stability": stability,
                "interference": interference,
                "decayPressure": _bounded(
                    1.0
                    - (
                        0.34 * stability
                        + 0.28 * reuse
                        + 0.24 * recurrence
                        + 0.14 * activation
                    )
                    + 0.18 * interference
                ),
            },
            rehearsals,
            related_synapses,
        )

    @staticmethod
    def _continuous_scores(
        *,
        signals: Mapping[str, float],
        importance: float,
        novelty: float,
        prediction_error: float,
        salience: float,
    ) -> Dict[str, float]:
        retention = _bounded(
            0.25 * _bounded(importance)
            + 0.14 * _bounded(signals.get("reuse"))
            + 0.11 * _bounded(signals.get("recurrence"))
            + 0.10 * _bounded(prediction_error)
            + 0.10 * _bounded(salience)
            + 0.07 * _bounded(novelty)
            + 0.10 * _bounded(signals.get("stability"))
            + 0.08 * _bounded(signals.get("activation"))
            + 0.05 * (1.0 - _bounded(signals.get("interference")))
        )
        plasticity = _bounded(
            0.28 * _bounded(novelty)
            + 0.25 * _bounded(prediction_error)
            + 0.18 * _bounded(salience)
            + 0.16 * _bounded(signals.get("activation"))
            + 0.13 * (1.0 - _bounded(signals.get("stability")))
        )
        reinforcement = _bounded(
            retention
            * (
                0.30 * _bounded(signals.get("recurrence"))
                + 0.25 * _bounded(signals.get("reuse"))
                + 0.20 * _bounded(signals.get("stability"))
                + 0.15 * _bounded(signals.get("activation"))
                + 0.10 * _bounded(salience)
            )
        )
        unfinished = _bounded(
            _bounded(prediction_error)
            * (0.55 + 0.45 * _bounded(salience))
            * (1.0 - 0.45 * _bounded(signals.get("stability")))
        )
        return {
            "retention": retention,
            "activity": _bounded(signals.get("activation")),
            "plasticity": plasticity,
            "reinforcement": reinforcement,
            "unfinished": unfinished,
        }

    @staticmethod
    def assess_retention_candidate(
        *,
        novelty: float = 0.0,
        reuse: float = 0.0,
        salience: float = 0.0,
        prediction_error: float = 0.0,
        rehearsals: int = 1,
        stability: float = 0.0,
        related_coactivation: float = 0.0,
        interference: float = 0.0,
        recurrence: float = 0.0,
        activation: float = 0.0,
        observations: int = 1,
    ) -> Dict[str, Any]:
        """Score continuous retention and slow replay without gating admission.

        Every accepted experience may first form generic fast episodic,
        temporal, recurrent, and afterimage state.  This pure scorer only
        measures how strongly that state should resist fading and how urgently
        it should receive later slow cortical replay.  It never decides whether
        a fast episode is allowed to exist and has no threshold, parser, source
        label, fact grammar, or manual memory command.
        """

        def positive_count(value: Any, default: int = 1) -> int:
            try:
                number = int(value)
            except (TypeError, ValueError, OverflowError):
                number = default
            return max(1, number)

        observation_count = positive_count(observations)
        rehearsal_count = positive_count(rehearsals)
        bounded = {
            "novelty": _bounded(novelty),
            "reuse": _bounded(reuse),
            "salience": _bounded(salience),
            "predictionError": _bounded(prediction_error),
            "stability": _bounded(stability),
            "relatedCoactivation": _bounded(related_coactivation),
            "interference": _bounded(interference),
            "recurrence": _bounded(recurrence),
            "activation": _bounded(activation),
        }

        # A single observation has no recurrence evidence merely by existing;
        # matching observations and rehearsal add smoothly diminishing support.
        observation_support = 1.0 - math.exp(
            -max(0, observation_count - 1) / 1.35
        )
        rehearsal_support = 1.0 - math.exp(
            -max(0, rehearsal_count - 1) / 1.70
        )
        exposure_strength = _bounded(
            0.34 * bounded["reuse"]
            + 0.24 * bounded["recurrence"]
            + 0.22 * observation_support
            + 0.20 * rehearsal_support
        )
        retention_score = _bounded(
            0.20 * exposure_strength
            + 0.18 * bounded["novelty"]
            + 0.17 * bounded["salience"]
            + 0.17 * bounded["predictionError"]
            + 0.10 * bounded["stability"]
            + 0.10 * bounded["relatedCoactivation"]
            + 0.08 * bounded["activation"]
            - 0.16 * bounded["interference"]
        )
        slow_replay_priority = _bounded(
            retention_score
            * (
                0.30 * exposure_strength
                + 0.25 * bounded["predictionError"]
                + 0.18 * bounded["novelty"]
                + 0.15 * bounded["salience"]
                + 0.12 * bounded["relatedCoactivation"]
            )
            * (0.65 + 0.35 * (1.0 - bounded["stability"]))
            * (1.0 - 0.65 * bounded["interference"])
        )
        fade_pressure = _bounded(
            (
                0.46 * bounded["interference"]
                + 0.22 * (1.0 - exposure_strength)
                + 0.18 * (1.0 - bounded["stability"])
                + 0.14 * (1.0 - bounded["salience"])
            )
            * (1.0 - 0.35 * bounded["relatedCoactivation"])
        )

        reasons: List[str] = []
        if observation_count > 1 or rehearsal_count > 1 or bounded["reuse"] > 0.0:
            reasons.append("repeated-or-rehearsed")
        if bounded["novelty"] >= 0.50:
            reasons.append("novelty")
        if bounded["salience"] >= 0.50:
            reasons.append("salience")
        if bounded["predictionError"] >= 0.50:
            reasons.append("prediction-error")
        if bounded["relatedCoactivation"] >= 0.50:
            reasons.append("related-coactivation")
        if bounded["stability"] >= 0.50:
            reasons.append("metaplastic-stability")
        if bounded["interference"] >= 0.50:
            reasons.append("unrelated-interference")
        if (
            observation_count == 1
            and rehearsal_count == 1
            and exposure_strength < 0.20
            and retention_score < 0.35
        ):
            reasons.append("weak-one-off")

        return {
            "retentionScore": retention_score,
            "slowReplayPriority": slow_replay_priority,
            "fadePressure": fade_pressure,
            "binaryAdmissionGate": False,
            "fastEpisodePolicy": "unconditional-generic",
            "manualConsolidationRequired": False,
            "reasons": reasons,
            "evidence": {
                **bounded,
                "observations": observation_count,
                "rehearsals": rehearsal_count,
                "observationSupport": observation_support,
                "rehearsalSupport": rehearsal_support,
                "exposureStrength": exposure_strength,
            },
        }

    @staticmethod
    def _write_scores(
        target: Dict[str, Any],
        *,
        scores: Mapping[str, float],
        signals: Mapping[str, float],
        timestamp: str,
    ) -> None:
        # Remove the v1 label if this record came from an older checkpoint.
        target.pop("memory_stage", None)
        target["memory_strength"] = _bounded(scores.get("retention"))
        target["retention_score"] = _bounded(scores.get("retention"))
        target["activity_score"] = _bounded(scores.get("activity"))
        target["plasticity_score"] = _bounded(scores.get("plasticity"))
        target["reinforcement_drive"] = _bounded(scores.get("reinforcement"))
        target["unfinished_score"] = _bounded(scores.get("unfinished"))
        target["unfinished"] = _bounded(scores.get("unfinished")) >= 0.35
        target["settling_signals"] = dict(signals)
        target["last_settled_at"] = timestamp

    def _write_assembly_scores(
        self,
        memory: Any,
        assembly_id: str,
        record: Dict[str, Any],
        *,
        scores: Mapping[str, float],
        signals: Mapping[str, float],
        timestamp: str,
    ) -> None:
        if isinstance(memory.assemblies, PagedAssemblyView):
            memory._edit_assembly_by_id(
                assembly_id,
                lambda current: self._write_scores(
                    current, scores=scores, signals=signals, timestamp=timestamp
                ),
            )
        else:
            self._write_scores(
                record, scores=scores, signals=signals, timestamp=timestamp
            )

    def _write_neuron_scores(
        self,
        memory: Any,
        neuron_id: str,
        *,
        scores: Mapping[str, float],
        signals: Mapping[str, float],
        timestamp: str,
    ) -> None:
        if isinstance(memory.neurons, PagedNeuronMetadata):
            memory.edit_neuron_by_id(
                neuron_id,
                lambda current: self._write_scores(
                    current, scores=scores, signals=signals, timestamp=timestamp
                ),
            )
        else:
            node = memory.neurons.get(neuron_id)
            if node is not None:
                self._write_scores(
                    node, scores=scores, signals=signals, timestamp=timestamp
                )

    def _rescore_afterimages(
        self,
        memory: Any,
        *,
        active_assembly_id: str,
        active_vector: Optional[torch.Tensor] = None,
        forgetting_rate: float,
        record_index: Mapping[str, Dict[str, Any]],
        synapse_index: Mapping[str, Sequence[Dict[str, Any]]],
        focus_index: Mapping[str, Mapping[str, Any]],
    ) -> None:
        """Warm or cool every existing afterimage from the live cycle state."""

        kept_vectors: List[torch.Tensor] = []
        kept_items: List[Dict[str, Any]] = []
        for vector, item in zip(self.afterimage_vectors, self.afterimage_items):
            candidate_id = str(item.get("assemblyId", ""))
            age = max(0, self.cycle - int(item.get("introducedCycle", self.cycle)))
            inactive_age = max(
                0, self.cycle - int(item.get("lastActiveCycle", self.cycle))
            )
            if candidate_id == active_assembly_id and active_vector is not None:
                # Distinct episodes may share a semantic field. This turn
                # rewarms only the matching afterimage, not every sibling.
                active = self._same_episode_vector(vector, active_vector)
            else:
                active = candidate_id == active_assembly_id or bool(
                    focus_index.get(candidate_id, {}).get("currentlyFiring", False)
                )
            prior_salience = _bounded(item.get("salience"), 0.35)
            live_salience = _bounded(
                prior_salience * (0.995 if active else 0.965)
            )
            signals, rehearsals, _ = self._measure_signals(
                memory,
                candidate_id,
                salience_boost=live_salience if active else 0.0,
                record_index=record_index,
                synapse_index=synapse_index,
                focus_index=focus_index,
            )
            record = record_index.get(candidate_id) or {}
            scores = self._continuous_scores(
                signals=signals,
                importance=_bounded(record.get("importance"), 0.0),
                novelty=_bounded(item.get("novelty"), 0.0),
                prediction_error=_bounded(item.get("predictionError"), 0.0),
                salience=live_salience,
            )

            # The target itself cools with dormancy. Recurrent use and stable
            # support lengthen the time constant, but no score is permanent.
            dormancy_scale = (
                6.0
                + 12.0 * _bounded(signals.get("reuse"))
                + 14.0 * _bounded(signals.get("stability"))
                + 8.0 * _bounded(scores.get("unfinished"))
            )
            dormancy = 1.0 if active else math.exp(-inactive_age / dormancy_scale)
            target = _bounded(scores["retention"] * dormancy)
            prior_strength = _bounded(item.get("strength"), target)
            responsiveness = 0.42 if active else 0.22
            strength = prior_strength + responsiveness * (target - prior_strength)
            pressure = min(
                0.14,
                0.012
                + max(0.0, float(forgetting_rate)) * 0.12
                + 0.045 * _bounded(signals.get("decayPressure")),
            )
            strength = _bounded(strength * (1.0 - pressure * (0.25 if active else 1.0)))

            if active:
                item["lastActiveCycle"] = self.cycle
                inactive_age = 0
            item.update(
                {
                    "strength": strength,
                    "salience": live_salience,
                    "rehearsals": rehearsals,
                    "signals": dict(signals),
                    "retentionScore": scores["retention"],
                    "activityScore": scores["activity"],
                    "plasticityScore": scores["plasticity"],
                    "reinforcementDrive": scores["reinforcement"],
                    "unfinishedScore": scores["unfinished"],
                    "unfinished": scores["unfinished"] >= 0.35,
                    "ageCycles": age,
                    "inactiveCycles": inactive_age,
                    "lastRescoredCycle": self.cycle,
                    "lastSettledCycle": self.cycle,
                }
            )
            if record:
                self._write_assembly_scores(
                    memory,
                    candidate_id,
                    record,
                    scores={**scores, "retention": strength},
                    signals=signals,
                    timestamp=self.last_settled_at,
                )
                if candidate_id in memory.neurons:
                    self._write_neuron_scores(
                        memory, candidate_id,
                        scores={**scores, "retention": strength},
                        signals=signals,
                        timestamp=self.last_settled_at,
                    )

            # Expiration only removes a transient trace. A later activation
            # can admit a fresh afterimage for the unchanged learned assembly.
            if (
                age >= 8
                and inactive_age >= 8
                and strength < 0.035
                and scores["unfinished"] < 0.20
            ):
                self.expired_afterimages += 1
                continue
            kept_vectors.append(vector)
            kept_items.append(item)
        self.afterimage_vectors = kept_vectors
        self.afterimage_items = kept_items

    def _admit_afterimage(
        self,
        vector: torch.Tensor,
        *,
        assembly_id: str,
        source: str,
        strength: float,
        salience: float,
        novelty: float,
        prediction_error: float,
        rehearsals: int,
        scores: Mapping[str, float],
        signals: Mapping[str, float],
    ) -> None:
        flat = vector.detach().cpu().float().reshape(-1)
        for index, item in enumerate(self.afterimage_items):
            if str(item.get("assemblyId", "")) != assembly_id:
                continue
            existing = self.afterimage_vectors[index].float().reshape(-1)
            if not self._same_episode_vector(existing, flat):
                continue
            self.afterimage_vectors[index] = F.normalize(
                (existing * 0.72 + flat * 0.28).reshape(1, -1), dim=-1
            )[0]
            item.update(
                {
                    "source": source,
                    "strength": _bounded(
                        0.62 * _bounded(item.get("strength"))
                        + 0.38 * strength
                        + 0.012 * min(4.0, math.log1p(rehearsals))
                    ),
                    "salience": _bounded(
                        0.55 * _bounded(item.get("salience")) + 0.45 * salience
                    ),
                    "novelty": novelty,
                    "predictionError": prediction_error,
                    "rehearsals": max(1, rehearsals),
                    "signals": dict(signals),
                    "retentionScore": _bounded(scores.get("retention")),
                    "activityScore": _bounded(scores.get("activity")),
                    "plasticityScore": _bounded(scores.get("plasticity")),
                    "reinforcementDrive": _bounded(scores.get("reinforcement")),
                    "unfinishedScore": _bounded(scores.get("unfinished")),
                    "unfinished": _bounded(scores.get("unfinished")) >= 0.35,
                    "lastActiveCycle": self.cycle,
                    "lastRescoredCycle": self.cycle,
                    "lastSettledCycle": self.cycle,
                    "inactiveCycles": 0,
                }
            )
            return
        self.afterimage_vectors.append(flat)
        self.afterimage_items.append(
            {
                "id": uuid.uuid4().hex,
                "assemblyId": assembly_id,
                "source": source,
                "strength": _bounded(strength),
                "salience": _bounded(salience),
                "novelty": _bounded(novelty),
                "predictionError": _bounded(prediction_error),
                "rehearsals": max(1, rehearsals),
                "signals": dict(signals),
                "retentionScore": _bounded(scores.get("retention")),
                "activityScore": _bounded(scores.get("activity")),
                "plasticityScore": _bounded(scores.get("plasticity")),
                "reinforcementDrive": _bounded(scores.get("reinforcement")),
                "unfinishedScore": _bounded(scores.get("unfinished")),
                "unfinished": _bounded(scores.get("unfinished")) >= 0.35,
                "introducedCycle": self.cycle,
                "lastActiveCycle": self.cycle,
                "lastRescoredCycle": self.cycle,
                "lastSettledCycle": self.cycle,
                "ageCycles": 0,
                "inactiveCycles": 0,
            }
        )

    def settle(
        self,
        *,
        memory: Any,
        router: Any,
        vector: torch.Tensor,
        assembly_id: str,
        source: str,
        salience: float,
        novelty: float,
        prediction_error: float,
        importance: float,
        spike_rate: float,
        forgetting_rate: float,
        long_term_threshold: float,
        resting: bool = False,
    ) -> Dict[str, Any]:
        """Settle an experience or prompt-free rest cycle continuously.

        ``long_term_threshold`` remains accepted for checkpoint/API
        compatibility, but it is not used as a stage boundary. This operation
        never proposes an action and never emits user-visible Ponder text.
        """

        del long_term_threshold
        self.cycle += 1
        self.last_settled_at = _iso_now()
        self.last_source = str(source)
        if resting:
            self.settled_rest_cycles += 1
        else:
            self.settled_experiences += 1

        salience = _bounded(salience, 0.4)
        novelty = _bounded(novelty, 0.5)
        prediction_error = _bounded(prediction_error, 0.0)
        importance = _bounded(importance, 0.5)
        spike_rate = _bounded(spike_rate, 0.0)
        self._discard_legacy_stage_labels_for_settle(memory)
        record_index = getattr(memory, "assembly_by_id", None)
        if not isinstance(record_index, Mapping):
            record_index = {
                str(record.get("id", "")): record
                for record in memory.assemblies
                if record.get("id")
            }
        # Focus must be current before selecting the bounded set of persisted
        # edges needed for this settling cycle.
        self._update_focus(memory, assembly_id, salience)
        relevant_assemblies = {
            assembly_id,
            *(
                str(item.get("assemblyId", ""))
                for item in self.afterimage_items
                if item.get("assemblyId")
            ),
            *(
                str(item.get("assemblyId", ""))
                for item in self.active_focus
                if item.get("assemblyId")
            ),
        }
        connected_records = getattr(memory.synapses, "connected_records", None)
        selected_synapses = (
            connected_records(relevant_assemblies)
            if callable(connected_records)
            else memory.synapses.values()
        )
        indexed_synapses: Dict[str, List[Dict[str, Any]]] = {}
        for synapse in selected_synapses:
            source_id = str(synapse.get("source_id", ""))
            target_id = str(synapse.get("target_id", ""))
            if source_id:
                indexed_synapses.setdefault(source_id, []).append(synapse)
            if target_id and target_id != source_id:
                indexed_synapses.setdefault(target_id, []).append(synapse)

        # In v1 the previous cycle's signals were merely decayed and a high
        # score could permanently leave the rescoreable trail.
        focus_index = {
            str(item.get("assemblyId", "")): item
            for item in self.active_focus
            if item.get("assemblyId")
        }
        self._rescore_afterimages(
            memory,
            active_assembly_id=assembly_id,
            active_vector=vector,
            forgetting_rate=forgetting_rate,
            record_index=record_index,
            synapse_index=indexed_synapses,
            focus_index=focus_index,
        )

        signals, rehearsals, related_synapses = self._measure_signals(
            memory,
            assembly_id,
            salience_boost=salience,
            spike_rate=spike_rate,
            record_index=record_index,
            synapse_index=indexed_synapses,
            focus_index=focus_index,
        )
        scores = self._continuous_scores(
            signals=signals,
            importance=importance,
            novelty=novelty,
            prediction_error=prediction_error,
            salience=salience,
        )

        continuous_decay = max(0.0, min(0.02, float(forgetting_rate) * 0.12))
        if continuous_decay:
            memory.decay(continuous_decay, synapses=related_synapses)
            router.synapses.decay_unused(continuous_decay * 0.45)

        record = record_index.get(assembly_id)
        if record is not None:
            self._write_assembly_scores(
                memory,
                assembly_id,
                record,
                scores=scores,
                signals=signals,
                timestamp=self.last_settled_at,
            )
            if assembly_id in memory.neurons:
                self._write_neuron_scores(
                    memory, assembly_id,
                    scores=scores,
                    signals=signals,
                    timestamp=self.last_settled_at,
                )
            self._admit_afterimage(
                vector,
                assembly_id=assembly_id,
                source=source,
                strength=max(0.025, scores["retention"]),
                salience=salience,
                novelty=novelty,
                prediction_error=prediction_error,
                rehearsals=rehearsals,
                scores=scores,
                signals=signals,
            )

            # Synaptic stabilization is proportional and reversible through
            # ordinary decay; it is never switched on by crossing a category.
            if scores["reinforcement"] > 0.0:
                self.reinforcement_events += 1
                for synapse in related_synapses:
                    synapse["stability"] = min(
                        20.0,
                        float(synapse.get("stability", 0.0))
                        + 0.025 * scores["reinforcement"],
                    )
                    active_trace = memory.effective_eligibility(synapse)
                    direction = (
                        -1.0
                        if active_trace > 0.0
                        and float(synapse.get("eligibility", 0.0)) < 0.0
                        else 1.0
                    )
                    synapse["eligibility"] = direction * min(
                        1.0,
                        active_trace
                        + 0.018 * salience * scores["reinforcement"],
                    )
                    memory.mark_attention_synapse(str(synapse.get("id", "")))

        recurring = next(
            (
                item
                for item in sorted(
                    self.afterimage_items,
                    key=lambda value: (
                        _bounded(value.get("unfinishedScore")),
                        _bounded(value.get("activityScore")),
                        _bounded(value.get("strength")),
                    ),
                    reverse=True,
                )
                if _bounded(item.get("unfinishedScore")) >= 0.35
                and self.cycle - int(item.get("lastActiveCycle", self.cycle)) >= 3
            ),
            None,
        )
        return {
            "automatic": True,
            "visiblePonderForced": False,
            "cycle": self.cycle,
            "retentionScore": scores["retention"],
            "activityScore": scores["activity"],
            "plasticityScore": scores["plasticity"],
            "reinforcementDrive": scores["reinforcement"],
            "unfinishedScore": scores["unfinished"],
            "strength": scores["retention"],
            "signals": signals,
            "unfinished": scores["unfinished"] >= 0.35,
            "fixedStage": False,
            "scoresRecomputedEachCycle": True,
            "activeAssemblyIds": [
                str(item.get("assemblyId", "")) for item in self.active_focus
            ],
            "recurringAssemblyId": (
                str(recurring.get("assemblyId", "")) if recurring else ""
            ),
            "afterimageCount": len(self.afterimage_items),
        }

    def state_tensors(self, prefix: str = "memory_lifecycle.") -> Dict[str, torch.Tensor]:
        if not self.afterimage_vectors:
            return {}
        return {
            prefix + "afterimage_vectors": torch.stack(
                [
                    value.detach().cpu().float().reshape(-1)
                    for value in self.afterimage_vectors
                ]
            )
        }

    def metadata(self) -> Dict[str, Any]:
        afterimages = [dict(item) for item in self.afterimage_items]
        return {
            "schema": self.SCHEMA,
            "cycle": self.cycle,
            "afterimageItems": afterimages,
            # Kept during the schema transition so v3 provenance guards that
            # predate the clearer name still reject non-empty transient state.
            "scratchItems": [dict(item) for item in afterimages],
            "activeFocus": [dict(item) for item in self.active_focus],
            "settledExperiences": self.settled_experiences,
            "settledRestCycles": self.settled_rest_cycles,
            "expiredAfterimages": self.expired_afterimages,
            "reinforcementEvents": self.reinforcement_events,
            "lastSettledAt": self.last_settled_at,
            "lastSource": self.last_source,
            "fixedStages": False,
            "scoresRecomputedEachCycle": True,
            "rawTextStored": False,
            "rawTokenIdsStored": False,
        }

    def clear_attention(self) -> Dict[str, int]:
        """Clear transient afterimages/focus without touching learned state."""

        cleared = {
            "afterimageItems": len(self.afterimage_items),
            "scratchItems": len(self.afterimage_items),
            "activeFocus": len(self.active_focus),
        }
        self.afterimage_vectors = []
        self.afterimage_items = []
        self.active_focus = []
        self.last_settled_at = ""
        self.last_source = ""
        return cleared

    @classmethod
    def from_state(
        cls,
        metadata: Optional[Mapping[str, Any]],
        tensors: Mapping[str, torch.Tensor],
        prefix: str = "memory_lifecycle.",
    ) -> "OrganicMemoryLifecycle":
        raw = dict(metadata or {})
        items = raw.get("afterimageItems", raw.get("scratchItems", []))
        lifecycle = cls(
            cycle=max(0, int(raw.get("cycle", 0))),
            afterimage_items=[
                dict(item) for item in items if isinstance(item, Mapping)
            ],
            active_focus=[
                dict(item)
                for item in raw.get("activeFocus", [])
                if isinstance(item, Mapping)
            ],
            settled_experiences=max(0, int(raw.get("settledExperiences", 0))),
            settled_rest_cycles=max(0, int(raw.get("settledRestCycles", 0))),
            expired_afterimages=max(
                0, int(raw.get("expiredAfterimages", raw.get("fadedItems", 0)))
            ),
            reinforcement_events=max(
                0,
                int(raw.get("reinforcementEvents", raw.get("promotedItems", 0))),
            ),
            last_settled_at=str(raw.get("lastSettledAt", "")),
            last_source=str(raw.get("lastSource", "")),
        )
        for item in lifecycle.afterimage_items:
            item.pop("stage", None)
            signals = item.get("signals", {})
            if not isinstance(signals, Mapping):
                signals = {}
            item.setdefault("retentionScore", _bounded(item.get("strength")))
            item.setdefault("activityScore", _bounded(signals.get("activation")))
            item.setdefault("plasticityScore", 0.0)
            item.setdefault("reinforcementDrive", 0.0)
            item.setdefault(
                "unfinishedScore", 0.35 if bool(item.get("unfinished")) else 0.0
            )
            item.setdefault("lastRescoredCycle", int(item.get("lastSettledCycle", 0)))
            item.setdefault(
                "inactiveCycles",
                max(
                    0,
                    lifecycle.cycle
                    - int(item.get("lastActiveCycle", lifecycle.cycle)),
                ),
            )
        stored = tensors.get(prefix + "afterimage_vectors")
        if stored is None:
            stored = tensors.get(prefix + "scratch_vectors")
        if lifecycle.afterimage_items:
            if stored is None or stored.ndim != 2:
                raise ValueError("afterimage metadata is missing its neural vectors")
            if int(stored.shape[0]) != len(lifecycle.afterimage_items):
                raise ValueError("afterimage metadata and neural vectors do not align")
            lifecycle.afterimage_vectors = [row.detach().cpu() for row in stored]
        elif stored is not None and int(stored.shape[0]) > 0:
            raise ValueError("afterimage neural vectors are missing metadata")
        return lifecycle

    def snapshot(self, memory: Any) -> Dict[str, Any]:
        self.discard_legacy_stage_labels(memory)
        retention_values = [
            _bounded(record.get("retention_score", record.get("memory_strength")))
            for record in memory.assemblies
            if "retention_score" in record or "memory_strength" in record
        ]
        tracked_ids = {
            str(item.get("assemblyId", ""))
            for item in self.afterimage_items
            if item.get("assemblyId")
        }
        connected_records = getattr(memory.synapses, "connected_records", None)
        selected_synapses = (
            connected_records(tracked_ids)
            if callable(connected_records)
            else memory.synapses.values()
        )
        reinforced_synapses = sum(
            1
            for synapse in selected_synapses
            if (
                str(synapse.get("source_id", "")) in tracked_ids
                or str(synapse.get("target_id", "")) in tracked_ids
            )
            and float(synapse.get("stability", 0.0) or 0.0) > 0.0
        )
        average_strength = (
            sum(_bounded(item.get("strength")) for item in self.afterimage_items)
            / float(len(self.afterimage_items))
            if self.afterimage_items
            else 0.0
        )
        average_retention = (
            sum(retention_values) / float(len(retention_values))
            if retention_values
            else 0.0
        )
        return {
            "activeFocus": {
                "count": len(self.active_focus),
                "items": [dict(item) for item in self.active_focus],
            },
            "afterimageTrail": {
                "count": len(self.afterimage_items),
                "averageStrength": average_strength,
                "items": [dict(item) for item in self.afterimage_items],
                "reversible": True,
                "rawTextStored": False,
                "rawTokenIdsStored": False,
            },
            "retentionDynamics": {
                "trackedAssemblies": len(retention_values),
                "averageScore": average_retention,
                "minimumScore": min(retention_values, default=0.0),
                "maximumScore": max(retention_values, default=0.0),
                "reinforcedSynapses": reinforced_synapses,
                "fixedStages": False,
                "reversible": True,
            },
            "automaticSettling": {
                "requiredManualAction": False,
                "cycles": self.cycle,
                "experienceCycles": self.settled_experiences,
                "restCycles": self.settled_rest_cycles,
                "expiredAfterimages": self.expired_afterimages,
                "reinforcementEvents": self.reinforcement_events,
                "scoresRecomputedEachCycle": True,
                "lastAt": self.last_settled_at or None,
                "lastSource": self.last_source or None,
                "visiblePonderForced": False,
            },
        }


__all__ = ["OrganicMemoryLifecycle"]
