"""Declared native graph recall source, not a prompt or a stored answer."""
from __future__ import annotations

import hashlib
import weakref


class ConnectedRecallBinding:
    def __init__(self, source, workspace_slots):
        if type(workspace_slots) is not int or workspace_slots < 1: raise ValueError("connected recall needs exact configured workspace slots")
        self._source, self.workspace_slots = weakref.ref(source), workspace_slots
    @property
    def source(self):
        value = self._source()
        if value is None: raise RuntimeError("connected recall source was released")
        return value
    def __call__(self, cue):
        from .training_randomness import training_wave_active, learning_transfer
        if training_wave_active():
            target = cue.device
            value = self.source.recall_vector(learning_transfer(cue, "cpu"),
                workspace_slots=self.workspace_slots, record_activity=False)[0]
            return learning_transfer(value, target)
        return self.source.recall_vector(cue, workspace_slots=self.workspace_slots, record_activity=False)[0]
    @property
    def authoritative_source(self): return self.source
    @property
    def authority_id(self): return str(id(self))
    def private(self, *, cancelled, transfer, reserve_metadata):
        # Reuse the existing private derived readout/frontier ownership shape,
        # but preserve the original complete source store authority.
        from .mcp_sampling import _memory_view
        names = ("attention_active_neuron_ids", "attention_recalled_assembly_ids", "attention_eligible_synapse_ids")
        values = [(name, getattr(self.source, name)) for name in names]
        if any(not isinstance(value, (set, frozenset)) for _name, value in values):
            raise ValueError("connected attention membership is not a declared native set")
        size = 4096 + sum(len(value) * 64 for _name, value in values)
        lease = reserve_metadata(size)
        try:
            view = _memory_view(self.source, cancelled)
            for name, value in values: setattr(view, name, frozenset(value))
            binding = PrivateConnectedRecallBinding(view, self, transfer, cancelled)
            lease.mark_allocated(size)
            return view, binding, lease
        except BaseException: lease.release(); raise


class PrivateConnectedRecallBinding(ConnectedRecallBinding):
    def __init__(self, view, original, transfer, cancelled):
        super().__init__(view, original.workspace_slots)
        self.original, self.transfer, self.cancelled = original, transfer, cancelled
    @property
    def authoritative_source(self): return self.original.authoritative_source
    @property
    def authority_id(self): return self.original.authority_id
    def __call__(self, cue):
        if self.cancelled(): raise InterruptedError("private connected recall stopped before its source read")
        target = cue.device
        try:
            value = self.source.recall_vector(self.transfer(cue, "cpu"), workspace_slots=self.workspace_slots,
                record_activity=False, cancelled=self.cancelled)[0]
        finally:
            state = getattr(self.source, "_last_recall_paged_state", None)
            if state is not None:
                self.source._last_recall_paged_state = None
                state.close()
        if self.cancelled(): raise InterruptedError("private connected recall stopped before home delivery")
        return self.transfer(value, target)


def recall_source_snapshot(binding, *, reserve, check_cancelled=lambda: None):
    """Exact supported live paged-store revisions, not a whole learned hash.

    Resident/opaque stores have no complete cross-owner mutation protocol here
    and refuse; that is not a claim that their neural recall is absent/broken.
    """
    from .paged_assembly_scoring import PagedAssemblyVectorProvider
    from .paged_assembly_view import PagedAssemblyView
    from .paged_neuron_metadata import PagedNeuronMetadata
    from .paged_packed_vectors import PagedPackedVectors
    from .vsa import NeuralSubstrate
    check_cancelled()
    if type(binding) not in {ConnectedRecallBinding, PrivateConnectedRecallBinding}: raise ValueError("opaque connected recall callback cannot enter Auto")
    memory = binding.authoritative_source
    if type(memory) is not NeuralSubstrate or any(name in vars(memory) for name in
            ("recall_vector", "_recall_vector_impl", "_recall_vector_paged", "release_paged_recall_scratch")):
        raise ValueError("connected recall requires its declared native source implementation")
    if not isinstance(memory.assemblies, PagedAssemblyView) or not isinstance(memory.neurons, PagedNeuronMetadata) or not isinstance(memory.neuron_vectors, PagedPackedVectors):
        raise ValueError("connected Auto requires complete supported paged graph owner revisions")
    synapses = memory.synapses
    if type(getattr(synapses, "graph_revision", None)) is not int or type(getattr(synapses, "persistence_revision", None)) is not int:
        raise ValueError("connected graph synapse revision is unavailable")
    sets = [(name, getattr(memory, name)) for name in ("attention_active_neuron_ids", "attention_recalled_assembly_ids", "attention_eligible_synapse_ids")]
    size = 4096 + sum(sum(64 + len(value) * 4 for value in values) for _name, values in sets)
    with reserve(size):
        attention = hashlib.sha256()
        for name, values in sets:
            check_cancelled()
            if not isinstance(values, (set, frozenset)) or any(not isinstance(value, str) for value in values):
                raise ValueError("connected attention membership is opaque")
            attention.update(name.encode())
            for value in sorted(values):
                check_cancelled()
                raw = value.encode(); attention.update(len(raw).to_bytes(8, "little")); attention.update(raw)
        check_cancelled()
        provider = PagedAssemblyVectorProvider(memory.assemblies.index, memory.neuron_vectors)
        generation = provider._read_generation()
        with memory.neurons._transaction() as connection:
            neuron = (memory.neurons._meta(connection, "store_id"), memory.neurons._revision(connection),
                      memory.neurons._graph_revision(connection), memory.neurons._decay_state(connection))
        return {"sourceObject": str(id(memory)), "workspaceSlots": binding.workspace_slots,
            "assembliesVectors": vars(generation), "neuronState": neuron,
            "synapses": [str(id(synapses)), synapses.graph_revision, synapses.persistence_revision],
            "sourceStateRevision": int(memory.state_revision), "attentionEpoch": int(memory.attention_epoch),
            "attentionLegacy": bool(memory.attention_legacy_raw_active), "attentionSetsSha256": attention.hexdigest()}
