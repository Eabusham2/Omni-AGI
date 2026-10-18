"""Bounded regional before-images for one admitted connected experience.

SQLite stores share a real transaction for the duration of the admission;
their existing bounded write transactions become savepoints. Sparse-router
tile journals are retained until the enclosing admission succeeds. Learned
weights are journaled only on first mutation, never cloned into a float model.
"""
from __future__ import annotations

import copy
import json
import shutil
import sqlite3
import tempfile
import weakref
from contextvars import ContextVar
from pathlib import Path

import torch

_ACTIVE = ContextVar("connected_native_experience_transaction", default=None)


class _ConnectionLease:
    def __init__(self, participant):
        self._participant = participant
        self._savepoints = []
        self._closed = False
        participant.leases.add(self)

    @property
    def in_transaction(self):
        return self._participant.connection.in_transaction

    def _begin(self):
        parent = self._participant
        parent.serial += 1
        name = "connected_step_%d" % parent.serial
        parent.connection.execute("SAVEPOINT " + name)
        self._savepoints.append(name)
        return parent.connection.execute("SELECT 1")

    def execute(self, statement, parameters=()):
        normalized = statement.strip().rstrip(";").upper()
        if normalized.startswith("BEGIN"):
            return self._begin()
        if normalized in {"COMMIT", "END", "END TRANSACTION", "COMMIT TRANSACTION"}:
            self.commit()
            return self._participant.connection.execute("SELECT 1")
        if normalized in {"ROLLBACK", "ROLLBACK TRANSACTION"}:
            self.rollback()
            return self._participant.connection.execute("SELECT 1")
        if normalized.startswith("PRAGMA WAL_CHECKPOINT"):
            # The enclosing admission owns the commit; no WAL checkpoint can
            # occur while its transaction is still open.
            return self._participant.connection.execute("SELECT 0,0,0")
        return self._participant.connection.execute(statement, parameters)

    def executemany(self, statement, parameters):
        return self._participant.connection.executemany(statement, parameters)

    def executescript(self, script):
        # sqlite3.executescript implicitly COMMITs. Execute complete SQLite
        # statements individually so trigger bodies still parse correctly.
        pending = ""
        for character in script:
            pending += character
            if character == ";" and sqlite3.complete_statement(pending):
                self.execute(pending)
                pending = ""
        if pending.strip():
            self.execute(pending)
        return self

    def commit(self):
        if self._savepoints:
            self._participant.connection.execute("RELEASE SAVEPOINT " + self._savepoints.pop())

    def rollback(self):
        if self._savepoints:
            name = self._savepoints.pop()
            self._participant.connection.execute("ROLLBACK TO SAVEPOINT " + name)
            self._participant.connection.execute("RELEASE SAVEPOINT " + name)
        else:
            self._participant.abort_only = True

    def close(self):
        if self._closed:
            return
        while self._savepoints:
            self.rollback()
            self._participant.abort_only = True
        self._closed = True

    def __enter__(self):
        self._begin()
        return self

    def __exit__(self, kind, value, traceback):
        self.rollback() if kind else self.commit()
        return False

    def __getattr__(self, name):
        return getattr(self._participant.connection, name)


class _Participant:
    def __init__(self, connection):
        self.connection = connection
        self.serial = 0
        self.abort_only = False
        self.leases = weakref.WeakSet()
        connection.execute("BEGIN IMMEDIATE")


def borrow_paged_connection(path):
    from .training_cow import borrow_training_connection
    private = borrow_training_connection(path)
    if private is not None:
        return private
    scope = _ACTIVE.get()
    if scope is None:
        from .paged_read_session import borrow_read_connection
        return borrow_read_connection(path)
    if str(Path(path).resolve()) not in scope.allowed_databases:
        return None
    value = scope.databases.get(str(Path(path).resolve()))
    return _ConnectionLease(value) if value is not None else None


def enlist_paged_connection(path, connection):
    scope = _ACTIVE.get()
    key = str(Path(path).resolve())
    if scope is None:
        from .paged_read_session import enlist_read_connection
        return enlist_read_connection(path, connection)
    if key not in scope.allowed_databases:
        return connection
    if key in scope.databases:
        connection.close()
    else:
        scope.databases[key] = _Participant(connection)
    return _ConnectionLease(scope.databases[key])


def observe_synapse_access(owner, identifier):
    scope = _ACTIVE.get()
    if scope is not None and id(owner) in scope.synapse_owners:
        scope.capture_synapse(owner, str(identifier))


def retain_router_journal(journal, state, generation):
    scope = _ACTIVE.get()
    if scope is None or id(state.owner) not in scope.router_owners:
        return False
    scope.router_journals.append((journal, state, generation))
    return True


class ConnectedExperienceTransaction:
    """Rollback only this uncommitted admission, never a previous accepted turn."""
    _MEMORY_FIELDS = ("growth_events", "growth_pauses", "state_revision", "attention_epoch",
        "attention_legacy_raw_active", "attention_active_neuron_ids", "attention_recalled_assembly_ids",
        "attention_eligible_synapse_ids", "_last_recall_audit")

    def __init__(self, brain):
        self.brain = brain
        self.connected = brain._connected_regions
        self.memories = dict(self.connected.iter_memories())
        self.allowed_databases = set()
        self.databases = {}
        self.synapse_owners = {}
        self.router_owners = set()
        self.router_journals = []
        self._token = None
        self.directory = None
        self.journal = None
        self.controls = {}
        self.control_reader = None
        self.overlay = None
        self.memory_controls = {}
        self.synapse_controls = {}
        self.state_controls = {}
        self.layout = None
        self._publication_boundary = False

    def _reserve_memory(self, count, label):
        from .offload import NeuralStateResourcePause
        status = self.brain.resource_policy.status(estimated_ram_bytes=count)
        if status.get("memoryPressure"):
            raise NeuralStateResourcePause(label + " waits for its bounded RAM reservation", status)

    def __enter__(self):
        if _ACTIVE.get() is not None:
            raise RuntimeError("connected experience transactions must not overlap")
        from .vsa import LazyPersistedSynapses
        from .paged_packed_vectors import PagedPackedVectors
        from .paged_neuron_metadata import PagedNeuronMetadata
        from .paged_assembly_view import PagedAssemblyView
        from .slow_state_snapshot import admit_snapshot_metadata
        from .bounded_tensor_io import atomic_save_tensors_bounded, BoundedTensorFile
        from .parameter_diagnostics import ParameterDeltaJournal
        try:
            for name, memory in self.memories.items():
                if not (isinstance(memory.neuron_vectors, PagedPackedVectors)
                        and isinstance(memory.neurons, PagedNeuronMetadata)
                        and isinstance(memory.assemblies, PagedAssemblyView)
                        and isinstance(memory.synapses, LazyPersistedSynapses)):
                    raise RuntimeError("connected admission requires its complete paged regional stores")
                self.allowed_databases.update(str(owner.path.resolve()) for owner in
                    (memory.neuron_vectors, memory.neurons, memory.assemblies.index))
                state = {key: getattr(memory, key) for key in self._MEMORY_FIELDS}
                admit_snapshot_metadata(self.brain.resource_policy, state, "regional activity before-image")
                self.memory_controls[name] = copy.deepcopy(state)
                self.synapse_owners[id(memory.synapses)] = (name, memory.synapses)
                self.synapse_controls[name] = (memory.synapses.graph_revision, memory.synapses.persistence_revision)
            for name, region in self.brain.decoder.named_modules():
                from .connected_regions import NativeRegion
                if not isinstance(region, NativeRegion):
                    continue
                self.router_owners.add(id(region.router.synapses))
                sparse = region.router.synapses._sparse_state
                self.state_controls[name] = (sparse, sparse._generation)
                self.controls[name + ".liquid_state"] = region.liquid_state
                for field in ("membrane", "spike_count"):
                    self.controls[name + ".population." + field] = getattr(region.router.population, field)
            amount = sum(value.numel() * value.element_size() for value in self.controls.values())
            self.brain.resource_policy.require_disk(amount + 131072, "regional control before-images")
            directory = self.brain._live_paging_cache_directory / "connected-admission"
            directory.mkdir(parents=True, exist_ok=True)
            self.directory = Path(tempfile.mkdtemp(prefix="admission-", dir=directory))
            self.overlay = sqlite3.connect(self.directory / "overlays.sqlite3")
            self.overlay.execute("PRAGMA journal_mode=OFF")
            self.overlay.execute("PRAGMA synchronous=OFF")
            self.overlay.execute("PRAGMA cache_size=-128")
            self.overlay.execute("CREATE TABLE originals(owner TEXT, identifier TEXT, payload TEXT, PRIMARY KEY(owner,identifier))")
            path = self.directory / "controls.safetensors"
            atomic_save_tensors_bounded(path, self.controls)
            self.control_reader = BoundedTensorFile(path)
            self.layout = self.brain.decoder.layout()
            self.journal = ParameterDeltaJournal((("decoder", self.brain.decoder),),
                directory=self.directory / "changed-native-bytes", reserve_ram=lambda size: self._reserve_memory(size, "regional changed-byte journal"),
                reserve_disk=self.brain.resource_policy.require_disk, include_state=True,
                resource_policy=self.brain.resource_policy)
            self._token = _ACTIVE.set(self)
            return self
        except BaseException:
            self.close()
            raise

    def capture_synapse(self, owner, identifier):
        name = self.synapse_owners[id(owner)][0]
        if self.overlay.execute("SELECT 1 FROM originals WHERE owner=? AND identifier=?", (name, identifier)).fetchone():
            return
        raw = {"dirty": dict(owner._dirty[identifier]) if identifier in owner._dirty else None,
               "dirtyLocation": owner._dirty_locations.get(identifier),
               "deletedLocation": owner._deleted_locations.get(identifier),
               "originalUses": owner._original_uses.get(identifier),
               "new": identifier in owner._new_ids}
        payload = json.dumps(raw, sort_keys=True, separators=(",", ":"), allow_nan=False)
        self.brain.resource_policy.require_disk(len(payload.encode("utf-8")) + 4096, "regional synapse before-image")
        self.overlay.execute("INSERT INTO originals VALUES(?,?,?)", (name, identifier, payload))
        self.overlay.commit()

    def _restore_overlays(self):
        for name, identifier, payload in self.overlay.execute("SELECT owner,identifier,payload FROM originals"):
            owner = self.memories[name].synapses
            old = json.loads(payload)
            if old["dirty"] is None:
                owner._dirty.pop(identifier, None)
            else:
                owner[identifier] = old["dirty"]
            for field, source in (("_dirty_locations", "dirtyLocation"), ("_deleted_locations", "deletedLocation")):
                store = getattr(owner, field)
                if old[source] is None:
                    store.pop(identifier, None)
                else:
                    store[identifier] = (tuple(old[source][0]), old[source][1])
            if old["originalUses"] is None:
                owner._original_uses.pop(identifier, None)
            else:
                owner._original_uses[identifier] = old["originalUses"]
            owner._new_ids.add(identifier) if old["new"] else owner._new_ids.discard(identifier)
        for name, memory in self.memories.items():
            for field, value in self.memory_controls[name].items():
                setattr(memory, field, copy.deepcopy(value))
            memory.synapses.graph_revision, memory.synapses.persistence_revision = self.synapse_controls[name]
            memory.neuron_vectors.invalidate_all()
            memory.invalidate_assembly_index()
            memory.release_paged_recall_scratch()
            memory._recall_graph_cache = None
            memory._weighted_representation_cache = None

    def _rollback(self):
        for participant in self.databases.values():
            participant.connection.rollback()
        for journal, state, generation in reversed(self.router_journals):
            journal.rollback()
            for block in state.owner.blocks.values():
                if block._router_birth == generation:
                    block._router_unpublished = True
            state._discard_generation(generation)
        self.brain.decoder.restore_layout(self.layout)
        self.journal.restore()
        for name, tensor in self.controls.items():
            self.control_reader.copy_into(name, tensor)
        for state, generation in self.state_controls.values():
            state._generation = generation
        self._restore_overlays()

    def prepare_publication(self):
        """Release derived-cache writer locks before immutable root staging.

        The root brain.json remains the sole durable authority. Journals and
        control before-images stay alive until its save succeeds. If staging
        then fails, learned bytes roll back and a cold load rebuilds the already
        committed derived caches from the previous root's immutable graph.
        Keeping BEGIN IMMEDIATE open through DirtyShardPlan is a self-deadlock.
        """
        if self._publication_boundary:
            return
        if any(owner.abort_only for owner in self.databases.values()):
            raise RuntimeError("cannot publish an aborted regional admission")
        for participant in self.databases.values():
            if any(lease._savepoints for lease in participant.leases):
                raise RuntimeError("regional publication left an open store write")
        if self._token is not None:
            _ACTIVE.reset(self._token)
            self._token = None
        committed = 0
        try:
            for participant in self.databases.values():
                participant.connection.commit()
                committed += 1
            self._publication_boundary = True
        except BaseException:
            if committed:
                self.brain._connected_recovery_required = True
            raise

    def __exit__(self, kind, value, traceback):
        if self._token is not None:
            _ACTIVE.reset(self._token)
            self._token = None
        try:
            if kind is not None or any(owner.abort_only for owner in self.databases.values()):
                self._rollback()
                if kind is not None:
                    # The surrounding root experience may have changed its
                    # own state before regional admission began. Its existing
                    # worker recovery must reload the last atomic root; never
                    # continue warm from a half-failed enclosing operation.
                    self.brain._connected_recovery_required = True
                if kind is None:
                    raise RuntimeError("regional admission contained an aborted graph transaction")
            elif not self._publication_boundary:
                committed = 0
                try:
                    for participant in self.databases.values():
                        if any(lease._savepoints for lease in participant.leases):
                            raise RuntimeError("regional admission left an open store write")
                        participant.connection.commit()
                        committed += 1
                except BaseException:
                    if committed:
                        # A cross-file commit interruption needs cold replay
                        # from the last atomic brain checkpoint, never a
                        # partially admitted warm graph.
                        self.brain._connected_recovery_required = True
                    else:
                        self._rollback()
                    raise
        except BaseException:
            self.brain._connected_recovery_required = True
            raise
        finally:
            self.close()
        return False

    def close(self):
        if self._token is not None:
            _ACTIVE.reset(self._token)
            self._token = None
        for participant in self.databases.values():
            participant.connection.close()
        self.databases.clear()
        for journal, _state, _generation in self.router_journals:
            journal.close()
        self.router_journals.clear()
        if self.journal is not None:
            self.journal.close()
            self.journal = None
        if self.overlay is not None:
            self.overlay.close()
            self.overlay = None
        if self.directory is not None and self.directory.is_dir():
            shutil.rmtree(self.directory)
        self.directory = None
