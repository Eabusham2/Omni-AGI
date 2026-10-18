"""Borrow an existing registered native region; never construct another brain.

Only temporary owner metadata and private writable SQLite activity are copied.
Packed cortical/router owners remain the identical objects registered inside
the connected decoder. The outer queue owns persistence and action identity.
"""
from __future__ import annotations

import copy
import os
from pathlib import Path
import shutil
import sqlite3
from contextlib import closing
from types import MethodType

import torch

from .bounded_tensor_io import BoundedTensorFile
from .config import OmniConfig
from .conversation_ledger import NeuralConversationLedger
from .memory_lifecycle import OrganicMemoryLifecycle
from .offload import DurableReplayBuffer, PagedWorkingMemory
from .persistence import atomic_write_json, read_json
from .recent_token_activity import RecentTokenActivity


class StagedEvents:
    def __init__(self):
        self.rows = []

    def append(self, kind, payload, job_id=None):
        self.rows.append((kind, copy.deepcopy(payload), job_id))


def registered_owner(root, path):
    connected = root._connected_regions
    for position, identifier in enumerate(path):
        matches = [region for region in connected.decoder.regions if region.identity.region_id == identifier]
        if len(matches) != 1:
            raise ValueError("inherited execution lost its exact registered native owner")
        region = matches[0]
        if position == len(path) - 1:
            return region, connected.memories[identifier], (connected.nested or {}).get(identifier)
        connected = (connected.nested or {}).get(identifier)
        if connected is None:
            raise ValueError("inherited execution addresses a missing nested native owner")
    raise ValueError("inherited execution has no native owner")


def private_runtime_engine(source, destination, policy):
    """Private writable pointers/databases; immutable tensors may share inodes."""
    destination = Path(destination)
    if destination.exists():
        if destination.is_symlink() or not destination.is_dir():
            raise ValueError("inherited runtime engine is not an owned directory")
        return destination
    destination.mkdir(parents=True, mode=0o700)
    try:
        for name in ("core.safetensors", "plasticity.safetensors"):
            origin = source / name
            if origin.is_symlink() or not origin.is_file():
                raise ValueError("inherited runtime is missing its exact native tensor seal")
            target = destination / name
            try:
                os.link(origin, target)
            except OSError:
                policy.require_disk(origin.stat().st_size, "inherited immutable native tensor closure")
                shutil.copyfile(origin, target)
                with target.open('rb') as synced:
                    os.fsync(synced.fileno())
        atomic_write_json(destination / "brain.json", read_json(source / "brain.json"))
        from .concept_id_views import IdView
        import json
        views = source / "state/concept-id-views"
        if views.exists():
            if views.is_symlink() or not views.is_dir():
                raise ValueError("inherited structural arguments have an unsafe view directory")
            for path in views.iterdir():
                if path.name.startswith('.'):
                    continue
                if path.is_symlink() or not path.is_file():
                    raise ValueError("inherited structural argument view is not a regular file")
                with path.open('rb') as incoming:
                    header = json.loads(incoming.readline(4096))
                descriptor = {**header, 'path': 'state/concept-id-views/' + path.name,
                    'sha256': path.stem, 'bytes': path.stat().st_size}
                IdView(source, descriptor, brain_id=header['brainId'], turn_id=header['turnId'])
                policy.require_disk(path.stat().st_size + 65536, "inherited exact structural arguments")
                target = destination / descriptor['path']; target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(path, target)
                with target.open('rb') as synced:
                    os.fsync(synced.fileno())
        for name in ("conversation.sqlite3", "state/replay.sqlite3", "state/working-memory.sqlite3"):
            origin, target = source / name, destination / name
            if origin.is_symlink() or not origin.is_file():
                raise ValueError("inherited runtime activity database is missing")
            policy.require_disk(origin.stat().st_size * 2 + 65536, "inherited private runtime activity")
            for suffix in ('-wal', '-journal'):
                sidecar = Path(str(origin) + suffix)
                if sidecar.exists() and sidecar.stat().st_size:
                    raise ValueError("freeze/checkpoint inherited runtime SQLite before borrowing its immutable state")
            target.parent.mkdir(parents=True, exist_ok=True)
            with closing(sqlite3.connect(origin.resolve().as_uri() + "?mode=ro&immutable=1", uri=True)) as incoming, closing(sqlite3.connect(target)) as outgoing:
                incoming.backup(outgoing, pages=256)
                if outgoing.execute("PRAGMA quick_check").fetchone() != ("ok",):
                    raise ValueError("inherited private activity failed integrity verification")
                outgoing.execute("PRAGMA journal_mode=DELETE")
        return destination
    except BaseException:
        # This exact destination was created by this call and is not a parent
        # snapshot. Removing an incomplete private activity copy is recoverable.
        shutil.rmtree(destination)
        raise


def discard_new_private_runtime(root, entry_id, runtime):
    """Only a generated, unreferenced failed-prepare UUID, never a source."""
    runtime = Path(runtime).resolve()
    expected = (root.engine_path / 'connected-executions' / entry_id / 'runtime').resolve()
    import re
    if (runtime.name != 'engine' or runtime.parent.parent != expected
            or not re.fullmatch('[a-f0-9]{32}', runtime.parent.name)):
        raise ValueError('failed native runtime cleanup target is not its generated UUID')
    if runtime.parent.is_symlink():
        raise ValueError('failed native runtime cleanup cannot follow links')
    if runtime.parent.is_dir():
        shutil.rmtree(runtime.parent)


class BorrowedRegionalOwner:
    def __init__(self, root, entry, source_engine, runtime_engine):
        self.root, self.leases, self.configs, self.flags, self.closed = root, [], [], [], False
        try:
            self._initialize(root, entry, source_engine, runtime_engine)
        except BaseException:
            self.close()
            raise

    def _initialize(self, root, entry, source_engine, runtime_engine):
        from .brain import AdaptiveBrain
        self.root, self.entry = root, entry
        self.region, self.memory, self.nested = registered_owner(root, entry["regionPath"])
        self.source = Path(source_engine)
        self.runtime = Path(runtime_engine)
        self.saved = read_json(self.source / "brain.json")
        if self.saved.get("brain_id") != entry["parentId"]:
            raise ValueError("inherited native owner metadata changed identity")
        # A Python owner view is not a second model: no constructor, random
        # initialization, module duplication, dtype conversion or optimizer.
        view = object.__new__(AdaptiveBrain)
        view.__dict__.update(root.__dict__)
        self.brain = view
        view.brain_id = entry["parentId"]
        view.storage_path, view.engine_path = self.runtime.parent, self.runtime
        view.config = OmniConfig.from_dict(copy.deepcopy(self.saved["config"]))
        if str(root.device) != entry["backend"]:
            raise ValueError("exact inherited continuation requires its recorded compute backend")
        view._connected_regions = self.nested
        view._connected_graph_states = copy.deepcopy(self.saved.get("connected_graph_states"))
        view.connected_executions = None
        # Host/root app observations have a different FIFO/evidence owner.
        # A shallow root head must never masquerade as a parent receipt chain.
        view.app_observation_frontier = copy.deepcopy(self.saved.get('app_observation_frontier'))
        view.connected_pending_ingestions = copy.deepcopy(self.saved.get("connected_pending_ingestions", []))
        view.connected_runtime_adoption = copy.deepcopy(self.saved.get("connected_runtime_adoption"))
        for name in ("decoder", "memory_bridge", "idea_adapter", "router", "liquid", "modalities"):
            setattr(view, name, getattr(self.region, name))
        view.memory, view.liquid_state = self.memory, self.region.liquid_state
        view.mutable_state_manifest = copy.deepcopy(self.saved["mutable_state"])
        from .activity_clock import ActivityClock
        if 'activity_clock' not in self.saved:
            raise ValueError("unsupported inherited snapshot: no persisted source activity clock")
        view.activity_clock = ActivityClock.from_state(self.saved['activity_clock'],
            legacy_markers=tuple(self.saved.get(name, 0.) for name in
                ('last_activity_decay', 'last_idle_cycle_at', 'last_idle_visible_action_at')))
        for name in ("counters", "training_sources", "completed_ingestions", "completed_chat_turns", "accepted_chat_inputs",
                     "pending_chat_slow_learning", "completed_chat_slow_learning", "ingestion_checkpoints", "ingestion_joint_generation",
                     "workspace_items", "current_context", "fresh_attention_boundary", "modality_training", "installed_modality_packs",
                     "supplementary_capability_receipts", "ground_up_training_manifest", "geometry_holdout_registration", "packed_ternary_manifest",
                     "created_at", "updated_at", "novelty_streak", "growth_pause", "resource_pause", "last_activity_decay", "last_idle_cycle_at", "last_idle_visible_action_at"):
            setattr(view, name, copy.deepcopy(self.saved.get(name, getattr(root, name, None))))
        view.recent_token_context = list(self.saved.get("recent_token_context", []))
        view.recent_token_activity = RecentTokenActivity.from_state(self.saved.get("recent_token_activity"),
            view.recent_token_context, human_id=view.tokenizer.human_id)
        view._fresh_attention_paged_clear_pending = False
        view._connected_recovery_required = False
        view._live_paging_cache_directory = root._live_paging_cache_directory / "ancestral" / entry["id"]
        view._live_paging_cache_directory.mkdir(parents=True, exist_ok=True)
        plastic = BoundedTensorFile(self.source / "plasticity.safetensors")

        def tensor(name):
            if name not in plastic.specs:
                return None
            size = plastic.specs[name].byte_count
            self.leases.append(root.resource_policy.reserve_ram(size + 4096, "inherited temporary native activity"))
            return plastic.tensor(name)

        working = tensor("state.working_memory")
        view.working_memory = [] if working is None else [row for row in working]
        if len(view.workspace_items) != len(view.working_memory):
            raise ValueError("inherited working activity differs from its source owner")
        afterimages = tensor("memory_lifecycle.afterimage_vectors")
        if afterimages is None:
            afterimages = tensor("memory_lifecycle.scratch_vectors")
        view.memory_lifecycle = OrganicMemoryLifecycle.from_state(self.saved.get("memory_lifecycle"),
            {} if afterimages is None else {"memory_lifecycle.afterimage_vectors": afterimages})
        for field, name in (("language", "state.starter_action_language_features"),
                            ("internal", "state.starter_action_internal_features"), ("target", "state.starter_action_targets")):
            setattr(view, "_starter_action_" + field + "_cache", tensor(name))
        view.slow_anchors, view.slow_importance = {}, {}
        view.replay = DurableReplayBuffer(self.runtime / "state/replay.sqlite3", root.resource_policy)
        view.paged_working_memory = PagedWorkingMemory(self.runtime / "state/working-memory.sqlite3", root.resource_policy)
        from .working_activity import checkpoint_wall_time
        view.paged_working_memory.bind_activity_clock(lambda: view._owned_activity_clock().now(),
            checkpoint_wall_at=checkpoint_wall_time(self.saved.get("updated_at")))
        view.conversation = NeuralConversationLedger(self.runtime / "conversation.sqlite3", view.brain_id)
        view.messages, view.traces = [], []
        view.events = StagedEvents()
        self.before_counters = copy.deepcopy(view.counters)
        self.before_lifecycle = view.memory_lifecycle.metadata()
        self.before_pending = {row["jobId"] for row in view.pending_chat_slow_learning}
        with view.replay._connect() as database:
            self.before_replay_sequence = database.execute("SELECT COALESCE(MAX(sequence),0) FROM replay").fetchone()[0]
        # Resources remain one root lease. A regional fast observation cannot
        # reconfigure the shared KV pager or serialize the root optimizer.
        view._configure_working_attention_resources = lambda: None
        view._maintain_neural_state_resources = lambda: view._state_offload_status(root.resource_policy.status())
        view.runtime_card = lambda: {"inheritedNativeOwner": entry["parentId"], "hiddenBehavioralPromptTokens": 0}
        view.metrics = lambda: {"inheritedNativeOwner": entry["parentId"]}
        # The enclosing root first-write transaction already journals these
        # identical modules/stores. Do not open a second nested transaction.
        def learn(self, text, kind="knowledge", source="conversation", source_label="", steps=None,
                  importance=.5, structural_detail=True):
            if steps not in (None, 0) or steps is None and self.config.online_steps:
                raise ValueError("inherited completion queues slow learning rather than running another training job")
            return self._learn_experience_impl(text, kind=kind, source=source, source_label=source_label,
                steps=0, importance=importance, structural_detail=structural_detail)
        view.learn_experience = MethodType(learn, view)

    def enter(self):
        from .execution_rng import capture_execution_rng
        self.rng = capture_execution_rng(self.root.device)
        seen = set()
        for owner in self.brain._trainable_modules():
            for module in owner.modules():
                if id(module) in seen:
                    continue
                seen.add(id(module)); self.flags.append((module, module.training))
                config = getattr(module, "config", None)
                if config is not None and hasattr(config, "max_seq_len"):
                    self.configs.append((config, config.max_seq_len))
                    config.max_seq_len = self.brain.config.max_seq_len
        return self

    def close(self):
        if self.closed:
            return
        self.closed = True
        for config, value in reversed(self.configs):
            config.max_seq_len = value
        # Assign individual flags: .train() recursively would overwrite a
        # deliberately mixed child mode belonging to the connected owner.
        for module, value in self.flags:
            module.training = value
        if hasattr(self, "rng"):
            from .execution_rng import restore_execution_rng
            restore_execution_rng(self.rng, self.root.device)
        if hasattr(self, "brain") and hasattr(self.brain, "conversation") and self.brain.conversation is not self.root.conversation:
            self.brain.conversation.close()
        for lease in self.leases:
            lease.release()
        self.leases.clear()
        # Deliberately never call AdaptiveBrain.close: modules, pagers and
        # graph indices are registered root owners, not borrowed-view owners.
