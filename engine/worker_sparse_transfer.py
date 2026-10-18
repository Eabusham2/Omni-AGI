"""Committed sparse readers and one explicitly loaded, reviewed writer.

No operation creates a brain or translates foreign cortical weights. Native
methods receive host-owned custody paths, not paths supplied by neural operands.
"""
from contextlib import ExitStack, contextmanager
from pathlib import Path

SPARSE_READ_METHODS = frozenset({"sparse_substrate_inspect", "sparse_substrate_export",
    "sparse_substrate_validate_bundle", "sparse_substrate_prepare", "sparse_substrate_prepare_rollback", "sparse_substrate_head"})


class WorkerSparseTransfer:
    @contextmanager
    def _sparse_scope(self, request_id, brain=None, storage=None):
        policy = brain.resource_policy if brain is not None else self._stdio_resource_policy()
        check = self._request_cancel_check(request_id)
        cancelled = lambda: not self.running or check()
        maximum = 0
        with ExitStack() as leases:
            def reserve(size, label):
                nonlocal maximum
                size = int(size)
                if size > maximum:
                    leases.enter_context(policy.reserve_ram(size - maximum, label))
                    maximum = size
                # Retain the maximum page escrow, not every prior row's size.
                # Writes still check the actual free-disk floor as they grow.
                policy.require_disk(size * 2, label)
                if storage is not None:
                    import shutil
                    floor = max(20 * (1 << 30), int(policy.disk_reserve_bytes))
                    if shutil.disk_usage(storage).free - size * 2 < floor:
                        raise OSError("Sparse transfer waits for its actual destination drive reserve")
            yield cancelled, reserve

    def _sparse_path(self, params, field, kind):
        root = self._storage(params, self._brain_id(params))
        directory = root / "substrate-transfers" / kind
        path = Path(str(params.get(field, "")))
        if (not path.is_absolute() or path.parent != directory or path.suffix != ".sqlite3"
                or not path.stem or any(ch not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789._-" for ch in path.stem)):
            raise ValueError("Sparse custody path is not an exact owned artifact")
        for candidate in (root, directory.parent, directory):
            if candidate.is_symlink() or not candidate.is_dir():
                raise ValueError("Sparse custody directory is unsafe")
        if path.is_symlink():
            raise ValueError("Sparse custody artifact is a symlink")
        return path

    def _sparse_snapshot(self, params, cancelled, reserve):
        from omni_core.sparse_substrate_transfer import CommittedSparseSnapshot
        return CommittedSparseSnapshot(self._storage(params, self._brain_id(params)) / "engine",
            self._brain_id(params), cancelled=cancelled, reserve=reserve)

    def sparse_substrate_inspect(self, params, request_id=None):
        from omni_core.sparse_substrate_transfer import inspect_sparse_snapshot
        with self._sparse_scope(request_id, storage=self._storage(params, self._brain_id(params))) as (cancelled, reserve):
            return inspect_sparse_snapshot(self._sparse_snapshot(params, cancelled, reserve), params.get("query", {}), cancelled=cancelled, reserve=reserve)

    def sparse_substrate_export(self, params, request_id=None):
        from omni_core.sparse_substrate_transfer import export_sparse_bundle
        destination = self._sparse_path(params, "destination", "bundles")
        with self._sparse_scope(request_id, storage=destination.parent) as (cancelled, reserve):
            return export_sparse_bundle(self._sparse_snapshot(params, cancelled, reserve), params.get("selection", {}), destination, cancelled=cancelled, reserve=reserve)

    def sparse_substrate_validate_bundle(self, params, request_id=None):
        from omni_core.sparse_substrate_transfer import sparse_bundle_descriptor
        bundle = self._sparse_path(params, "bundlePath", "bundles")
        with self._sparse_scope(request_id, storage=bundle.parent) as (cancelled, reserve):
            return sparse_bundle_descriptor(bundle, params.get("bundleSha256"), cancelled=cancelled, reserve=reserve)

    def sparse_substrate_prepare(self, params, request_id=None):
        from omni_core.sparse_substrate_transfer import prepare_sparse_overlay
        bundle = self._sparse_path(params, "bundlePath", "bundles")
        destination = self._sparse_path(params, "destination", "plans")
        with self._sparse_scope(request_id, storage=destination.parent) as (cancelled, reserve):
            return prepare_sparse_overlay(self._sparse_snapshot(params, cancelled, reserve), bundle,
                params.get("bundleSha256"), destination, params.get("mode"), cancelled=cancelled, reserve=reserve)

    def sparse_substrate_head(self, params, request_id=None):
        from omni_core.execution_checkpoint import ExecutionCheckpointStore
        from omni_core.sparse_substrate_transfer import validate_sparse_transfer_head
        engine = self._storage(params, self._brain_id(params)) / "engine"
        with self._sparse_scope(request_id) as (cancelled, _reserve):
            if cancelled():
                raise InterruptedError("Sparse head inspection cancelled")
            metadata = ExecutionCheckpointStore(engine, self._brain_id(params), policy=self._stdio_resource_policy())._read(engine / "brain.json")
            if metadata.get("brain_id") != self._brain_id(params):
                raise ValueError("Sparse native head belongs to another identity")
            head = validate_sparse_transfer_head(metadata.get("sparse_transfer_head"))
            return None if head is None else {**head, "generationSha256": metadata["substrate"]["persistence"]["activeGeneration"]}

    def sparse_substrate_prepare_rollback(self, params, request_id=None):
        from omni_core.sparse_substrate_transfer import prepare_sparse_rollback
        original = self._sparse_path(params, "originalPlan", "plans")
        destination = self._sparse_path(params, "destination", "plans")
        with self._sparse_scope(request_id, storage=destination.parent) as (cancelled, reserve):
            return prepare_sparse_rollback(self._sparse_snapshot(params, cancelled, reserve), original,
                params.get("originalPlanSha256"), params.get("receiptJson"), params.get("receiptSha256"), destination,
                cancelled=cancelled, reserve=reserve)

    def sparse_substrate_apply(self, params, request_id=None):
        from omni_core.sparse_substrate_transfer import apply_sparse_overlay, SparseOverlayCommitError, SparseOverlayStale
        from worker import RpcFault
        # _get is load-only, never allow_load/create. Verify retained execution
        # and imagination are quiescent before any live row can be replaced.
        brain = self._get(params)
        with self._steering_lock:
            if any(owner == brain.brain_id for owner, _ in self._chat_steering):
                raise RpcFault(-32032, "Sparse import waits for the current reply boundary")
        current = self._execution_store(params, brain=brain).current()
        if current is not None and current["status"] == "suspended":
            raise RpcFault(-32032, "Sparse import waits for the suspended reply without changing its frontier")
        with self._inline_lock:
            if any(record.brain_id == brain.brain_id for record in self._inline_generations.values()):
                raise RpcFault(-32032, "Sparse import waits for the current imagination boundary")
        plan = self._sparse_path(params, "planPath", "plans")
        transaction = None
        @contextmanager
        def mutation_scope():
            nonlocal transaction
            with ExitStack() as scope:
                if getattr(brain, "_connected_regions", None) is not None:
                    from omni_core.connected_transaction import ConnectedExperienceTransaction
                    transaction = scope.enter_context(ConnectedExperienceTransaction(brain))
                yield
        def publication(head):
            brain.sparse_transfer_head = head
            if transaction is not None:
                transaction.prepare_publication()
            brain.save()
        try:
            with self._sparse_scope(request_id, brain, storage=brain.storage_path) as (cancelled, reserve):
                return apply_sparse_overlay(brain, plan, params, publication=publication, mutation_scope=mutation_scope, cancelled=cancelled, reserve=reserve)
        except SparseOverlayStale as error:
            raise RpcFault(-32033, str(error), {"brainId": brain.brain_id, "nativeSparseTransfer": True, "needsReview": True}) from error
        except SparseOverlayCommitError as error:
            release = self._release_failed_app_observation_owner(brain, committed=error.committed, reason="sparse-import-checkpoint-boundary")
            raise RpcFault(-32031, str(error), {**release, "nativeSparseTransfer": True, "reason": "sparse-import-checkpoint-boundary"}) from error
        except InterruptedError as error:
            raise RpcFault(-32800, str(error), {"brainId": brain.brain_id, "nativeSparseTransfer": True,
                "safeBoundary": True, "cancelled": True, "committed": False, "nativeOwnerReleased": False}) from error
