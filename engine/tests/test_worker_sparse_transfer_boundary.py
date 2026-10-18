"""Host-boundary tests only: fake owners/helpers, no neural construction."""
import importlib.util
from contextlib import contextmanager
from pathlib import Path
from tempfile import TemporaryDirectory
from threading import RLock
from types import ModuleType, SimpleNamespace
import sys
import unittest
from unittest.mock import patch

ENGINE = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("owned_sparse_worker_fixture", ENGINE / "worker_sparse_transfer.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class Fault(Exception):
    def __init__(self, code, message, data=None):
        super().__init__(message)
        self.code, self.data = code, data


class Policy:
    disk_reserve_bytes = 20 * (1 << 30)
    def __init__(self):
        self.held, self.maximum, self.allocations = 0, 0, []
    @contextmanager
    def reserve_ram(self, count, _label):
        self.allocations.append(count)
        self.held += count
        self.maximum = max(self.maximum, self.held)
        try:
            yield
        finally:
            self.held -= count
    def require_disk(self, *_args):
        pass


class Host(module.WorkerSparseTransfer):
    running = True
    def __init__(self, root):
        self.root, self.policy = Path(root), Policy()
        self._steering_lock, self._inline_lock = RLock(), RLock()
        self._chat_steering, self._inline_generations = {}, {}
        self.owner = None
        self.get_calls = 0
    def _storage(self, _params, _id):
        return self.root
    def _brain_id(self, params):
        return params["brainId"]
    def _stdio_resource_policy(self):
        return self.policy
    def _request_cancel_check(self, _request):
        return lambda: False
    def _get(self, _params):
        self.get_calls += 1
        if self.owner is None:
            raise Fault(-32004, "Explicit Load is required")
        return self.owner
    def _execution_store(self, _params, brain=None):
        return SimpleNamespace(current=lambda: None)


class WorkerSparseTransferBoundaryTest(unittest.TestCase):
    def setUp(self):
        self.temp = TemporaryDirectory(prefix="omni-sparse-wrapper-")
        self.addCleanup(self.temp.cleanup)
        self.host = Host(self.temp.name)
        for kind in ("bundles", "plans"):
            (self.host.root / "substrate-transfers" / kind).mkdir(parents=True)

    def test_page_escrow_holds_largest_footprint_not_sum_of_rows(self):
        with self.host._sparse_scope("request") as (_cancelled, reserve):
            for count in (1024, 1024, 8192, 2048):
                reserve(count, "fixture")
            self.assertEqual(self.host.policy.held, 8192)
            self.assertEqual(self.host.policy.allocations, [1024, 7168])
        self.assertEqual(self.host.policy.held, 0)

    def test_committed_inspection_never_gets_or_initializes_brain(self):
        helper = ModuleType("omni_core.sparse_substrate_transfer")
        helper.CommittedSparseSnapshot = lambda *_args, **_kwargs: "committed"
        helper.inspect_sparse_snapshot = lambda snapshot, query, **_kwargs: {"snapshot": snapshot, "query": query}
        parent = ModuleType("omni_core"); parent.__path__ = []
        with patch.dict(sys.modules, {"omni_core": parent, "omni_core.sparse_substrate_transfer": helper}):
            result = self.host.sparse_substrate_inspect({"brainId": "fixture", "query": {"entity": "synapses"}})
        self.assertEqual(result, {"snapshot": "committed", "query": {"entity": "synapses"}})
        self.assertEqual(self.host.get_calls, 0)

    def test_artifact_paths_cannot_escape_to_source_or_another_owner(self):
        params = {"brainId": "fixture", "planPath": str(self.host.root / "engine" / "brain.json")}
        with self.assertRaisesRegex(ValueError, "exact owned artifact"):
            self.host._sparse_path(params, "planPath", "plans")
        path = self.host.root / "substrate-transfers" / "plans" / "reviewed.sqlite3"
        self.assertEqual(self.host._sparse_path({**params, "planPath": str(path)}, "planPath", "plans"), path)

    def test_apply_requires_existing_owner_and_preserves_current_reply(self):
        helper = ModuleType("omni_core.sparse_substrate_transfer")
        helper.apply_sparse_overlay = lambda *_args, **_kwargs: self.fail("Active reply was mutated")
        helper.SparseOverlayCommitError = type("SparseOverlayCommitError", (Exception,), {})
        helper.SparseOverlayStale = type("SparseOverlayStale", (Exception,), {})
        parent = ModuleType("omni_core"); parent.__path__ = []
        worker = ModuleType("worker"); worker.RpcFault = Fault
        with patch.dict(sys.modules, {"omni_core": parent, "omni_core.sparse_substrate_transfer": helper, "worker": worker}):
            with self.assertRaisesRegex(Fault, "Explicit Load"):
                self.host.sparse_substrate_apply({"brainId": "fixture"})
            self.host.owner = SimpleNamespace(brain_id="fixture")
            self.host._chat_steering[("fixture", "turn")] = object()
            with self.assertRaisesRegex(Fault, "current reply boundary"):
                self.host.sparse_substrate_apply({"brainId": "fixture"})


if __name__ == "__main__":
    unittest.main()
