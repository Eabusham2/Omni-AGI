"""NEW constructor-free scoped epoch checks: no Torch/model/device import."""
import ast
import copy
import importlib.util
import sys
import threading
import types
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("owned_scoped_owner_fixture", ROOT / "omni_core/scoped_owner_observation.py")
SCOPED = importlib.util.module_from_spec(spec); spec.loader.exec_module(SCOPED)


class Tensor:
    def __init__(self, version=0): self._version = version


def methods():
    tree = ast.parse((ROOT / "omni_core/native_core_paging.py").read_text())
    cls = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "NativeCorePager")
    selected = []
    for node in cls.body:
        if isinstance(node, ast.FunctionDef) and node.name in {"_observe_mutations", "_rebind_learned_tokens", "retire_learning_observation", "learning_revision"}:
            node = copy.deepcopy(node); node.decorator_list = []; selected.append(node)
    future = ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0)
    namespace = {"torch": types.SimpleNamespace(Tensor=Tensor)}
    exec(compile(ast.fix_missing_locations(ast.Module(body=[future, *selected], type_ignores=[])), "actual-scoped-pager-source", "exec"), namespace)
    return namespace


class Lease:
    def __init__(self): self.closed = False
    def mark_allocated(self, _size): pass
    def release(self): self.closed = True


class Policy:
    def __init__(self): self.leases, self.fail = [], False
    def reserve_ram(self, size, _label):
        if self.fail: raise RuntimeError("owned metadata pressure")
        assert size < 32 * 1024
        result = Lease(); self.leases.append(result); return result


def fixture():
    namespace, policy = methods(), Policy()
    pager = types.SimpleNamespace(_lock=threading.RLock(), _closed=False, shared_resource_policy=policy,
        _learning_revision_active=False, _learning_revision=0, _learning_tracking_lease=None, _revision=0, _owners={})
    for name in ("_observe_mutations", "retire_learning_observation"):
        setattr(pager, name, types.MethodType(namespace[name], pager))
    pager._rebind_learned_tokens = namespace["_rebind_learned_tokens"]
    modules = []
    for _index in range(2):
        module = type("Module", (), {})()
        module._buffers = {key: Tensor() for key in ("sign", "strength", "repeat", "sort")}
        module._non_persistent_buffers_set = {"sort"}
        owner = types.SimpleNamespace(cold=dict(module._buffers), versions={key: 0 for key in module._buffers},
            loaded=True, learned_tokens={}, scoped_learning_observers=0, scoped_learning_revision=0,
            maps={}, device=None, dirty_names=set())
        pager._owners[module] = owner; modules.append(module)
    return pager, policy, modules, namespace


class ScopedOwnerCases(unittest.TestCase):
    def test_exact_persistent_mutations_only_and_known_moves_do_not_retire_other_scope(self):
        pager, policy, (a, b), _namespace = fixture()
        with SCOPED.ScopedLearningOwners(pager, [a]) as scope:
            baseline = scope.snapshot()
            b._buffers["strength"]._version += 1; pager._observe_mutations(b, pager._owners[b])
            a._buffers["sort"]._version += 1
            self.assertEqual(scope.snapshot(), baseline)
            a._buffers["strength"]._version += 1
            self.assertEqual(scope.snapshot()[str(id(a))], 1)
            a._buffers["sign"] = Tensor(7)
            pager._rebind_learned_tokens(a, pager._owners[a])  # Known exact representation move.
            self.assertEqual(scope.snapshot()[str(id(a))], 1)
            a._buffers["repeat"]._version += 1
            self.assertEqual(scope.snapshot()[str(id(a))], 2)
            a._buffers["sign"] = Tensor(7)  # Unknown replacement despite the same scalar version.
            self.assertEqual(scope.snapshot()[str(id(a))], 3)
            self.assertEqual(pager._learning_revision, 0)
        self.assertTrue(all(lease.closed for lease in policy.leases))

    def test_global_observer_start_stop_does_not_mask_or_retire_independent_scoped_owner(self):
        pager, policy, (a, _b), namespace = fixture()
        first = SCOPED.ScopedLearningOwners(pager, [a]); second = SCOPED.ScopedLearningOwners(pager, [a])
        a._buffers["strength"]._version += 1
        namespace["learning_revision"](pager)
        self.assertEqual(first.snapshot()[str(id(a))], 1)
        pager.retire_learning_observation()
        a._buffers["repeat"]._version += 1
        self.assertEqual(first.snapshot()[str(id(a))], 2)
        first.close(); self.assertEqual(pager._owners[a].scoped_learning_observers, 1)
        self.assertEqual(second.snapshot()[str(id(a))], 2)
        second.close(); self.assertEqual(pager._owners[a].learned_tokens, {})
        self.assertTrue(all(lease.closed for lease in policy.leases))

    def test_pressure_partial_unloaded_retirement_and_replacement_close_all_metadata(self):
        pager, policy, (a, b), _namespace = fixture()
        policy.fail = True
        with self.assertRaisesRegex(RuntimeError, "metadata pressure"): SCOPED.ScopedLearningOwners(pager, [a])
        self.assertEqual(pager._owners[a].scoped_learning_observers, 0)
        policy.fail = False; pager._owners[b].loaded = False
        with self.assertRaisesRegex(RuntimeError, "completely loaded"): SCOPED.ScopedLearningOwners(pager, [a, b])
        self.assertEqual(pager._owners[a].scoped_learning_observers, 0)
        with SCOPED.ScopedLearningOwners(pager, [a]) as scope:
            old = pager._owners.pop(a)
            with self.assertRaisesRegex(RuntimeError, "retired or replaced"): scope.snapshot()
        self.assertEqual(old.scoped_learning_observers, 0)
        with self.assertRaisesRegex(RuntimeError, "retired"): scope.snapshot()
        self.assertTrue(all(lease.closed for lease in policy.leases))


if __name__ == "__main__": unittest.main()
