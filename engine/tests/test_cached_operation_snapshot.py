"""NEW pure recursive activity ownership cases; no neural/model work."""
import importlib
import sys
import types
import unittest
from pathlib import Path

PACKAGE = "owned_complete_cache_fork"
package = types.ModuleType(PACKAGE); package.__path__ = [str(Path(__file__).resolve().parents[1] / "omni_core")]
sys.modules[PACKAGE] = package
SNAPSHOT = importlib.import_module(PACKAGE + ".cached_operation_snapshot")


class Page:
    def __init__(self, data, references=None): self.data, self.references, self.closed = tuple(data), references or [1], False
    def fork(self): self.references[0] += 1; return Page(self.data, self.references)
    def close(self):
        if not self.closed: self.references[0] -= 1; self.closed = True


class Chunk:
    def __init__(self, queries, mass, weighted_values): self.queries, self.mass, self.weighted_values = queries, mass, weighted_values
    def close(self):
        for value in (self.queries, self.mass, self.weighted_values):
            if isinstance(value, Page): value.close()


class Attention:
    def __init__(self, key, value): self.key, self.value = key, value
    def close(self): self.key.close(); self.value.close()


class Ordinary:
    def __init__(self, length, signature, workspace, attention, hidden_sum, memory_hidden, neural_state):
        self.length, self.signature, self.workspace, self.attention = length, signature, workspace, attention
        self.hidden_sum, self.memory_hidden, self.neural_state = hidden_sum, memory_hidden, neural_state
    def close(self):
        for owner in (*self.workspace, *self.attention):
            if owner is not None: owner.close()


class Connected:
    def __init__(self, length, signature, workspace, attention, temporal, hidden_sum, nested):
        self.length, self.signature, self.workspace, self.attention = length, signature, workspace, attention
        self.temporal, self.hidden_sum, self.nested = temporal, hidden_sum, nested
    def close(self):
        for group in (*self.workspace, *self.attention):
            for owner in group:
                if owner is not None: owner.close()
        for child in self.nested:
            if child is not None: child.close()


class Lease:
    def __init__(self): self.closed = False
    def release(self): self.closed = True
    def mark_allocated(self, size): self.size = size


def ordinary():
    return Ordinary(3, "original", [Chunk(Page([1]), [2], [3])], [Attention(Page([4, 5]), Page([6, 7]))], [8], [9], [10])


class RecursiveCacheOwnershipCases(unittest.TestCase):
    def fork(self, source, *, fail=None):
        leases = []
        def reserve(size): lease = Lease(); leases.append(lease); return lease
        def tensor(value):
            if fail is not None and value == fail: raise RuntimeError("fake raw activity admission refused")
            return list(value), reserve(len(value) * 4)
        return SNAPSHOT.fork_operation_cache(source, kind=lambda node: "ordinary" if type(node) is Ordinary else "connected" if type(node) is Connected else None,
            activity=lambda value: value.fork() if isinstance(value, Page) else None, tensor=tensor,
            workspace=Chunk, attention=Attention, ordinary=Ordinary, connected=Connected, reserve_metadata=reserve), leases

    def test_nested_temporal_and_running_state_are_independent_while_immutable_pages_are_shared(self):
        child = ordinary()
        root = Connected(3, "root", [[]], [[]], [([11], [12], [13])], [[14]], [child])
        fork, leases = self.fork(root)
        copy = fork.cache
        copy.temporal[0][0].append(99); copy.hidden_sum[0].append(88); copy.nested[0].hidden_sum.append(77)
        self.assertEqual(root.temporal, [([11], [12], [13])]); self.assertEqual(root.hidden_sum, [[14]])
        self.assertEqual(child.hidden_sum, [8]); self.assertEqual(child.attention[0].key.references[0], 2)
        fork.close(); self.assertEqual(child.attention[0].key.references[0], 1)
        self.assertTrue(all(lease.closed for lease in leases)); self.assertFalse(child.attention[0].key.closed)

    def test_partial_failure_closes_only_private_page_references_and_reservations(self):
        source = ordinary()
        before = source.attention[0].key.references[0]
        with self.assertRaisesRegex(RuntimeError, "admission refused"): self.fork(source, fail=[8])
        self.assertEqual(source.attention[0].key.references[0], before)
        self.assertFalse(source.workspace[0].queries.closed)

    def test_adoption_keeps_original_owner_alive_and_recursive_page_ownership_until_close(self):
        child = ordinary(); root = Connected(3, "root", [[]], [[]], [([1], [2], [3])], [[4]], [child])
        fork, leases = self.fork(root); adopted = fork.take()
        self.assertEqual(child.attention[0].key.references[0], 2); self.assertTrue(all(lease.closed for lease in leases))
        root.close(); self.assertEqual(adopted.nested[0].attention[0].key.references[0], 1)
        self.assertFalse(adopted.nested[0].attention[0].key.closed)
        adopted.close(); self.assertEqual(child.attention[0].key.references[0], 0)

    def test_missing_temporal_opaque_and_aliased_mutable_caches_refuse_instead_of_flattening(self):
        child = ordinary()
        invalid = [object(), Connected(3, "r", [[]], [[]], [None], [[4]], [child]),
                   Connected(3, "r", [[], []], [[], []], [([1], [2], [3]), ([1], [2], [3])], [[4], [4]], [child, child])]
        for source in invalid:
            with self.subTest(source=type(source).__name__), self.assertRaises(ValueError): self.fork(source)
        self.assertEqual(child.attention[0].key.references[0], 1)


if __name__ == "__main__": unittest.main()
