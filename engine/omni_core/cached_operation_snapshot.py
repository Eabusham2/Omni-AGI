"""Recursive closed-cache COW, no model/Torch/I/O on import.

Callbacks supply admitted immutable-page forks, independent live tensor copies
and exact cache constructors. No decoder deepcopy, learned weights or text is
stored here; arbitrary nested connected caches keep their real tree shape.
"""
from __future__ import annotations

from .cached_activity_snapshot import CacheFork


def fork_operation_cache(cache, *, kind, activity, tensor, workspace, attention, ordinary, connected, reserve_metadata):
    result, acquired, assembled, seen = CacheFork(), [], {}, set()
    stack = [(cache, False)]
    def clone(value):
        if value is None: return None
        shared = activity(value)
        if shared is not None: acquired.append(shared); return shared
        copied, lease = tensor(value)
        if lease is not None: result.leases.append(lease)
        return copied
    def chunks(values): return [workspace(clone(value.queries), clone(value.mass), clone(value.weighted_values)) for value in values]
    def layers(values): return [None if value is None else attention(clone(value.key), clone(value.value)) for value in values]
    try:
        while stack:
            node, finish = stack.pop()
            mode = kind(node)
            if mode not in {"ordinary", "connected"}: raise ValueError("opaque inference cache cannot be speculatively copied")
            if not finish:
                if id(node) in seen: raise ValueError("aliased/cyclic mutable cache ownership is unsupported")
                seen.add(id(node))
                size = 1024 + 512 * (len(node.workspace) + len(node.attention) +
                    (len(node.temporal) + len(node.hidden_sum) + len(node.nested) if mode == "connected" else 0))
                lease = reserve_metadata(size)
                if lease is not None: result.leases.append(lease); lease.mark_allocated(size)
                stack.append((node, True))
                if mode == "connected":
                    for child in reversed(node.nested):
                        if child is not None: stack.append((child, False))
                continue
            if mode == "ordinary":
                value = ordinary(node.length, node.signature, chunks(node.workspace), layers(node.attention),
                                 clone(node.hidden_sum), clone(node.memory_hidden), clone(node.neural_state))
            else:
                if not len(node.workspace) == len(node.attention) == len(node.temporal) == len(node.hidden_sum) == len(node.nested):
                    raise ValueError("connected cache lost a regional activity owner")
                temporal = []
                for state in node.temporal:
                    if not isinstance(state, tuple) or len(state) != 3: raise ValueError("connected retained frontier needs complete temporal triples")
                    temporal.append(tuple(clone(value) for value in state))
                value = connected(node.length, node.signature, [chunks(group) for group in node.workspace],
                    [layers(group) for group in node.attention], temporal, [clone(item) for item in node.hidden_sum],
                    [None if child is None else assembled.pop(id(child)) for child in node.nested])
            assembled[id(node)] = value
        result.cache = assembled.pop(id(cache))
        return result
    except BaseException:
        if result.cache is None:
            for owner in reversed(acquired): owner.close()
        result.close()
        raise
