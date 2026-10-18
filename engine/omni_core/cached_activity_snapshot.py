"""Structural COW snapshot of one cache, no model/Torch/I/O on import.

The native caller validates closed cache types and supplies activity sharing,
small tensor copies and constructors. Never deepcopy a decoder or replay text.
"""


class CacheFork:
    def __init__(self, cache=None):
        self.cache, self.leases = cache, []
        self.closed = False

    def close(self):
        if self.closed: return
        if self.cache is not None:
            self.cache.close()
            self.cache = None
        for lease in reversed(self.leases): lease.release()
        self.leases.clear()
        self.closed = True

    def take(self):
        value, self.cache = self.cache, None
        self.close()  # Tensor objects/pages in value keep their real owner.
        return value


def fork_decoder_cache(cache, *, activity, tensor, workspace, attention, decoder, reserve_metadata):
    """Independent mutable containers; only immutable pages are shared.

    Every raw tensor callback returns (independent tensor, held reservation).
    On a partial failure all acquired page references/leases are released.
    """
    result = CacheFork()
    acquired = []
    try:
        lease = reserve_metadata(1024 + 512 * (len(cache.workspace) + len(cache.attention)))
        if lease is not None: result.leases.append(lease)
        def clone(value):
            if value is None: return None
            shared = activity(value)
            if shared is not None:
                acquired.append(shared)
                return shared
            copied, lease = tensor(value)
            if lease is not None: result.leases.append(lease)
            return copied
        chunks = [workspace(clone(chunk.queries), clone(chunk.mass), clone(chunk.weighted_values)) for chunk in cache.workspace]
        layers = [attention(clone(layer.key), clone(layer.value)) for layer in cache.attention]
        result.cache = decoder(cache.length, cache.signature, chunks, layers,
            clone(cache.hidden_sum), clone(cache.memory_hidden), clone(cache.neural_state))
        if result.leases and callable(getattr(result.leases[0], "mark_allocated", None)):
            result.leases[0].mark_allocated(1024 + 512 * (len(cache.workspace) + len(cache.attention)))
        return result
    except BaseException:
        if result.cache is None:
            for owner in reversed(acquired): owner.close()
        result.close()
        raise
