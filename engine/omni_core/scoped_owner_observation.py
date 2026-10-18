"""Process-private packed-owner epoch lease; no Torch/model/I/O on import."""
from __future__ import annotations

import uuid


class ScopedLearningOwners:
    def __init__(self, pager, modules):
        self.pager, self.id = pager, uuid.uuid4().hex
        self.owners, self.lease, self.closed = [], None, False
        with pager._lock:
            if pager._closed: raise RuntimeError("packed owner pager is closed")
            selected = list(dict.fromkeys(modules))
            size = sum(256 + len(pager._owners[module].cold) * 192 for module in selected)
            reserve = getattr(pager.shared_resource_policy, "reserve_ram", None)
            self.lease = reserve(size, "scoped packed-owner version metadata") if callable(reserve) else None
            try:
                for module in selected:
                    owner = pager._owners[module]
                    if not owner.loaded: raise RuntimeError("scoped observation requires completely loaded packed owners")
                    if owner.scoped_learning_observers or pager._learning_revision_active:
                        pager._observe_mutations(module, owner)
                    else: pager._rebind_learned_tokens(module, owner)
                    owner.scoped_learning_observers += 1
                    self.owners.append((module, owner))
                if self.lease is not None: self.lease.mark_allocated(size)
            except BaseException:
                self.close(); raise

    def snapshot(self):
        with self.pager._lock:
            if self.closed or self.pager._closed: raise RuntimeError("scoped packed-owner observation is retired")
            result = {}
            for module, owner in self.owners:
                if self.pager._owners.get(module) is not owner or not owner.loaded:
                    raise RuntimeError("scoped packed owner was retired or replaced")
                self.pager._observe_mutations(module, owner)
                result[str(id(module))] = int(owner.scoped_learning_revision)
            return result

    def close(self):
        with self.pager._lock:
            if self.closed: return
            for _module, owner in self.owners:
                owner.scoped_learning_observers -= 1
                if owner.scoped_learning_observers < 0: raise RuntimeError("scoped packed-owner lease underflow")
                if not owner.scoped_learning_observers and not self.pager._learning_revision_active:
                    owner.learned_tokens.clear()
            self.owners.clear()
            if self.lease is not None: self.lease.release(); self.lease = None
            self.closed = True

    def __enter__(self): return self
    def __exit__(self, *_args): self.close()
    def __del__(self):
        try: self.close()
        except Exception: pass
