"""Actual-turn sensory ownership, separate from private inference candidates.

Only an explicit world boundary calls ``poll``. A private branch has no
reference to this consumer. Bindings admitted during Ponder/search remain
pending until the real decoder uses the resulting canonical memory bias.
Payload and retained-binding allocations remain owned by the worker inbox.
"""
from __future__ import annotations


class SensoryTurnBoundary:
    def __init__(self, provider, episode_assembly_id):
        self.consume = provider.live_sensory_conditioning
        self.pending = getattr(provider, "live_sensory_pending", lambda: True)
        self.episode_assembly_id = str(episode_assembly_id or "")
        self.bindings = []

    def has_pending(self):
        return bool(self.pending())

    def poll(self, current, step):
        if not self.has_pending():
            return {}
        result = self.consume(current, step, self.episode_assembly_id)
        bindings = list(result.get("observationBindings", []))
        if bindings:
            self.bindings.extend(bindings)
        return result

    def condition(self, current, step):
        result = self.poll(current, step)
        current = result.get("memoryBias", current)
        return self.condition_pending(current)

    def condition_pending(self, current):
        """Take only already-admitted proof, without consuming another frame.

        The actual decoder also uses this at Unload before its first forward;
        its cursor persists the bias and these exact bindings together.
        """
        if not self.bindings:
            return {}
        bindings, self.bindings = self.bindings, []
        return {"memoryBias": current, "observationBindings": bindings}
