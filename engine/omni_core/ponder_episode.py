"""Invocation-owned operational Ponder provenance, without neural imports.

These frames report selected computation, not private thought text. Only an
active, registered Ponder invocation can own a child action; wall-clock
proximity and model-supplied tool arguments never establish ancestry.
"""
from __future__ import annotations

import copy
import math
import time


def validate_ponder_episode_frame(value):
    if (not isinstance(value, dict) or set(value) != {"state", "actionId", "action"}
            or value["state"] not in {"started", "finished"} or not isinstance(value["actionId"], str)
            or len(value["actionId"]) != 32 or any(character not in "0123456789abcdef" for character in value["actionId"])):
        raise ValueError("invalid native Ponder episode frame")
    action = value["action"]
    if (not isinstance(action, dict) or action.get("kind") != "ponder"
            or action.get("actionId") != value["actionId"] or action.get("parentPonderActionId") is not None
            or action.get("selectionPhase") not in {"pre-speech", "mid-generation"}
            or not isinstance(action.get("selectionStep"), int) or isinstance(action["selectionStep"], bool)):
        raise ValueError("native Ponder episode has a different selected owner")
    if value["state"] == "finished":
        arguments = action.get("arguments", {})
        trace = arguments.get("ponderTrace", {})
        if (arguments.get("handledInTurn") is not True or not isinstance(trace, dict)
                or trace.get("activated") is not True or trace.get("activated_by") != "learned-action-head"
                or trace.get("private") is not True or trace.get("visibleMagicTags") is not False
                or trace.get("phase") != action["selectionPhase"]
                or not isinstance(trace.get("completed"), bool)
                or arguments.get("completedInTurn") is not trace["completed"]
                or isinstance(trace.get("elapsed_ms"), bool) or not isinstance(trace.get("elapsed_ms"), (int, float))
                or not math.isfinite(trace["elapsed_ms"]) or trace["elapsed_ms"] < 0
                or trace.get("completed") != (trace.get("stop_reason") in {"converged", "learned-action-transition"})
                or trace.get("stop_reason") not in {"converged", "learned-action-transition", "steered", "suspended", "resource-pressure", "cancelled", "failed"}):
            raise ValueError("native Ponder completion lacks its measured invocation")
        incomplete = trace.get("measurementIncomplete") is True
        if incomplete:
            if trace["completed"] or trace["stop_reason"] not in {"cancelled", "failed"} or trace.get("passes") is not None or trace.get("final_delta") is not None:
                raise ValueError("incomplete Ponder measurement cannot claim completed passes")
        elif (not isinstance(trace.get("passes"), int) or isinstance(trace["passes"], bool) or trace["passes"] < 0
                or isinstance(trace.get("final_delta"), bool) or not isinstance(trace.get("final_delta"), (int, float))
                or not math.isfinite(trace["final_delta"]) or trace["final_delta"] < 0):
            raise ValueError("native Ponder pass measurement is invalid")
    return value


class NativePonderEpisode:
    def __init__(self, ledger, action, emit=None):
        self.ledger, self.action, self.emit = ledger, action, emit
        self.started = None
        self.finished = False

    def __enter__(self):
        if (self.action.get("kind") != "ponder"
                or not any(value is self.action for value in self.ledger.actions.values())
                or self.ledger.active_ponder_action_id is not None
                or self.action.get("arguments", {}).get("handledInTurn") is True):
            raise ValueError("Ponder episode requires its own registered inactive parent")
        self.started = time.perf_counter()
        self.ledger.active_ponder_action_id = self.action["actionId"]
        try:
            self._emit("started")
        except BaseException:
            self.ledger.active_ponder_action_id = None
            raise
        return self

    def _emit(self, state):
        if self.emit is not None:
            self.emit("ponder-episode", {"state": state, "actionId": self.action["actionId"],
                "action": copy.deepcopy(self.action)})

    def select(self, action, *, step, phase):
        # Private and visible candidates perform the same neural argument
        # computation and local proposal bookkeeping (including native Stop).
        # Only the visible invocation has an executable outward emitter.
        registered = self.ledger.register(action, step=step, phase=phase,
            parent_ponder_action_id=self.action["actionId"])
        if registered is not None and self.emit is not None:
            self.emit("action", {"actionId": registered["actionId"], "action": registered})
        return registered

    def finish(self, trace):
        if self.finished or self.ledger.active_ponder_action_id != self.action["actionId"]:
            raise ValueError("Ponder episode completion has no active invocation")
        self.action["arguments"] = {**dict(self.action.get("arguments", {})),
            "handledInTurn": True, "completedInTurn": trace["completed"], "ponderTrace": dict(trace)}
        self.finished = True
        self._emit("finished")

    def __exit__(self, exception_type, exception, traceback):
        try:
            if not self.finished:
                # Setup/cleanup can fail before the loop's measured-finally
                # callback. Duration is known; pass/delta totals are not.
                # Never manufacture zero or a successful completion.
                self.finish({"activated": True, "activated_by": "learned-action-head",
                    "completed": False, "passes": None, "final_delta": None, "measurementIncomplete": True,
                    "elapsed_ms": max(0., time.perf_counter() - self.started) * 1000.,
                    "stop_reason": "cancelled" if exception_type and exception_type.__name__ == "ChatGenerationCancelled" else "failed",
                    "phase": self.action["selectionPhase"], "private": True, "visibleMagicTags": False})
        finally:
            self.ledger.active_ponder_action_id = None
        return False
