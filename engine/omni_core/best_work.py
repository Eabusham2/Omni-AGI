"""Optional native inference search without rewards or candidate side effects.

Agreement and sampled likelihood are search signals, not calibrated truth.
The scheduler interleaves branch steps over one immutable native decoder;
it makes no claim of simultaneous accelerator batches.
"""
from __future__ import annotations

import ast
import hashlib
import math
import re
import time
from dataclasses import dataclass
from typing import Callable


@dataclass(frozen=True)
class BranchSpec:
    index: int
    seed: int
    temperature: float
    ordinary: bool = False


def exploration_target(normalized_entropy: float, budget: int) -> int:
    if type(budget) is not int or not 1 <= budget <= 16:
        raise ValueError("Best Work exploration budget must be 1 through 16")
    if not math.isfinite(normalized_entropy):
        raise ValueError("Best Work entropy must be finite")
    value = max(0., min(1., normalized_entropy))
    return min(budget, 0 if value < .25 else 4 if value < .5 else 8 if value < .7 else 16)


def temperature_ladder(count: int) -> list[float]:
    if count <= 0:
        return []
    if count == 1:
        return [.20]
    return [.20 + (.95 - .20) * (index / (count - 1)) ** 1.35 for index in range(count)]


def static_verification(text: str) -> dict:
    """Inspect syntax without running candidate code or contacting services."""
    checks = []
    for language, source in re.findall(r"```([\w+-]*)[^\S\n]*\n(.*?)```", text, flags=re.DOTALL):
        if language.casefold() not in {"py", "python", "python3"}:
            continue
        try:
            ast.parse(source)
            checks.append({"kind": "python-syntax", "passed": True})
        except (SyntaxError, ValueError, TypeError, RecursionError) as error:
            checks.append({"kind": "python-syntax", "passed": False, "reason": type(error).__name__})
    return {"checks": checks, "status": "failed" if any(not check['passed'] for check in checks) else "syntax-valid" if checks else "unverified",
            "correctnessProven": False, "codeExecuted": False}


def candidate_agreement(candidates: list[dict]) -> list[float]:
    if not candidates:
        return []
    fingerprints = [hashlib.sha256(" ".join(str(candidate.get('text', '')).split()).encode()).hexdigest() for candidate in candidates]
    return [fingerprints.count(value) / len(fingerprints) for value in fingerprints]


def select_candidate(candidates: list[dict]) -> tuple[int, dict]:
    if not candidates:
        raise ValueError("Best Work has no completed native candidates")
    agreements = candidate_agreement(candidates)
    scores = []
    for candidate, agreement in zip(candidates, agreements):
        log_probability = candidate.get('meanLogProbability')
        likelihood = math.exp(min(0., float(log_probability))) if isinstance(log_probability, (int, float)) and math.isfinite(float(log_probability)) else 0.
        support = agreement * likelihood
        candidate.update(agreement=agreement, uncalibratedSupport=support)
        scores.append(support)
    eligible = [index for index, candidate in enumerate(candidates) if candidate['verification']['status'] != 'failed']
    if not eligible:
        chosen = next((index for index, candidate in enumerate(candidates) if candidate['ordinary']), 0)
        reason = 'ordinary-fallback-all-static-checks-failed'
    else:
        chosen = max(eligible, key=lambda index: (scores[index], candidates[index]['ordinary'], -candidates[index]['index']))
        reason = 'agreement-times-sampled-likelihood'
    return chosen, {"reason": reason, "correctnessProven": False, "calibratedConfidence": False,
                    "agreement": agreements[chosen], "uncalibratedSupport": scores[chosen]}


def run_best_work(*, seed: int, ordinary_temperature: float, normalized_entropy: float,
                  exploration_budget: int, start: Callable, finish: Callable,
                  cancelled: Callable[[], bool], steered: Callable[[], bool],
                  progress: Callable[[dict], None] | None = None) -> dict:
    """Run resource-admitted branches with one safe advance per round.

    ``start`` owns admission and returns (private session, branch metadata,
    release callback), or None when an *additional* branch cannot fit.
    ``finish`` reads native results; it must not dispatch host actions.
    Every admitted cache/lease is released, including interruption and errors.
    """
    if not math.isfinite(ordinary_temperature) or not 0 < ordinary_temperature <= .95:
        raise ValueError("Best Work ordinary sampling temperature must respect the 0.95 ceiling")
    desired = exploration_target(normalized_entropy, exploration_budget)
    active, completed, failures, releases = [], [], [], []
    explored_temperatures = []
    next_index, rounds = 0, 0
    admission_paused = False
    early_stopped = []
    terminal_reason = 'selected'
    last_progress_at = -math.inf

    def interrupted():
        return 'cancelled' if cancelled() else 'steered' if steered() else None

    def admit(temperature, ordinary):
        nonlocal next_index, admission_paused
        specification = BranchSpec(next_index, int(seed) + next_index * 7919, temperature, ordinary)
        opened = start(specification)
        if opened is None:
            if ordinary:
                raise RuntimeError("ordinary Best Work branch could not be admitted")
            admission_paused = True
            return False
        session, metadata, release = opened
        active.append((specification, session, metadata))
        releases.append((session, release))
        next_index += 1
        if not ordinary:
            explored_temperatures.append(temperature)
        return True

    def expand(count):
        ladder = temperature_ladder(count)
        choices = [value for value in ladder if all(abs(value - old) > 1e-10 for old in explored_temperatures)]
        for temperature in choices:
            if len(explored_temperatures) >= count or admission_paused or interrupted():
                break
            admit(temperature, False)

    try:
        reason = interrupted()
        if reason:
            terminal_reason = reason
            return {"interrupted": reason, "candidates": [], "selection": None, "admittedBranches": 0}
        admit(ordinary_temperature, True)
        expand(desired)
        while active:
            rounds += 1
            for specification, session, metadata in tuple(active):
                reason = interrupted()
                if reason:
                    terminal_reason = reason
                    return {"interrupted": reason, "candidates": completed, "selection": None, "admittedBranches": next_index}
                try:
                    event = session.advance()
                except Exception as error:
                    reason = interrupted()
                    if reason:
                        terminal_reason = reason
                        return {"interrupted": reason, "candidates": completed, "selection": None, "admittedBranches": next_index}
                    failures.append({"index": specification.index, "ordinary": specification.ordinary, "error": type(error).__name__, "message": str(error)})
                    active.remove((specification, session, metadata))
                    continue
                observed_at = time.perf_counter()
                if progress is not None and event is not None and observed_at - last_progress_at >= .05:
                    progress({"phase": "best-work", "active": True, "branch": specification.index, "ordinary": specification.ordinary,
                              "step": event['step'], "entropy": event['entropy'], "admittedBranches": next_index,
                              "completedBranches": len(completed), "concurrency": "cooperative-interleaved"})
                    last_progress_at = observed_at
                if session.done:
                    candidate = finish(specification, session, metadata)
                    candidate.update(index=specification.index, seed=specification.seed, temperature=specification.temperature,
                                     ordinary=specification.ordinary, meanLogProbability=session.mean_log_probability)
                    candidate['verification'] = static_verification(candidate['text'])
                    structural = candidate.pop('structuralChecks', [])
                    if structural:
                        candidate['verification']['checks'].extend(structural)
                        candidate['verification']['status'] = ('failed' if any(not item['passed'] for item in candidate['verification']['checks'])
                            else 'structurally-valid')
                    completed.append(candidate)
                    active.remove((specification, session, metadata))
            if len(completed) >= 2 and active:
                chosen, evidence = select_candidate(completed)
                if completed[chosen]['ordinary'] and evidence['uncalibratedSupport'] >= .95 and completed[chosen]['verification']['status'] != 'failed':
                    early_stopped = [specification.index for specification, _session, _metadata in active]
                    active.clear()
                    terminal_reason = 'agreement-convergence'
                    break
            if not active and completed and not admission_paused and len(explored_temperatures) < exploration_budget:
                _, evidence = select_candidate(completed)
                uncertainty = max(0., min(1., normalized_entropy))
                needs_more = uncertainty >= .7 or (len(completed) > 1 and evidence['agreement'] < 2. / 3.)
                if needs_more:
                    desired = min(exploration_budget, max(4, len(explored_temperatures) * 2))
                    expand(desired)
        if not completed:
            raise RuntimeError("all native Best Work candidates failed: " + "; ".join(item['message'] for item in failures))
        completed.sort(key=lambda candidate: candidate['index'])
        selected, evidence = select_candidate(completed)
        return {"interrupted": None, "candidates": completed, "selected": selected, "selection": evidence,
                "admittedBranches": next_index, "requestedExplorationBudget": exploration_budget,
                "admissionPaused": admission_paused, "failedBranches": failures, "schedulerRounds": rounds,
                "earlyStoppedBranches": early_stopped,
                "concurrency": "cooperative-interleaved", "candidateHostEffects": 0}
    except BaseException:
        terminal_reason = 'failed'
        raise
    finally:
        cleanup_error = None
        for session, release in reversed(releases):
            try:
                session.close()
            except BaseException as error:
                cleanup_error = cleanup_error or error
            finally:
                try:
                    release()
                except BaseException as error:
                    cleanup_error = cleanup_error or error
        if progress is not None:
            progress({"phase": "best-work", "active": False, "stopReason": 'failed' if cleanup_error else terminal_reason,
                "admittedBranches": next_index, "completedBranches": len(completed), "concurrency": "cooperative-interleaved"})
        if cleanup_error is not None:
            raise cleanup_error
