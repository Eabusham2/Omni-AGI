# Chat output cadence

Ordinary streamed output is coalesced once per animation frame, not paced at
96 characters per frame. The renderer now publishes the entire received burst
on its next frame. This removes a default presentation backlog without changing
sampling, native computation, token counts, source text or the existing chat UI.

`createTextFrameBatcher` still accepts an explicit finite budget for an opt-in
display-pacing consumer. Such pacing never awaits or throttles the native worker.
It preserves Unicode surrogate pairs, rejects invalid budgets instead of
scheduling a never-draining loop, and flushes all remaining text immediately at
the caller's terminal boundary. Reset retires only scheduled presentation work.

Four new `tests/chatFrameBudget.test.ts` cases passed in 3 ms: an exact large
multi-delta burst publishes in one default frame, explicit pacing keeps a
surrogate pair intact, terminal flush preserves the paced suffix, and invalid
budgets are refused. Existing closed streaming suites were not rerun.

This is focused source evidence, not a measured increase in model TPS, a new
readable-speed Ponder control, or full UI/navigation qualification. The
separately implemented clipboard portion of R090 is documented in
[Instance and host clipboards](clipboard.md), with its own source-only evidence
and live-host limits. No model or corpus ran for this cadence check.
