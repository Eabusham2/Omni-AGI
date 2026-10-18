# Native improvement tools

## Implemented contract

Explicit `neural`, `data`, `substrate`, or `architecture` proposals to
`source.self-modify.propose` use the existing `EvolutionController` and protected
native evaluator. An absent kind or `source` retains the Git source workflow.
Unknown kinds and incompatible source/native operands fail visibly; they do not
silently enter a different workflow. This exposes existing candidate mechanisms
to permissioned custom tools/modules, not an arbitrary whole-cortex editor or a
second intelligence implementation.

The proposal returns a durable queued job before waiting for the tracked brain
writer. `native-status` and `native-stop` use its exact job ID. Real run and
candidate IDs are attached only after the existing controller creates them.
Exact declared operands, request hashes, errors, and committed outcomes live in
the existing evolution archive. Pending records copied or encountered after
restart are historical evidence, never a restored permission grant.

The service retains the current main-owned source, permission, and abort scope
in RAM. It checks owner and authority after acquiring the brain writer and
before controller effects. The loaded-only supervisor binds the same existing
child, awaits the current authority check inside its acquired serial slot, then
checks child identity and cancellation again before dispatch. It does not call
Start, preempt a worker, or replace one after waiting. The worker's existing
`_get(..., allow_load=False)` separately refuses an absent resident brain.

Explicit Load/current activity boundary events can retry a still-authorized
waiting request. Permission change, owner recovery/suspension, module Stop, and
shutdown revoke its live scope. There is no periodic scan that restarts dormant
archived work. Successful reviewed native promotion still activates its actual
replacement state through the existing promotion implementation.

Recursive observation of a completed experiment runs only after releasing the
outer writer, through the ordinary loaded-only structured-experience path. It
rechecks authority inside that path's own writer and before neural admission.
An unavailable/revoked observation remains pending; a real accepted neural ACK
is not erased or replayed merely because Stop arrived afterward.

## Evidence and boundaries

- `tests/nativeCandidateTool.test.ts`: 11 distinct new fake-controller/FIFO
  cases passed in 17 ms: routing, strict operands, queued ownership, request
  identity, revoked access, explicit unloaded hold, dormant restart, Stop, real
  controller run correlation, post-writer delivery, and late promotion.
- `tests/nativeCandidateController.test.ts`: four affected new fake-engine
  cases passed in the final transport correction (63 ms). Authorized dispatch
  uses only the loaded transport and fails when it is absent. The earlier test
  iteration caught a genuine stage callback returning no archived record; that
  source defect was repaired, not recorded as a pass.
- `tests/nativeCandidateSupervisor.test.ts`: two constructor-free fake-pipe
  cases passed in 4 ms. They cover queued async admission, same-child refusal,
  revoked authority, no Start/preempt fallback, and retaining a real late ACK.
- `tests/loadedExperienceAuthority.test.ts`: three constructor-free
  fake-service cases passed in 5 ms at the recorded source point. They cover
  writer-acquisition admission, no preflight Load, and accepted-ACK retention.

These are scoped source/fake-host results, not actual model loading, training,
native candidate evaluation, successful learned self-improvement, or packaged
acceptance. The expressly deferred new-brain/merged-resume proof remains
deferred. Full R243 integration and release qualification remain open.
