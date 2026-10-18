# Native sparse synapse state

The recurrent LIF/STDP router uses allocated 64×64 blocks. It does not allocate
the entire potential neuron-pair matrix. Each block is owned by the existing
RAM-first router pager, which can move its registered fields to the designated
storage pool.

## Learned connections

Every allocated contact has:

- a packed two-bit ternary sign;
- one learned byte: low four bits encode strength minus one (`1..16`), and high
  four bits encode resistance to change (`0..15`);
- its existing signed int16 timing eligibility and stability/usage controls.

The forward contribution is ternary sign times integer strength. Controls are
not a floating-point shadow copy of the learned weights. Actual state cost is
reported: packed signs plus eleven bytes per allocated contact for strength,
eligibility, stability and usage, plus registered-block metadata.

When reinforcement reaches a contact's strength representation limit and its
existing parallel contacts cannot absorb another unit, another independent
contact is allocated. Parallel layers use keys such as `r0_c0_p1`. One module
stores a block of contacts, with separate sign, strength, eligibility and usage
at each position. There is no product limit on parallel layers or connectivity;
RAM, scratch and storage admission still apply before allocation.

## STDP and decay

Causal presynaptic-before-postsynaptic activity gives positive timing pressure;
the reverse order gives negative pressure. The original timing traces and
stability equation remain in use. Signed integer eligibility now retains
subthreshold evidence rather than clearing it whenever a sign does not change.
Unrelated active pairs do not erase another pair's waiting eligibility.

The threshold is `256 × (1 + resistance)`. Each crossing changes strength by one,
or crosses through zero before changing sign. Consistent reinforcing updates
increase integer resistance, while opposing updates reduce it. Eligibility is
kept within the bound for the current resistance. Ordinary decay reduces one
strength unit before erasing a connection. Parallel contacts have distinct decay
positions and their own controls.

The router reports unique active pairs, contact updates and new contacts
separately. Recurrence includes every allocated active contact in canonical
order, including inhibitory contacts and repeated paths.

## Persistence and recovery

Sparse layout version 2 requires connection-state bytes. A version-2 checkpoint
missing them is rejected. Native version-1 sparse or dense checkpoints migrate
with unit strengths and zero new resistance; all existing signs, timing vectors,
eligibility, stability, usage and counters are preserved.

The bounded loader streams fields into private mutable backing. It does not
adopt immutable checkpoints as writable mappings. Block growth and every field
change participate in before-image rollback. Failed admission or cancellation
restores committed contacts, controls and topology, including the parallel-layer
index. Strength-only changes participate in parameter diagnostics and packed
rollback.

Population growth copies strength bytes and exact repeated-contact identities
for both input/output projections, alongside recurrent contact layers and the
existing activity state.

## Verification boundary

`test_weighted_sparse_router.py`, `test_sparse_router_state.py`,
`test_sparse_router_replay.py` and `test_router_state_paging.py` cover tiny
arithmetic/storage fixtures: signed recurrence, incremental timing, weakening,
resistance, independent repeated paths, migration, growth, paging, diagnostics
and exact rollback. One fixture constructs two tiny projections to check their
growth copy; no full brain or training-corpus run is performed.

These tests establish the implemented storage and update equations. Useful
learned behavior and large trained-brain performance require the separately
deferred training/qualification run. The overall v2 checkpoint/cursor migration
also requires the application-level binding audit, beyond these router tests.
