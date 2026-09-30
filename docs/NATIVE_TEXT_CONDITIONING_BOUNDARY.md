# Native text boundary and recurrent Ponder

Native chat no longer forces an ASCII first token. The byte-token text channel
permits natural EOS at the initial boundary and valid UTF-8 scalar encodings.
NUL is excluded because the existing saved-message transport rejects it; no
language or answer preference is applied.
Its filter is wire syntax only: no language, persona or answer preference is
inserted. Partial scalar bytes are not streamed; cooperative Stop/Steer retains
only the exact emitted valid prefix. The existing typed no-reply receipt handles
natural EOS without inventing assistant prose or marking human cancellation.

Voluntary mid-generation Ponder returns its refined same-cortex recurrent cue
to the text loop. Subsequent tokens use that conditioning. Old attention/KV and
workspace statistics are closed, then the exact retained prefix is recomputed
under the new cue; cached and full-prefix routes follow the same update protocol.
The token sampled before that safe boundary is not retroactively rewritten.
Ponder can be selected again at a later genuine prefix. Its exact decision
replay stays idempotent, while repeated external tool effects remain suppressed
across the turn. This does not force or schedule a Ponder choice.

Zero-visible warm Steer retains the actual predecessor human request in a
private main-owned payload bound to brain, original turn/hash, successor and
native attention epoch. Only actual HUMAN role-token segments enter the next
temporary prompt; no assistant reply, behavioral instruction or ledger-history
lookup is manufactured. Chained zero-visible requests retain their order, while
already saved partial turns are not duplicated. The payload is not learned or
stored as a completed predecessor turn. Native Fresh epoch mismatch discards it.
Complete carry plus current input must fit the selected context and admitted
token RAM or pause explicitly. For ordinary input, older prompt words yield room
without erasing previously learned state; a new message itself larger than the
selected window blocks Send rather than being cropped or enlarging the setting.

Validation uses pure UTF-8 state and scripted-logit/cache method fixtures only.
These demonstrate syntax, ownership and conditioning plumbing, not learned
language quality, meaningful thought or native useful-capability qualification.
No brain/model constructor, neural inference, training, app or CI run is part of
this correction.
