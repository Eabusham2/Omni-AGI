# Initial native receipt accounting

## Confirmed failure

The existing organic-memory restart fixture failed during initial curriculum
receipt validation, before its memory restart assertions. Inspection identified
one failing comparison: the seeded core had **332,919** logical connections;
the curriculum allocated a 24-by-24 recurrent block, adding **576** contacts and
bringing the live core to **333,495**. The validator excluded new repeated
projection contacts but incorrectly compared those new recurrent contacts with
the pre-training count.

The same observation confirmed matching post-training neural checksums, both
modality checksums, both imagination-selector checksums, all 436 native tensor
names/counts, and the recorded substrate totals. The fix does not reinterpret a
checksum failure as success.

## Correction

New sparse recurrent owners start with no allocated contacts. Initial-count
validation now subtracts their actual subsequently allocated contacts,
including repeated recurrent layers, once per owner. The full live count still
includes those contacts. Every sign, strength byte and named tensor remains
subject to the existing exact post-training checksum and receipt binding.
Coverage, architecture dimensions, transient-state reset and modality
provenance checks are unchanged.

Five tiny packed-owner regression cases pass without constructing or training
a brain. They cover base counts with sparse/repeated growth and shared owners,
re-sealed wrong initial counts, mutated recurrent signs/strengths, deleted
contacts, and failed record-coverage/architecture checks. The affected original
`test_continuous_scores_focus_and_afterimages_survive_restart` fixture also
passed, completing real tiny initial creation, subsequent experiences, save,
reload, and its existing memory-dynamics assertions in 275.237 seconds. No full
corpus or saved user brain is part of that check. Python compilation and the
focused whitespace check passed. This is receipt/restart evidence, not a
conversational-competence or full trained-capability claim.
