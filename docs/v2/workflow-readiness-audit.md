# Local workflow readiness audit — October 13

Scope: the existing `.github` policy/workflows and local Git evidence in this
checkout. The bundled deterministic maintainer-defense auditor completed with
exit0: **5 review leads — 2 medium, 2 low, 1 note; no critical/high findings**.
These are not findings of exploitation, compromise, or verified remote policy.
No GitHub workflow was dispatched, no package/release was made, and no token or
secret was read or printed.

This is historical October13 evidence. The credential-persistence correction
below was subsequently authorized and applied; current source evidence is in
[workflow-source-audit.md](workflow-source-audit.md). Its old pending-approval
wording is not an outstanding approval requirement.

| Rule | Evidence | Disposition |
| --- | --- | --- |
| MD-WF-006, medium | Historical publish checkout had `contents: write` without disabling credential persistence | Corrected subsequently: `persist-credentials: false`, explicit publication `GH_TOKEN`, current-repository origin check and process-only authenticated tag fetch preserve the commit/tag check. No live token or remote policy was inspected by this local audit. |
| MD-GOV-002, medium | No `.github/CODEOWNERS` | Governance suggestion, not evidence of a required feature being absent. A real responsible reviewer and intended review policy must be chosen; do not invent an owner or block product completion solely on this suggestion. |
| MD-GOV-005, low | No machine-readable dependency update policy | Suggest a reviewed update cadence/Dependabot configuration. Do not upgrade pins or add external automation as an unsolicited audit mutation. |
| MD-GOV-004, low | No structured issue form | Optional maintainer usability improvement; not a promised product feature or mandatory release blocker. |
| MD-GOV-006, note | Branch-protection expectations not documented locally | Actual rulesets, required checks, installed Apps, secrets and organization policy were not inspected. Local absence does not prove remote protection is disabled. |

The existing release workflow uses a verified commit, waits for its desktop and
mobile jobs, verifies artifact receipts/checksums, and refuses to replace an
existing release's assets. This source inspection does not prove those jobs
pass for the current dirty v2 tree. Package versions and the old v1 release
metadata/validators still need a deliberate v2 release preparation pass after
the implementation/acceptance gates. Do not call the historical v1 release a
v2 release or skip the pending cross-platform package evidence.

No generated patch was applied. The approved full goal still includes current
green CI and checksummed released packages; this local audit closes neither.

## Recent contract coverage wiring

`package.json` now chains `test:node:v2:contracts` after the existing
`test:node:ci` script. The ten selected existing files cover the new action
canvas/storage/ordinary documentation declaration, provider discovery/picker,
library presentation, read-only portable working-clock admission, notification
expiry, current control documentation, and the new constructor-free inherited
lost-ACK reconciliation case and current native App-frontier ID contract. They use source/presentation or tiny
filesystem/SQLite/fake-HTTP fixtures, not a constructed/forwarded/trained brain
or real API provider. All listed paths and the script chain were checked without
running the suites again. This is deliberate incremental v2 coverage, not a
claim that every v2 test is selected or that CI has already passed. The broader
native and package matrix remains pending.
