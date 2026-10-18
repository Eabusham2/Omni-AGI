# Optional provider controls: bounded Electron check

October 15, 2026. Actual `ApiTeacherPanel` and `ApiTeacherModelPicker`, mounted
in an explicitly synthetic Electron window with simulated API replies. No
provider, credential, Ollama server/model, neural worker, brain creation or
training was invoked. This verifies presentation and callback/receipt behavior,
not provider interoperability, billing, learned quality or packaged execution.

## Observed flow

1. Default hosted selection offers OpenAI, Claude, Gemini, OpenRouter and local
   Ollama. An empty key and model cannot submit. No model ID is invented.
2. Selecting Ollama removes the password and key controls. Its local inventory
   can be inspected, including the exact excluded cloud entry and reason.
   Listing does not select a model or claim offline execution.
3. Entering two synthetic questions and a model enables submission. A simulated
   lost start acknowledgement exposes Retry same submission. The retry uses the
   identical request ID; both receipts resolve to one unique simulated job.
4. Pause exposes Resume. Resume restores Pause/Stop. Stop removes the active
   controls without a stuck running indicator. These are simulated host ACKs,
   not proof of actual provider cancellation or neural checkpointing.
5. Dark/380px content and light/wide views were inspected and scrolled. A
   stretched provider selector, weak Stop text and weak disabled distinction
   were found and fixed using the existing component classes. Container-width
   reflow keeps the compact fields in one column.
6. Switching to OpenRouter clears the old local model and exposes its own
   empty key field. No key was entered or stored. Submission stays disabled.

The preview exited normally with code0. Saved screenshots were reopened before
acceptance. No full accessibility or all-theme compliance is inferred.

## Accepted captures

- `test-results/api-provider-ui-2026-10-15/01-compact-dark-stopped.png`:
  lower controls, stopped state and exact synthetic retry/control receipts.
- `test-results/api-provider-ui-2026-10-15/02-compact-dark-local-fields.png`:
  upper local provider/model fields and key-free connection information.
- `test-results/api-provider-ui-2026-10-15/03-openrouter-light-key-gate.png`:
  aligned wide hosted fields, empty model/key and disabled submission.

The compact captures cover different scroll positions; neither is a full-page
capture. Earlier stretched-selector diagnostics are not final output.

## Separate source evidence

`apiTeacherPanel.test.tsx` has5 new pure/static cases; `apiProviderPreload.test.ts`
has1 fake bridge case. Provider HTTP/discovery and stale authority cases are
recorded in `provider-support.md`. Durable teacher SQLite/learning-ACK custody
has separate focused tests. Complete Node and renderer TypeScript checks passed
at this source point; no broad old suite or remote CI was rerun for this slice.
