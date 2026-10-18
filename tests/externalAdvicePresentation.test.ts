import { expect, it } from "vitest";
import { advicePanelActive, emptyAdvicePanel, mergeAdvicePanel } from "../src/renderer/src/externalAdvicePresentation";
import type { ExternalAdviceRequest } from "../src/shared/externalAdvice";

const receipt = (state: ExternalAdviceRequest["state"], response = ""): ExternalAdviceRequest => ({
  format: "omni-external-advice-request", version: 1, id: "fixture-request", brainId: "fixture-owner", channel: "assistant", provider: "openai", model: "fixture-model",
  state, response, prompt: "Fixture prompt", requestText: "Fixture request", createdAt: "2026-10-13T00:00:00.000Z", updatedAt: "2026-10-13T00:00:00.000Z",
  settingsRevision: 1, authorityRevision: 1, authorityOwnerEpoch: "fixture-device", maxOutputTokens: 2048, contextSha256: "a".repeat(64), previewSha256: "a".repeat(64), requestSha256: "a".repeat(64),
  advisoryOnly: true, automaticLearning: false, hostActionAuthority: false, credentialsStoredInBrain: false,
  context: { format: "omni-external-advice-selected-context", version: 1, brainId: "fixture-owner", channel: "assistant", selection: { includeTask: false, includeActivity: false, maxCharacters: 0 },
    items: [], includedCharacters: 0, omittedItems: 0, truncated: false }
});

it("does not treat historical pending state as a live provider or relock a completed panel from stale history", () => {
  const queued = receipt("queued"), streaming = receipt("streaming");
  const orphan = mergeAdvicePanel(emptyAdvicePanel, [{ request: queued, active: false }]);
  expect(advicePanelActive(orphan, "assistant")).toBe(false);
  const live = mergeAdvicePanel(orphan, [{ request: streaming, active: true }]);
  expect(advicePanelActive(live, "assistant")).toBe(true);
  const stale = mergeAdvicePanel(live, [{ request: queued, active: false }]);
  expect(advicePanelActive(stale, "assistant")).toBe(true); expect(stale.entries[0]?.state).toBe("streaming");
  const complete = mergeAdvicePanel(live, [{ request: receipt("complete", "Actual streamed reply"), active: false }]);
  const staleAgain = mergeAdvicePanel(complete, [{ request: streaming, active: true }]);
  expect(advicePanelActive(staleAgain, "assistant")).toBe(false); expect(staleAgain.entries[0]?.response).toBe("Actual streamed reply");
});
it("keeps received text distinct from the saved prefix without mutating the original receipt", () => {
  const request = receipt("streaming", "Received but not all saved");
  const state = mergeAdvicePanel(emptyAdvicePanel, [{ request, active: true, persistedCharacters: 4 }]);
  expect(state.persisted[request.id]).toBe(4); expect(state.entries[0]?.response).toBe(request.response);
  expect(request).not.toHaveProperty("persistence"); expect(request.state).toBe("streaming");
  const terminal = mergeAdvicePanel(state, [{ request: receipt("failed", request.response), active: false, persistedCharacters: 4 }]);
  expect(advicePanelActive(terminal, "assistant")).toBe(false); expect(terminal.persisted[request.id]).toBe(4);
});
