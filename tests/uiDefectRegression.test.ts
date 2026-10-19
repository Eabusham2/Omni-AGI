import { readFileSync } from "node:fs";
import { resolve } from "node:path";
import { describe, expect, it } from "vitest";

const root = resolve(import.meta.dirname, "..");
const app = readFileSync(resolve(root, "src/renderer/src/App.tsx"), "utf8");
const styles = readFileSync(resolve(root, "src/renderer/src/styles.css"), "utf8");
const responseCounter = readFileSync(resolve(root, "src/renderer/src/ResponseOutputCounter.tsx"), "utf8");
const submitControl = readFileSync(resolve(root, "src/renderer/src/ComposerSubmitControl.tsx"), "utf8");
const map = readFileSync(resolve(root, "src/renderer/src/WholeBrainMap.tsx"), "utf8");
const mapViewport = readFileSync(resolve(root, "src/shared/wholeBrainViewport.ts"), "utf8");
const learning = readFileSync(resolve(root, "src/renderer/src/LearningWorkspace.tsx"), "utf8");
const chatWorkspaceActivity = readFileSync(
  resolve(root, "src/renderer/src/chatWorkspaceActivity.ts"),
  "utf8"
);

describe("screenshot defect regressions", () => {
  it("describes context residency without denying real cold-attention spill", () => {
    expect(app).toContain("hot activity in RAM, cold attention can use the designated pool");
    expect(app).not.toContain("never paged to storage");
  });

  it("bounds diagnostics at the toast and compact job surfaces", () => {
    expect(app).toContain("const presentedMessage = conciseUiMessage(message)");
    const toast = readFileSync(resolve(root, "src/renderer/src/ToastNotice.tsx"), "utf8");
    expect(app).toContain("<ToastNotice");
    expect(toast).toContain("className=\"toast__message\"");
    expect(app).toContain("className=\"ui-error-copy\"");
    expect(app).toContain("conciseUiMessage(activeJob.error)");
    expect(styles).toContain("overflow-wrap: anywhere");
  });

  it("keeps Training and Brain Map navigable, retaining the previous consistent map during refresh", () => {
    expect(app).toContain('onBack={() => onView("chat")}');
    expect(app).toContain('aria-label="Back to conversation"');
    expect(app).toContain('<BrainMapWorkspace brain={brain} onBack={() => onView("chat")} />');
    expect(map).toContain('<button type="button" className="button" onClick={onBack}>Back to chat</button>');
    expect(map).toContain("hydrateWholeBrainViewport(api, brainId");
    expect(map).toContain('aria-label="Connected brain map"');
    expect(map).toContain("disabled={!api || busy}");
    const verifySnapshot = map.indexOf("projectWholeBrainInspection({ pages: [fresh.overview");
    const publishSnapshot = map.indexOf("setOverview(fresh.overview)");
    expect(verifySnapshot).toBeGreaterThan(-1);
    expect(publishSnapshot).toBeGreaterThan(verifySnapshot);
    expect(mapViewport).toContain("The brain changed while refreshing. The previous consistent map was kept.");
    expect(map).not.toContain('id: "substrate-overview"');
  });

  it("uses durable topology totals without calling cumulative updates live connections", () => {
    expect(app).toContain("persistedConnections: brain.substrateTotals?.synapses");
    expect(app).toContain("cumulativeUpdates: brain.neuralUpdates");
    expect(app).toContain("connections.topologyPending");
    expect(app).toContain("unique live connections");
    expect(app).toContain("cumulative updates");
    expect(app).toContain("compactNumber(connections.uniqueConnections)");
    expect(app).toContain("compactNumber(connections.cumulativeUpdates)");
    const counts = readFileSync(resolve(root, "src/renderer/src/uiPresentation.ts"), "utf8");
    expect(counts).toContain("const uniqueConnections = queried ?? persisted ?? mirrored");
    expect(counts).toContain("const cumulativeUpdates = uiCount(input.cumulativeUpdates) ?? 0");
    expect(app).not.toContain("Graph indexing connection updates…");
  });

  it("exposes exact keyboard semantics and a visible token-used counter", () => {
    expect(app.includes('aria-keyshortcuts="Enter Shift+Enter Control+Enter Meta+Enter"')).toBe(true);
    expect(app.includes('className="composer__input-count" title={composerTelemetryAriaLabel}')).toBe(true);
    expect(app.includes('aria-label={`${draftTokenCount.toLocaleString()} input tokens. ${composerTelemetryAriaLabel}`}')).toBe(true);
    expect(app.includes("utf8DraftTokenCount(input)")).toBe(true);
    expect(submitControl).toContain("Queue message · hold Ctrl or ⌘ to steer");
    expect(submitControl).toContain('active ? steering ? "Steer" : "Queue" : "Send"');
    expect(app.includes("pendingChatOutputPresentation({")).toBe(true);
    expect(app.includes("<ResponseOutputCounter presentation={responseOutputPresentation}")).toBe(true);
    expect(responseCounter).toContain("presentation.label");
    expect(responseCounter).toContain("response-token-counter--${presentation.phase}");
    expect(responseCounter).toContain("title={presentation.ariaLabel}");
    expect(styles).toContain(".response-token-counter--loading");
  });

  it("settles terminal chat activity and preserves immediate Steer receipts", () => {
    expect(app).toContain("preserveSteeredChatMessage(current, turnMetadata.replacesTurnId)");
    expect(app).toContain("settleOptimisticChatTurn(current, event.turnId, terminalState)");
    expect(app).toContain('message-failed-label">Not sent · retry ready');
    expect(app).toContain('data-turn-state={');
    expect(app).toContain('await persistDeliveryReceipt(turnId, "pending")');
    expect(app).toContain('mergeCommittedChatMessages(persistedMessages, sessionCommittedMessages)');
    expect(app).not.toContain(
      'current.filter((message) => message.id !== optimisticHuman.id)'
    );
    expect(app).not.toContain(
      'current.filter((message) => !message.id.startsWith("pending-"))'
    );
    expect(app).toContain("const finishGeneration = (turnId: string");
    expect(app).toContain("advanceChatGenerationPhase(previousPhase, event)");
    expect(app).not.toContain(
      "window.omni.chat.cancel(brain.id, turnMetadata.replacesTurnId)"
    );
    expect(app).toContain("Steer is a warm-worker handoff");
    expect(styles).toContain(".message--failed .message__content");
  });

  it("retains final output independently while its atomic save commits", () => {
    expect(app).toContain('event.type === "chat-phase"');
    expect(app).toContain("const presentation = replyCompleteLearningPresentation(event)");
    expect(app).toMatch(
      /event\.type === "chat-phase"[\s\S]{0,600}finishGeneration\(event\.turnId, event\.createdAt, !event\.turnCommitted\)/
    );
    expect(app).toContain("retainUncommittedChatOutput(");
    expect(app).toContain('event.type === "chat-reply-committed"');
    expect(app).toContain("Reply complete · save pending, not committed");
    expect(app).toContain('const active = sending || Boolean(activeTurnIdRef.current)');
    expect(app).toMatch(/if \(active\) \{\s*queueCurrentTurn\(\)/);
    expect(app).not.toContain("setReplyCompleteLearning(null)");
    expect(app).toContain("if (activeTurnIdRef.current !== turnId) return;");
  });

  it("preserves chat across navigation and exposes explicit busy-brain choices", () => {
    const mountedChat = app.indexOf("<ChatWorkspace");
    const offChatView = app.indexOf('{view !== "chat" ? (', mountedChat);
    expect(mountedChat).toBeGreaterThan(-1);
    expect(offChatView).toBeGreaterThan(mountedChat);
    expect(app.slice(mountedChat, offChatView)).toContain('hidden={view !== "chat"}');
    expect(app.slice(mountedChat, offChatView)).toContain("onActivityChange={setChatActivity}");
    expect(app).toContain("if (hidden || !followOutputRef.current) return;");
    expect(app).toContain("const userTimelineScrollRef = useRef(false);");
    expect(app).toContain("if (hidden || !userTimelineScrollRef.current) return;");
    expect(app).toContain('className="workspace-header__chat-status"');
    expect(app).toContain("chatActivityStatus.headerLabel");
    expect(app).toContain("<ComposerSubmitControl active={turnActive}");
    expect(app).toContain("onSend={submitOrdinaryTurn} onSteer={() => void steerCurrentTurn()}");
    expect(submitControl).toContain('className="send-button composer__submit"');
    expect(submitControl).toContain('active && canSteer && (event.ctrlKey || event.metaKey) ? onSteer() : onSend()');
    expect(submitControl).not.toContain('className="composer__steer-choice"');
    expect(app).toContain('className="chat-tool-status" role="status"');
    expect(app).toContain("Cancel the exact job in its action details.");
    expect(app).toContain("onStop={() => void cancelChatTool()}");
    expect(submitControl).toContain('className="composer__stop-choice"');
    expect(submitControl.indexOf('className="composer__stop-choice"'))
      .toBeLessThan(submitControl.indexOf('className="send-button composer__submit"'));
    expect(app).toContain("const showTurnActivity =\n    textTurnVisiblyActive || voicePondering;");
    expect(styles).toMatch(
      /\.message__streaming-text--reply-complete::after\s*\{[^}]*display: none;[^}]*animation: none;/
    );
  });

  it("shows the shared long-work queue without silently preempting chat", () => {
    expect(chatWorkspaceActivity).toContain('"training"');
    expect(chatWorkspaceActivity).toContain('"ingestion"');
    expect(chatWorkspaceActivity).toContain('"crawl"');
    expect(chatWorkspaceActivity).not.toContain("slice(-24)");
    expect(app).toContain("mergeChatVisibleLearningJob(current, job)");
    expect(app).toContain("learningJobs.map((job, order)");
    expect(app).toContain('stopDatasetActivity("pause", entry.job!.id)');
    expect(app).toContain('stopDatasetActivity("cancel", entry.job?.id)');
    expect(app).toContain("Current reply continues. New learning joins the shared queue.");
    expect(learning).toContain("const saved = await api.start({ ...common, inputs, requestId })");
    expect(learning).not.toContain("chat.cancel(");
    expect(learning).toContain('...manifests.map(manifest => ({ kind: "dataset" as const, manifestId: manifest.id }))');
    expect(app).toContain('className="workspace-view-host__waiting"');
    expect(app).not.toMatch(/onView\([^)]*\)[\s\S]{0,160}chat\.cancel/);
  });

  it("shows concrete web searches and human neural-memory labels", () => {
    expect(app).toContain("chatActionSearchActivity(event.action, event.state)");
    expect(app).toContain('"Temporary context"');
    expect(app).toContain('"Immediate neural changes"');
    expect(app).toContain('"Recorded learning counters"');
    expect(app).not.toContain('"Parameters only"');
    expect(app).not.toContain('"active patterns"');
    expect(app.toLocaleLowerCase()).not.toContain("connected memory");
    expect(app.toLocaleLowerCase()).not.toContain("connected patterns");
  });

  it("refreshes the authoritative worker PID without overlapping health reads", () => {
    expect(app).toContain("const WORKSPACE_HEALTH_REFRESH_MS = 5_000");
    expect(app).toContain("const result = await window.omni!.brain.health(brain.id)");
    expect(app).toContain("refreshTimer = window.setTimeout(");
    expect(app).toContain("if (refreshTimer !== undefined) window.clearTimeout(refreshTimer)");
    expect(app).toContain('setHealth("Python engine reconnecting…")');
  });

  it("keeps full source coverage and measured neural counters without restoring training-change clutter", () => {
    expect(learning).toContain("task.sources.map(source =>");
    expect(learning).toContain("source.coverage.processedRecords.toLocaleString()");
    expect(learning).toContain("source.coverage.rejectedRecords.toLocaleString()");
    expect(learning).toContain("source.coverage.processedBytes.toLocaleString()");
    expect(app).toContain("workspace.learning.fastNeuralMemory.connectionUpdatesTotal.toLocaleString()");
    expect(app).toContain("workspace.learning.fastNeuralMemory.parameterStepsTotal.toLocaleString()");
    expect(app).toContain('"Core weight verification"');
    expect(app).toContain('telemetry?.corticalWeightsText ?? "Not independently verified"');
    expect(app).not.toContain('className="source-neural-change"');
    expect(app).not.toContain("assembly deltas");
  });

  it("contains focus rings for clipped edge controls in every theme", () => {
    expect(styles).toContain("Chrome at a clipped viewport edge");
    expect(styles).toContain(".workspace-rail button:focus-visible");
    expect(styles).toContain(".workspace-header button:focus-visible");
    expect(styles).toContain("outline-offset: -3px");
  });
});
