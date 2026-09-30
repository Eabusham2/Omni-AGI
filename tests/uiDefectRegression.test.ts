import { readFileSync } from "node:fs";
import { resolve } from "node:path";
import { describe, expect, it } from "vitest";

const root = resolve(import.meta.dirname, "..");
const app = readFileSync(resolve(root, "src/renderer/src/App.tsx"), "utf8");
const styles = readFileSync(resolve(root, "src/renderer/src/styles.css"), "utf8");
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
    expect(app).toContain("className=\"toast__message\"");
    expect(app).toContain("className=\"ui-error-copy\"");
    expect(app).toContain("conciseUiMessage(activeJob.error)");
    expect(styles).toContain("overflow-wrap: anywhere");
  });

  it("keeps Data and Brain Map navigable, and the map populated during live update lag", () => {
    expect(app).toContain('onBack={() => onView("chat")}');
    expect(app).toContain('aria-label="Back to conversation"');
    expect(app).toContain('<BrainMapWorkspace brain={brain} onBack={() => onView("chat")} />');
    expect(app).toContain('id: "substrate-overview"');
    expect(app).toContain('aria-label="Search neural assemblies"');
    expect(app).toContain('aria-busy={substrateLoading || undefined}');
  });

  it("uses durable topology totals without calling cumulative updates live connections", () => {
    expect(app).toContain("persistedConnections: brain.substrateTotals?.synapses");
    expect(app).toContain("persistedOverview.totals.synapses");
    expect(app).toContain("unique live connections");
    expect(app).toContain("cumulative connection updates");
    expect(app).toContain("activity total · not unique connections");
    expect(app).not.toContain("Graph indexing connection updates…");
  });

  it("exposes exact keyboard semantics and a visible token-used counter", () => {
    expect(app).toContain('aria-keyshortcuts="Enter Shift+Enter Control+Enter Meta+Enter"');
    expect(app).toContain('className="composer__token-counter"');
    expect(app).toContain('aria-label={`Token counter and neural learning status. ${composerTelemetryAriaLabel}`}');
    expect(app).toContain('{draftTokenCount.toLocaleString()} draft · {contextTokenCopy} · {composerTelemetryCopy}');
    expect(app).toContain("utf8DraftTokenCount(input)");
    expect(app).toContain("Enter queues · Ctrl/Cmd Enter steers");
    expect(app).toContain("Enter to send · Shift Enter for a line break");
    expect(app).toContain("pendingChatOutputPresentation({");
    expect(app).toContain("responseOutputPresentation.label");
    expect(app).toContain("response-token-counter--${responseOutputPresentation.phase}");
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
    expect(app).toContain('className="composer__steer-choice"');
    expect(app).toContain("onClick={() => void steerCurrentTurn()}");
    expect(app).toContain(
      'title="Apply this direction at the next safe neural boundary; the prior request remains visible and the loaded mind stays warm"'
    );
    expect(app).toContain('className="send-button composer__queue-choice"');
    expect(app).toContain("onClick={submitOrdinaryTurn}");
    expect(app).toContain('className="chat-tool-status" role="status"');
    expect(app).toContain("Cancel the exact job in its action details.");
    expect(app).toContain('className="composer__stop-choice"');
    const queueControl = app.indexOf(
      'className="send-button composer__queue-choice"'
    );
    const stopControl = app.indexOf(
      'className="composer__stop-choice"',
      queueControl
    );
    expect(stopControl).toBeGreaterThan(queueControl);
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
    expect(app).toContain('dataActionAvailableDuringTurn("immediate-write", foregroundTurnActive)');
    expect(app).toContain("Uploads, training, and crawls started here join the visible queue");
    expect(app).toContain('className="workspace-view-host__waiting"');
    expect(app).not.toMatch(/onView\([^)]*\)[\s\S]{0,160}chat\.cancel/);
  });

  it("shows concrete web searches and human neural-memory labels", () => {
    expect(app).toContain("chatActionSearchActivity(event.action, event.state)");
    expect(app).toContain('"Neural memory + source" : "Neural memory"');
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

  it("shows complete source-ledger neural changes instead of one assembly", () => {
    expect(app).toContain('className="source-neural-change"');
    expect(app).toContain("compactParameterCount(source.learnedConcepts)");
    expect(app).toContain("compactParameterCount(source.learnedSynapses)");
    expect(app).toContain("compactParameterCount(source.learnedParameterSteps)");
    expect(app).toContain('source.learnedParameterSteps === undefined');
    expect(app).toContain('"Optimizer steps unknown"');
    expect(app).toContain('"composite neural checksum changed"');
    expect(app).toContain('dense model weight change is not separately verified.');
    expect(app).not.toContain("assembly deltas");
    expect(styles).toContain(".source-row > span.source-neural-change");
  });

  it("contains focus rings for clipped edge controls in every theme", () => {
    expect(styles).toContain("Chrome at a clipped viewport edge");
    expect(styles).toContain(".workspace-rail button:focus-visible");
    expect(styles).toContain(".workspace-header button:focus-visible");
    expect(styles).toContain("outline-offset: -3px");
  });
});
