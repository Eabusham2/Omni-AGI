import React, { useState } from "react";
import { createRoot } from "react-dom/client";
import { LearningWorkspace } from "../../src/renderer/src/LearningWorkspace";
import type { DatasetManifest, DatasetPreviewProgress } from "../../src/shared/types";
import type { FactCheckClaim, FactCheckWorkspace, LearningApi, LearningTask, LearningTaskEvent } from "../../src/shared/learning";
import type { UserQuestionEvent, UserQuestionRecord, UserQuestionsApi } from "../../src/shared/userQuestions";
import "../../src/renderer/src/styles.css";

const taskListeners = new Set<(event: LearningTaskEvent) => void>();
const previewListeners = new Set<(event: DatasetPreviewProgress) => void>();
const questionListeners = new Set<(event: UserQuestionEvent) => void>();
const tasks = new Map<string, LearningTask>();
const claims = new Map<string, FactCheckClaim[]>();
const receipts: unknown[] = [];
let sequence = 0;
let previewResolve: ((value: DatasetManifest | null) => void) | undefined;
let activePreview: string | undefined;
let notifyReceipts = () => {};
const record = (value: unknown): void => { receipts.push(value); notifyReceipts(); };
const emit = (task: LearningTask): LearningTask => { tasks.set(task.id, task); taskListeners.forEach(listener => listener({ type: "updated", task: structuredClone(task) })); return task; };
const update = (id: string, values: Partial<LearningTask>): LearningTask => emit({ ...tasks.get(id)!, ...values, revision: tasks.get(id)!.revision + 1, updatedAt: new Date().toISOString() });
function createTask(kind: LearningTask["kind"], request: { requestId?: string; instructions?: string; inputs?: unknown[]; constraints?: LearningTask["policy"]["constraints"] }): LearningTask {
  const now = new Date().toISOString(); const id = `fixture-task-${++sequence}`;
  return emit({ format: "omni-learning-task", version: 1, id, brainId: "fixture", requestId: request.requestId ?? id, requestSha256: "fixture-only",
    kind, state: "queued", createdAt: now, updatedAt: now, revision: 1, policy: { revision: 1, sha256: "fixture-only", constraints: request.constraints ?? {}, instructions: request.instructions ?? "", changedAt: now, changedBy: "user" },
    sources: (request.inputs ?? []).map((_, index) => ({ id: `source-${index}`, kind: "text", label: `Selected source ${index + 1}`, state: "pending" })),
    phase: "queued", label: kind === "fact-check" ? "Selected claim review" : "Selected source learning", epochs: 1, pendingQuestions: [], traversalComplete: false });
}
const question = (id: string, prompt: string): UserQuestionRecord => ({ format: "omni-user-question", version: 1, id, brainId: "fixture", requestId: id, requestSha256: "fixture-only",
  kind: "task-choice", prompt, mode: "text", options: [], minSelections: 0, maxSelections: 0, allowOther: false,
  createdAt: new Date().toISOString(), updatedAt: new Date().toISOString(), status: "pending", timeout: { enabled: false, seconds: 1200, policyRevision: "fixture" } });
const questionsById = new Map([question("selected-question", "Which publication should take priority?"), question("unrelated-question", "Unrelated question must not appear here.")].map(value => [value.id, value]));
const questions: UserQuestionsApi = {
  request: async () => { throw new Error("The fixture does not request new questions."); },
  list: async () => [...questionsById.values()].map(value => structuredClone(value)),
  respond: async request => {
    record({ operation: "question-answer", request });
    const prior = questionsById.get(request.questionId)!;
    const saved: UserQuestionRecord = { ...prior, status: "answered", resolution: { id: "fixture-answer", source: "user", resolvedAt: new Date().toISOString(), response: request.response, authority: "task-choice-only", permissionsGranted: false } };
    questionsById.set(saved.id, saved); questionListeners.forEach(listener => listener({ type: "resolved", brainId: "fixture", question: saved }));
    return saved;
  },
  cancel: async (_brain, id) => questionsById.get(id)!, resume: async () => [...questionsById.values()],
  preferences: async () => ({ timeoutEnabled: false, timeoutSeconds: 1200 }), setPreferences: async request => request,
  onEvent: listener => { questionListeners.add(listener); return () => { questionListeners.delete(listener); }; }
};
const api: LearningApi = {
  start: async request => { record({ operation: "learn", request }); return createTask("specific-learn", request); },
  startFactCheck: async request => {
    record({ operation: "fact-check", request }); const task = createTask("fact-check", request);
    const selected = request.claims.map((value, index): FactCheckClaim => ({ id: `claim-${index}`, text: value.text, sources: value.sources ?? [], state: "waiting-answer", questionId: index === 0 ? "selected-question" : undefined,
      evidence: [{ id: `evidence-${index}`, claimId: `claim-${index}`, url: "https://docs.example.com/source", fetchedAt: new Date().toISOString(), relation: "unreviewed", capture: "fetched-content", contentSha256: "fixture-only" }], findings: [] }));
    claims.set(task.id, selected); update(task.id, { state: "waiting", pendingQuestions: ["selected-question"], phase: "claim-review" });
    return { task: tasks.get(task.id)!, claims: selected, scope: "selected-claims-only", completeKnowledgeInventory: false };
  },
  list: async () => [...tasks.values()].map(value => structuredClone(value)), get: async (_brain, id) => structuredClone(tasks.get(id)!),
  factCheck: async (_brain, id) => ({ task: structuredClone(tasks.get(id)!), claims: structuredClone(claims.get(id) ?? []), scope: "selected-claims-only", completeKnowledgeInventory: false }),
  pause: async (_brain, id) => update(id, { state: "paused", phase: "safe-boundary" }),
  resume: async (_brain, id) => update(id, { state: "running", phase: "learning", currentSource: "source-0" }),
  stop: async (_brain, id) => update(id, { state: "stopped", phase: "safe-boundary" }),
  steer: async request => { record({ operation: "steer", request }); const task = tasks.get(request.taskId)!; return update(task.id, { policy: { ...task.policy, revision: task.policy.revision + 1, instructions: request.instructions ?? task.policy.instructions, constraints: request.constraints ?? task.policy.constraints } }); },
  review: async request => {
    record({ operation: "review", request }); const claim = claims.get(request.taskId)!.find(value => value.id === request.claimId)!;
    claim.findings.push({ id: request.requestId, claimId: claim.id, conclusion: request.conclusion, explanation: request.explanation, evidenceIds: request.evidenceIds, source: "user", createdAt: new Date().toISOString(), verifiedTruth: false,
      ...(request.proposedCorrection ? { proposedCorrection: request.proposedCorrection } : {}) }); claim.state = "reviewed";
    return api.factCheck("fixture", request.taskId);
  },
  acceptCorrection: async (_brain, id, claimId, findingId) => { record({ operation: "correction", id, claimId, findingId }); const child = createTask("specific-learn", { inputs: ["correction"] }); claims.get(id)!.find(value => value.id === claimId)!.correctionTaskId = child.id; return child; },
  onEvent: listener => { taskListeners.add(listener); return () => { taskListeners.delete(listener); }; }
};
const data = {
  preview: async (request: { requestId?: string }) => { activePreview = request.requestId; record({ operation: "preview", request });
    previewListeners.forEach(listener => listener({ schemaVersion: 1, requestId: request.requestId!, brainId: "fixture", phase: "hashing", discoveredFiles: 2001, discoveredBytes: 300_000_000,
      hashedFiles: 1900, hashedBytes: 250_000_000, startedAt: new Date().toISOString(), updatedAt: new Date().toISOString(), message: "Fixture metadata only" }));
    return new Promise<DatasetManifest | null>(resolve => { previewResolve = resolve; }); },
  cancelPreview: async (id: string) => { record({ operation: "stop-preview", id }); if (activePreview !== id) return false; previewResolve?.(null); previewResolve = undefined; activePreview = undefined; return true; },
  onPreviewProgress: (listener: (event: DatasetPreviewProgress) => void) => { previewListeners.add(listener); return () => { previewListeners.delete(listener); }; }
};
function Fixture() {
  const [mode, setMode] = useState<"specific-learn" | "fact-check">("specific-learn");
  const [, refresh] = useState(0); notifyReceipts = () => refresh(value => value + 1);
  return <main style={{ maxWidth: 1100, margin: "0 auto", padding: 16, height: "100vh", overflow: "hidden", boxSizing: "border-box", display: "grid", gridTemplateRows: "auto auto auto minmax(0,1fr) auto", gap: 8 }}>
    <h1>Learning controls fixture</h1><p>Synthetic transport only. No brain, training or web request occurs here.</p>
    <div style={{ display: "flex", flexWrap: "wrap", gap: 10 }}>
      <button className="button" onClick={() => setMode("specific-learn")}>Show Learn</button><button className="button" onClick={() => setMode("fact-check")}>Show Fact Check</button>
      <button className="button" onClick={() => { document.documentElement.dataset.colorScheme = document.documentElement.dataset.colorScheme === "dark" ? "light" : "dark"; }}>Switch appearance</button>
      <button className="button" onClick={() => { previewResolve?.({ schemaVersion: 1, id: "selected-manifest", brainId: "fixture", createdAt: new Date().toISOString(), updatedAt: new Date().toISOString(), roots: ["/fixture"], entryFile: "/fixture/entries", discoveredFiles: 2001, discoveredBytes: 300_000_000, manifestHash: "fixture-only" }); previewResolve = undefined; activePreview = undefined; }}>Finish fixture preparation</button>
      <button className="button" onClick={() => { const task = [...tasks.values()].findLast(value => value.kind === mode); if (task) update(task.id, { state: "running", progress: .99, currentSource: "source-0", phase: "learning" }); }}>Advance fixture source to 99%</button>
      <button className="button" onClick={() => { const task = [...tasks.values()].findLast(value => value.kind === mode); if (task) update(task.id, { state: "complete", traversalComplete: true, currentSource: undefined, progress: 1, phase: "finalized" }); }}>Finish fixture task</button>
      <a className="button" href="./mcp-auth-fixture.html">MCP controls fixture</a>
    </div>
    <LearningWorkspace brainId="fixture" mode={mode} api={api} questions={questions} data={data} />
    <details><summary>Fixture transport receipts</summary><pre style={{ whiteSpace: "pre-wrap", overflowWrap: "anywhere" }}>{JSON.stringify(receipts, null, 2)}</pre></details>
  </main>;
}
document.documentElement.dataset.colorScheme = "dark";
document.documentElement.dataset.layout = "glass";
createRoot(document.getElementById("root")!).render(<Fixture />);
