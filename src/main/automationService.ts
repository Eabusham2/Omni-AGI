import type { AutomationActivation, AutomationDefinition, AutomationJson, AutomationPageQuery, AutomationProgram, AutomationRun, AutomationStepReceipt, AutomationWorkspaceEvent, AutomationWorkspaceSnapshot } from "../shared/automation";
import type { ScheduledOccurrence } from "../shared/clock";
import type { ToolExecutionResult, ToolInvocation } from "../shared/types";
import type { ToolExecutor } from "./toolExecutor";
import { realHostClock, type HostClockSource } from "./hostClock";
import { automationCanonical, automationHash, automationId, automationObject, automationSha, evaluateAutomationCondition, resolveAutomationArguments } from "./automationFormat";
import { AutomationStore } from "./automationStore";
import { lstat } from "node:fs/promises";
import { join } from "node:path";

export interface AutomationServiceOptions {
  brainDirectory(brainId: string): Promise<string>;
  tools: Pick<ToolExecutor, "execute" | "cancel" | "cancelAndWait" | "invalidateApproval">;
  authority(brainId: string): Promise<{ enabled: boolean; revision: string }>;
  ownerEpoch(brainId: string): Promise<string>;
  rotateOwnerEpoch(brainId: string): Promise<string>;
  admissionOpen(): boolean;
  source?: HostClockSource;
  checkResources?(brainId: string, bytes: number, purpose: string): Promise<void>;
  reserveWrite?(root: string, bytes: number): Promise<void>;
  observeOutcome?(invocation: ToolInvocation, result: ToolExecutionResult): Promise<void>;
  onError?(error: Error, brainId: string): void;
}
const activeStates = ["queued", "running", "waiting", "approval-required", "stopping"];
const terminalStates = new Set(["complete", "stopped"]);
const cloneJson = <T>(value: T): T => JSON.parse(JSON.stringify(value)) as T;

/** Host action interpreter. It never decodes a token, loads a brain or grants tool access. */
export class AutomationService {
  private readonly source: HostClockSource;
  private readonly pumping = new Map<string, Promise<void>>();
  private readonly activeRequests = new Map<string, { brainId: string; definitionId: string; requestId: string }>();
  private readonly timers = new Map<string, NodeJS.Timeout>();
  private readonly listeners = new Set<(event: AutomationWorkspaceEvent) => void>();
  private readonly owners = new Set<string>();
  private readonly frozen = new Set<string>();
  private quitting = false;
  constructor(private readonly options: AutomationServiceOptions) { this.source = options.source ?? realHostClock; }
  private iso(): string { return new Date(this.source.wallMs()).toISOString(); }
  private key(brainId: string, runId: string): string { return JSON.stringify([brainId, runId]); }
  private async store(brainId: string): Promise<AutomationStore> {
    const root = await this.options.brainDirectory(automationId(brainId));
    await this.options.reserveWrite?.(root, 65536); this.owners.add(brainId);
    return AutomationStore.open(root, brainId);
  }
  subscribe(listener: (event: AutomationWorkspaceEvent) => void): () => void { this.listeners.add(listener); return () => { this.listeners.delete(listener); }; }
  private emit(brainId: string): void { const value: AutomationWorkspaceEvent = { brainId, type: "changed", at: this.iso() }; for (const listener of this.listeners) listener(value); }
  private resources(brainId: string, value: unknown, purpose: string): Promise<void> {
    return this.options.checkResources?.(brainId, Buffer.byteLength(automationCanonical(cloneJson(value))) * 12 + 65536, purpose) ?? Promise.resolve();
  }
  async definitions(brainId: string, query?: AutomationPageQuery) { const store = await this.store(brainId); try { return store.page<AutomationDefinition>("definition", query); } finally { store.close(); } }
  async runs(brainId: string, query?: AutomationPageQuery) { const store = await this.store(brainId); try { return store.page<AutomationRun>("run", query); } finally { store.close(); } }
  async steps(brainId: string, runId: string, query?: AutomationPageQuery) { const store = await this.store(brainId); try { return store.stepPage(runId, query); } finally { store.close(); } }
  async currentStep(brainId: string, runId: string): Promise<AutomationStepReceipt | null> {
    const store = await this.store(brainId); try { const run = store.get<AutomationRun>("run", automationId(runId)); if (!run) throw new Error("Action run not found.");
      return store.latestStep(run.id, run.nextStep) ?? null; } finally { store.close(); }
  }
  async program(brainId: string, definitionId: string, revisionSha256?: string): Promise<{ definition: AutomationDefinition; program: AutomationProgram }> {
    const store = await this.store(brainId); try {
      const definition = store.get<AutomationDefinition>("definition", automationId(definitionId)); if (!definition) throw new Error("Action file not found.");
      const program = await store.program(revisionSha256 ?? definition.revisionSha256, bytes => this.options.checkResources?.(brainId, bytes, "read action file") ?? Promise.resolve());
      return { definition, program };
    } finally { store.close(); }
  }
  async snapshot(brainId: string): Promise<AutomationWorkspaceSnapshot & { catalogRevision: number }> {
    const store = await this.store(brainId); try {
      const count = (kind: string, states?: string[]) => Number(store.db.prepare(`SELECT COUNT(*) AS n FROM records WHERE kind=?${states?.length ? ` AND state IN (${states.map(() => "?").join(",")})` : ""}`).get(kind, ...(states ?? []))?.n);
      return { brainId, catalogRevision: store.catalogRevision(), hostExecution: true, processResidentRequired: true,
        counts: { definitions: count("definition"), activeDefinitions: count("definition", ["active"]), runs: count("run"), activeRuns: count("run", activeStates), reviewRequired: count("run", ["review-required", "failed"]), approvals: count("run", ["approval-required"]) } };
    } finally { store.close(); }
  }
  private async grant(brainId: string, definition: AutomationDefinition, allowTriggers: boolean): Promise<AutomationActivation> {
    const authority = await this.options.authority(brainId);
    if (!authority.enabled) throw new Error("Independent automations are disabled for this instance.");
    return { enabled: true, allowTriggers, ownerBrainId: brainId, ownerEpoch: await this.options.ownerEpoch(brainId),
      catalogRevision: definition.revision, revisionSha256: definition.revisionSha256, policyRevision: authority.revision, changedAt: this.iso() };
  }
  private async validRun(brainId: string, run: AutomationRun, definition: AutomationDefinition): Promise<boolean> {
    const authority = await this.options.authority(brainId), epoch = await this.options.ownerEpoch(brainId);
    return authority.enabled && run.activation.enabled && run.activation.ownerBrainId === brainId && run.activation.ownerEpoch === epoch &&
      run.activation.policyRevision === authority.revision && run.activation.catalogRevision === definition.revision && run.revisionSha256 === definition.revisionSha256;
  }
  private requireMutationAdmission(brainId: string): void {
    if (this.quitting || this.frozen.has(brainId) || !this.options.admissionOpen()) throw new Error("Automation changes wait for the application's safe checkpoint boundary.");
  }
  /** Called only through ToolExecutor's current policy/approval/intent boundary. */
  async execute(brainId: string, action: string, argsValue: unknown, requestId: string, signal?: AbortSignal): Promise<unknown> {
    const args = automationObject(argsValue); automationId(requestId); signal?.throwIfAborted();
    const fields: Record<string, readonly string[]> = { list: ["after", "limit"], runs: ["after", "limit"], status: [], export: ["definitionId", "revisionSha256"], steps: ["runId", "after", "limit"], "current-step": ["runId"],
      save: ["program", "definitionId", "expectedRevision"], activate: ["definitionId", "expectedRevision", "revisionSha256", "allowTriggers"], disable: ["definitionId", "expectedRevision", "revisionSha256"],
      run: ["definitionId", "revisionSha256", "inputs"], pause: ["runId"], resume: ["runId"], stop: ["runId"], approve: ["runId", "approvalToken"], review: ["runId", "resolution", "acknowledgePossibleEffects", "reason"] };
    const mutates = ["save", "activate", "disable", "run", "pause", "resume", "stop", "approve", "review"].includes(action);
    if (!fields[action] || Object.keys(args).some(key => !fields[action]!.includes(key) && !(mutates && key === "requestKey"))) throw new Error("Action-file operation contains unsupported fields.");
    const commandId = args.requestKey === undefined ? requestId : automationId(args.requestKey);
    if (args.allowTriggers !== undefined && typeof args.allowTriggers !== "boolean") throw new Error("Trigger activation must be an explicit boolean.");
    if (action === "list") return this.definitions(brainId, args as AutomationPageQuery);
    if (action === "runs") return this.runs(brainId, args as AutomationPageQuery);
    if (action === "status") return this.snapshot(brainId);
    if (action === "export") return this.program(brainId, automationId(args.definitionId), args.revisionSha256 === undefined ? undefined : automationSha(args.revisionSha256));
    if (action === "steps") return this.steps(brainId, automationId(args.runId), args as AutomationPageQuery);
    if (action === "current-step") { const receipt = await this.currentStep(brainId, automationId(args.runId)); return { runId: args.runId, ...(receipt ? { receipt } : {}) }; }
    if (!(await this.options.authority(brainId)).enabled) throw new Error("Independent automations are disabled for this instance.");
    this.requireMutationAdmission(brainId);
    if (action === "save") return this.save(brainId, args, commandId, signal);
    if (action === "activate" || action === "disable") return this.activate(brainId, args, action === "activate", commandId, signal);
    if (action === "run") return this.createRun(brainId, args, commandId, undefined, signal);
    if (action === "pause" || action === "resume" || action === "stop") return this.control(brainId, automationId(args.runId), action, signal, requestId);
    if (action === "approve") return this.approve(brainId, automationId(args.runId), String(args.approvalToken ?? ""), signal);
    if (action === "review") return this.review(brainId, args, commandId, signal);
    throw new Error("Unknown action-file operation.");
  }
  private async save(brainId: string, args: Record<string, unknown>, requestId: string, signal?: AbortSignal): Promise<AutomationDefinition> {
    await this.resources(brainId, args.program, "save action file"); const store = await this.store(brainId);
    try {
      const requestHash = automationHash(automationCanonical(args)), retained = store.commandResult<AutomationDefinition>(requestId, requestHash);
      if (retained) return retained;
      const body = await store.saveProgram(args.program, bytes => this.options.reserveWrite?.(store.root, bytes) ?? Promise.resolve()); signal?.throwIfAborted();
      const id = args.definitionId === undefined ? AutomationStore.id() : automationId(args.definitionId), previous = store.get<AutomationDefinition>("definition", id);
      if (previous ? args.expectedRevision !== previous.revision : args.expectedRevision !== undefined) throw new Error("Action file changed; review the current revision.");
      const result = store.command(requestId, requestHash, () => {
        this.requireMutationAdmission(brainId);
        const current = store.get<AutomationDefinition>("definition", id);
        if (current?.revision !== previous?.revision) throw new Error("Action file changed while its new bytes were being saved.");
        const revision = store.nextCatalogRevision(store.catalogRevision()), now = this.iso();
        const row: AutomationDefinition = { id, brainId, revision, revisionSha256: body.sha256, name: body.program.name,
          createdAt: previous?.createdAt ?? now, updatedAt: now };
        store.put(row, "definition"); return row;
      }); this.emit(brainId); return result;
    } finally { store.close(); }
  }
  private async activate(brainId: string, args: Record<string, unknown>, enabled: boolean, requestId: string, signal?: AbortSignal): Promise<AutomationDefinition> {
    const store = await this.store(brainId); try {
      const requestHash = automationHash(automationCanonical({ action: enabled ? "activate" : "disable", ...args })), retained = store.commandResult<AutomationDefinition>(requestId, requestHash);
      if (retained) return retained;
      const old = store.get<AutomationDefinition>("definition", automationId(args.definitionId));
      if (!old || args.expectedRevision !== old.revision || args.revisionSha256 !== old.revisionSha256) throw new Error("Review the exact current action-file revision before activation.");
      const revision = store.catalogRevision() + 1, next = { ...old, revision, updatedAt: this.iso() };
      const grant = await this.grant(brainId, next, args.allowTriggers === true); signal?.throwIfAborted();
      const result = store.command(requestId, requestHash, () => {
        this.requireMutationAdmission(brainId);
        if (store.get<AutomationDefinition>("definition", old.id)?.revision !== old.revision) throw new Error("Action file changed during activation review.");
        store.nextCatalogRevision(revision - 1); const row = { ...next, activation: { ...grant, enabled, allowTriggers: enabled && grant.allowTriggers } };
        store.put(row, "definition"); return row;
      }); this.emit(brainId); if (!enabled) await this.revokeRuns(brainId, old.id, "Action file disabled."); return result;
    } finally { store.close(); }
  }
  private async createRun(brainId: string, args: Record<string, unknown>, requestId: string, occurrence?: ScheduledOccurrence, signal?: AbortSignal): Promise<AutomationRun> {
    const store = await this.store(brainId); try {
      const inputs = cloneJson(automationObject(args.inputs ?? {})) as Record<string, AutomationJson>;
      const requestHash = automationHash(automationCanonical({ definitionId: automationId(args.definitionId), revisionSha256: automationSha(args.revisionSha256), inputs,
        ...(occurrence ? { occurrenceId: occurrence.id, eventId: occurrence.eventId ?? null, eventData: occurrence.eventData ?? null } : {}) }));
      const retained = store.commandResult<AutomationRun>(requestId, requestHash);
      if (retained) { const actual = store.get<AutomationRun>("run", retained.id); if (!actual) throw new Error("A prior automation handoff lost its durable run; it will not be recreated."); return actual; }
      const definition = store.get<AutomationDefinition>("definition", automationId(args.definitionId));
      if (!definition || definition.revisionSha256 !== args.revisionSha256) throw new Error("The requested immutable action-file revision is not current.");
      await this.resources(brainId, inputs, "automation input");
      const grant = occurrence ? definition.activation : await this.grant(brainId, definition, false);
      if (!grant || occurrence && (!grant.enabled || !grant.allowTriggers)) throw new Error("Enable this exact action file for timer/event triggers before scheduling it.");
      const check: AutomationRun = { id: "new-run", brainId, definitionId: definition.id, revisionSha256: definition.revisionSha256,
        inputs, inputSha256: automationHash(automationCanonical(inputs)), activation: grant, state: "queued", nextStep: 0, createdAt: this.iso(), updatedAt: this.iso() };
      if (!await this.validRun(brainId, check, definition)) throw new Error("Action activation/policy changed; explicit current review is required."); signal?.throwIfAborted();
      const id = occurrence ? automationHash(`automation-occurrence-1\0${brainId}\0${occurrence.id}`) : AutomationStore.id();
      const proposed = store.command(requestId, requestHash, () => {
        this.requireMutationAdmission(brainId);
        const current = store.get<AutomationDefinition>("definition", definition.id);
        if (current?.revision !== definition.revision || current.revisionSha256 !== definition.revisionSha256) throw new Error("The action file changed before the durable run handoff.");
        const row: AutomationRun = { ...check, id, ...(occurrence ? { trigger: { occurrenceId: occurrence.id, ...(occurrence.eventId ? { eventId: occurrence.eventId } : {}),
          ...(occurrence.eventData !== undefined ? { eventData: cloneJson(occurrence.eventData) as AutomationJson } : {}) } } : {}) };
        store.put(row, "run"); store.event(id, "created", row); return row;
      });
      const result = store.get<AutomationRun>("run", proposed.id)!; this.emit(brainId); this.kick(brainId, result.id); return result;
    } finally { store.close(); }
  }
  /** Durable, idempotent Clock handoff. It never Load's or submits a chat message. */
  async acceptOccurrence(occurrence: ScheduledOccurrence): Promise<{ runId: string; accepted: true }> {
    const ref = automationObject((occurrence as unknown as { automation: unknown }).automation);
    if (this.quitting || this.frozen.has(occurrence.brainId) || !this.options.admissionOpen()) throw new Error("Host execution is at its shutdown/recovery boundary.");
    const run = await this.createRun(occurrence.brainId, ref, occurrence.id, occurrence);
    return { runId: run.id, accepted: true };
  }
  kick(brainId: string, runId: string): void {
    const key = this.key(brainId, runId); if (this.pumping.has(key) || this.quitting || this.frozen.has(brainId)) return;
    const timer = this.timers.get(key); if (timer) clearTimeout(timer); this.timers.delete(key);
    const pending = Promise.resolve().then(() => this.pump(brainId, runId)).catch(async error => {
      await this.changeRun(brainId, runId, { state: "review-required", detail: `Action admission/receipt requires attention: ${error instanceof Error ? error.message : String(error)}` }, "admission-failed").catch(() => undefined);
      this.options.onError?.(error instanceof Error ? error : new Error(String(error)), brainId);
    })
      .finally(() => { if (this.pumping.get(key) === pending) this.pumping.delete(key); }); this.pumping.set(key, pending);
  }
  async settle(): Promise<void> { await Promise.all([...this.pumping.values()]); }
  private async changeRun(brainId: string, runId: string, update: Partial<AutomationRun>, event: string): Promise<AutomationRun> {
    const store = await this.store(brainId); try { const row = store.transaction(() => {
      const old = store.get<AutomationRun>("run", runId); if (!old) throw new Error("Action run not found.");
      if (terminalStates.has(old.state)) return old;
      const next = { ...old, ...update, updatedAt: this.iso() }; store.put(next, "run"); store.event(runId, event, { at: this.iso(), ...update }); return next;
    }); this.emit(brainId); return row; } finally { store.close(); }
  }
  private async retainOutcome(brainId: string, runId: string, step: AutomationStepReceipt, result: ToolExecutionResult): Promise<void> {
    await this.resources(brainId, result, "retain actual automation outcome"); const store = await this.store(brainId);
    try { store.transaction(() => {
      const current = store.latestStep(runId, step.index);
      if (!current || current.requestId !== step.requestId) throw new Error("Actual action outcome lost its owned step intent.");
      if (["complete", "failed"].includes(current.state)) return;
      const next: AutomationStepReceipt = { ...current, state: result.state === "approval-required" ? "approval-required" : result.state, outcome: cloneJson(result), updatedAt: this.iso() };
      store.putStep(next); store.event(runId, "actual-outcome", next);
      const run = store.get<AutomationRun>("run", runId)!;
      const stopped = run.state === "stopping" || run.state === "stopped", paused = run.state === "paused" || run.state === "review-required";
      store.put({ ...run, ...(result.state === "complete" ? { nextStep: Math.max(run.nextStep, step.index + 1) } : {}),
        state: stopped ? "stopped" : paused ? run.state : result.state === "complete" ? "running" : result.state === "approval-required" ? "approval-required" : "failed", updatedAt: this.iso(),
        ...(result.error ? { detail: result.error } : {}) }, "run");
    }); this.emit(brainId); } finally { store.close(); }
  }
  private async pump(brainId: string, runId: string, approvalToken?: string): Promise<void> {
    while (!this.quitting && !this.frozen.has(brainId) && this.options.admissionOpen()) {
      const store = await this.store(brainId); let run: AutomationRun, program: AutomationProgram, receipt: AutomationStepReceipt;
      try {
        const value = store.get<AutomationRun>("run", runId); if (!value || !["queued", "running", "waiting", ...(approvalToken ? ["approval-required"] : [])].includes(value.state)) return; run = value;
        const definition = store.get<AutomationDefinition>("definition", run.definitionId);
        if (!definition || !await this.validRun(brainId, run, definition)) { await this.changeRun(brainId, runId, { state: "review-required", detail: "Current file, activation or access changed. No future step is automatically authorized." }, "authority-changed"); return; }
        program = await store.program(run.revisionSha256, bytes => this.options.checkResources?.(brainId, bytes, "read action revision") ?? Promise.resolve());
        if (run.nextStep >= program.steps.length) { await this.changeRun(brainId, runId, { state: "complete", detail: "Every requested step has an actual receipt or explicit skip." }, "complete"); return; }
        const step = program.steps[run.nextStep]!, old = store.latestStep(runId, run.nextStep);
        if (old && old.stepId !== step.id) throw new Error("The retained step identity differs from its immutable program.");
        if (old && ["complete", "skipped", "reviewed-skip"].includes(old.state)) { await this.changeRun(brainId, runId, { nextStep: run.nextStep + 1, state: "running" }, "advance-retained"); continue; }
        if (old && ["intent", "unknown", "failed"].includes(old.state) && old.review?.resolution !== "retry-confirmed") { await this.changeRun(brainId, runId, { state: "review-required", detail: "A prior attempt may have performed an effect. Inspect its receipt; it will not be replayed." }, "needs-review"); return; }
        const outcomes = new Map<string, AutomationJson>();
        for (const item of store.db.prepare("SELECT payload,sha256 FROM steps WHERE run_id=? AND state='complete' ORDER BY step_index,attempt").iterate(runId)) {
          if (automationHash(String(item.payload)) !== item.sha256) throw new Error("Prior action outcome checksum failed.");
          const prior = JSON.parse(String(item.payload)) as AutomationStepReceipt;
          if (prior.outcome && Object.hasOwn(prior.outcome, "output")) outcomes.set(prior.stepId, cloneJson(prior.outcome.output) as AutomationJson);
        }
        const attempt = old?.review?.resolution === "retry-confirmed" ? old.attempt : old?.attempt ?? 1;
        receipt = old ?? { runId, stepId: step.id, index: run.nextStep, attempt,
          requestId: automationHash(`automation-step-1\0${brainId}\0${runId}\0${run.nextStep}\0${attempt}`), state: "intent", startedAt: this.iso(), updatedAt: this.iso() };
        if (!evaluateAutomationCondition(step.when, run, outcomes)) {
          const current = store.get<AutomationRun>("run", runId)!; if (!validExecutionState(current.state, false)) return;
          store.transaction(() => { store.putStep({ ...receipt, state: "skipped", updatedAt: this.iso() }); store.event(runId, "condition-skipped", { stepId: step.id, index: run.nextStep });
            store.put({ ...current, state: "running", nextStep: run.nextStep + 1, updatedAt: this.iso() }, "run"); }); this.emit(brainId); continue;
        }
        if (step.kind === "wait") {
          const current = store.get<AutomationRun>("run", runId)!; if (!validExecutionState(current.state, false)) return;
          const remaining = receipt.wait ? receipt.wait.sessionId === this.source.sessionId ? Math.max(0, receipt.wait.monotonicDeadline - this.source.monotonicMs()) : Math.max(0, Date.parse(receipt.wait.dueAt) - this.source.wallMs()) : step.delayMs;
          if (receipt.wait && remaining <= 0) { store.transaction(() => { store.putStep({ ...receipt, state: "complete", updatedAt: this.iso() }); store.put({ ...current, nextStep: run.nextStep + 1, state: "running", updatedAt: this.iso() }, "run"); }); continue; }
          receipt = { ...receipt, state: "waiting", wait: { remainingMs: remaining, dueAt: new Date(this.source.wallMs() + remaining).toISOString(), sessionId: this.source.sessionId, monotonicDeadline: this.source.monotonicMs() + remaining }, updatedAt: this.iso() };
          store.transaction(() => { store.putStep(receipt); store.put({ ...current, state: "waiting", updatedAt: this.iso() }, "run"); });
          const key = this.key(brainId, runId), timer = setTimeout(() => { this.timers.delete(key); this.kick(brainId, runId); }, Math.min(remaining || 1, 2_147_483_647)); timer.unref(); this.timers.set(key, timer); this.emit(brainId); return;
        }
        const invocation: ToolInvocation = old?.invocation ? cloneJson(old.invocation) : { brainId, toolId: step.toolId, action: step.action, arguments: resolveAutomationArguments(step, run, outcomes) };
        await this.resources(brainId, invocation, "automation step");
        if (approvalToken) invocation.approvalToken = approvalToken;
        const durableInvocation = { ...invocation }; delete durableInvocation.approvalToken;
        receipt = { ...receipt, state: "intent", invocation: durableInvocation, invocationSha256: automationHash(automationCanonical(durableInvocation)), updatedAt: this.iso() };
        const current = store.get<AutomationRun>("run", runId)!;
        if (!validExecutionState(current.state, !!approvalToken) || this.quitting || this.frozen.has(brainId)) return;
        store.transaction(() => { store.putStep(receipt); store.event(runId, "pre-effect-intent", receipt); store.put({ ...current, state: "running", updatedAt: this.iso() }, "run"); });
      } finally { store.close(); }
      const invocation = receipt!.invocation!;
      const activeKey = this.key(brainId, runId), active = { brainId, definitionId: run!.definitionId, requestId: receipt!.requestId };
      this.activeRequests.set(activeKey, active);
      try {
        const result = await this.options.tools.execute({ ...invocation, ...(approvalToken ? { approvalToken } : {}) }, undefined, receipt!.requestId,
          async actual => { await this.retainOutcome(brainId, runId, receipt!, actual); await this.options.observeOutcome?.(invocation, actual); }, { actionEventId: receipt!.requestId });
        await this.retainOutcome(brainId, runId, receipt!, result); approvalToken = undefined;
        if (result.state !== "complete") return;
      } catch (error) {
        await this.changeRun(brainId, runId, { state: "review-required", detail: `Actual dispatch ownership needs review: ${error instanceof Error ? error.message : String(error)}` }, "unconfirmed-outcome"); return;
      } finally { if (this.activeRequests.get(activeKey) === active) this.activeRequests.delete(activeKey); }
      await new Promise<void>(resolve => setImmediate(resolve));
    }
  }
  private async control(brainId: string, runId: string, action: "pause" | "resume" | "stop", signal?: AbortSignal, callerRequestId?: string): Promise<AutomationRun> {
    const store = await this.store(brainId); let run: AutomationRun, step: AutomationStepReceipt | undefined;
    try { signal?.throwIfAborted(); const old = store.get<AutomationRun>("run", runId); if (!old) throw new Error("Action run not found."); run = old; if (terminalStates.has(run.state)) return run;
      step = store.latestStep(runId, run.nextStep);
      if (action === "resume" && ["review-required", "failed"].includes(run.state)) throw new Error("Review the uncertain/failed attempt explicitly; Resume cannot replay it.");
      if (action === "resume" && run.state === "approval-required") throw new Error("Approve or stop the exact pending step; Resume cannot grant its access.");
      if (action === "pause" && step?.wait) { const remaining = step.wait.sessionId === this.source.sessionId ? Math.max(0, step.wait.monotonicDeadline - this.source.monotonicMs()) : Math.max(0, Date.parse(step.wait.dueAt) - this.source.wallMs());
        store.putStep({ ...step, wait: { ...step.wait, remainingMs: remaining }, updatedAt: this.iso() }); }
      if (action === "resume" && step?.wait) store.putStep({ ...step, wait: { remainingMs: step.wait.remainingMs, dueAt: new Date(this.source.wallMs() + step.wait.remainingMs).toISOString(), sessionId: this.source.sessionId,
        monotonicDeadline: this.source.monotonicMs() + step.wait.remainingMs }, updatedAt: this.iso() });
      // Publish the pause/stop request before yielding back to a fast current
      // effect; its terminal callback must not admit another step first.
      run = store.transaction(() => {
        const current = store.get<AutomationRun>("run", runId)!;
        const next: AutomationRun = { ...current, state: action === "stop" ? "stopping" : action === "pause" ? "paused" : "queued", updatedAt: this.iso(),
          ...(action === "pause" ? { pausedFor: "user" } : action === "resume" ? { pausedFor: undefined } : { detail: "Stop requested; committed external effects are retained, not undone." }) };
        store.put(next, "run"); store.event(runId, action === "stop" ? "stop-requested" : action, { state: next.state, at: next.updatedAt }); return next;
      });
    } finally { store.close(); }
    const timer = this.timers.get(this.key(brainId, runId)); if (timer) clearTimeout(timer); this.timers.delete(this.key(brainId, runId));
    this.emit(brainId);
    if (action === "stop") {
      if (step?.outcome?.approvalToken) this.options.tools.invalidateApproval(brainId, step.outcome.approvalToken);
      if (step && callerRequestId === step.requestId) this.options.tools.cancel(brainId, step.requestId); // Never await this control call's own settlement.
      else if (step) await this.options.tools.cancelAndWait(brainId, step.requestId);
      return this.changeRun(brainId, runId, { state: "stopped", detail: "Stopped. Actual receipts remain saved; unconfirmed effects are not retried." }, "stopped");
    }
    if (action === "resume") this.kick(brainId, runId); return run;
  }
  private async approve(brainId: string, runId: string, token: string, signal?: AbortSignal): Promise<AutomationRun> {
    const store = await this.store(brainId); try {
      const run = store.get<AutomationRun>("run", runId), step = run && store.latestStep(runId, run.nextStep);
      if (!run || run.state !== "approval-required" || !step?.outcome?.approvalToken || step.outcome.approvalToken !== token) throw new Error("The exact pending action approval is unavailable.");
      signal?.throwIfAborted(); const key = this.key(brainId, runId);
      if (this.pumping.has(key)) throw new Error("This action already has an execution owner.");
      const pending = this.pump(brainId, runId, token).finally(() => this.pumping.delete(key)); this.pumping.set(key, pending);
      void pending.catch(error => this.options.onError?.(error, brainId)); return run;
    } finally { store.close(); }
  }
  private async review(brainId: string, args: Record<string, unknown>, requestId: string, signal?: AbortSignal): Promise<AutomationRun> {
    if (!["skip", "retry-confirmed"].includes(String(args.resolution)) || args.acknowledgePossibleEffects !== true || typeof args.reason !== "string" || !args.reason.trim()) throw new Error("Explicitly acknowledge possible prior effects and record why this attempt can be skipped or tried as a new attempt.");
    const store = await this.store(brainId); try {
      const requestHash = automationHash(automationCanonical(args)), retained = store.commandResult<AutomationRun>(requestId, requestHash);
      if (retained) return store.get<AutomationRun>("run", retained.id) ?? retained;
      const run = store.get<AutomationRun>("run", automationId(args.runId)); if (!run || !["review-required", "failed"].includes(run.state)) throw new Error("This run has no uncertain attempt to review.");
      if (this.pumping.has(this.key(brainId, run.id))) throw new Error("The original dispatch is still finishing its actual receipt; wait for that boundary before reviewing a new attempt.");
      const definition = store.get<AutomationDefinition>("definition", run.definitionId); if (!definition || definition.revisionSha256 !== run.revisionSha256) throw new Error("The exact original program must be selected before reviewing this run.");
      const program = await store.program(run.revisionSha256); const stepId = program.steps[run.nextStep]?.id;
      if (!stepId) throw new Error("No remaining original step is available for a reviewed attempt.");
      const grant = await this.grant(brainId, definition, false); signal?.throwIfAborted();
      const result = store.command(requestId, requestHash, () => {
        this.requireMutationAdmission(brainId);
        const current = store.get<AutomationRun>("run", run.id);
        if (!current || current.state !== run.state || current.nextStep !== run.nextStep || current.updatedAt !== run.updatedAt) throw new Error("The run changed during possible-effects review.");
        const old = store.latestStep(run.id, run.nextStep), attempt = (old?.attempt ?? 0) + 1;
        if (old?.outcome?.approvalToken) this.options.tools.invalidateApproval(brainId, old.outcome.approvalToken);
        const step: AutomationStepReceipt = { runId: run.id, stepId: old?.stepId ?? stepId, index: run.nextStep, attempt,
          requestId: automationHash(`automation-step-1\0${brainId}\0${run.id}\0${run.nextStep}\0${attempt}`), state: args.resolution === "skip" ? "reviewed-skip" : "unknown",
          startedAt: this.iso(), updatedAt: this.iso(), review: { at: this.iso(), reason: String(args.reason), resolution: args.resolution as "skip" | "retry-confirmed", acknowledgedPossibleEffects: true } };
        store.putStep(step); store.event(run.id, "operator-review", step);
        const row: AutomationRun = { ...run, activation: grant, state: "paused", updatedAt: this.iso(), ...(args.resolution === "skip" ? { nextStep: run.nextStep + 1 } : {}), detail: "Reviewed. Resume starts only the explicitly selected remaining/new attempt." };
        store.put(row, "run"); return row;
      }); this.emit(brainId); return result;
    } finally { store.close(); }
  }
  async recover(brainIds: readonly string[]): Promise<void> {
    for (const brainId of brainIds) {
      const root = await this.options.brainDirectory(automationId(brainId));
      if (!await lstat(join(root, "automation", "workspace.sqlite3")).catch((error: NodeJS.ErrnoException) => { if (error.code === "ENOENT") return undefined; throw error; })) continue;
      const store = await this.store(brainId); try { store.transaction(() => {
        for (const run of store.records<AutomationRun>("run", ["queued", "running", "waiting", "stopping"])) {
          const step = store.latestStep(run.id, run.nextStep), uncertain = step?.state === "intent" || step?.state === "unknown";
          if (uncertain && step) store.putStep({ ...step, state: "unknown", updatedAt: this.iso() });
          store.put({ ...run, state: uncertain || run.state === "stopping" ? "review-required" : "paused", pausedFor: "shutdown", updatedAt: this.iso(),
            detail: uncertain ? "An interrupted effect has no terminal receipt. Review it; it will never be replayed automatically." : "Host sequence retained at its committed boundary. Resume explicitly." }, "run");
        }
      }); } finally { store.close(); }
      this.emit(brainId);
    }
  }
  async revokeRuns(brainId: string, definitionId?: string, detail = "Automation access changed."): Promise<void> {
    // Abort what can be aborted immediately, even if a disk-pressure condition
    // prevents the later durable status write. Never wait for a neural writer.
    for (const active of this.activeRequests.values()) if (active.brainId === brainId && (!definitionId || active.definitionId === definitionId)) this.options.tools.cancel(brainId, active.requestId);
    const store = await this.store(brainId); const pending: Array<{ id: string; requestId?: string; approvalToken?: string }> = [];
    try { store.transaction(() => { for (const run of store.records<AutomationRun>("run", activeStates)) {
      if (definitionId && run.definitionId !== definitionId) continue; const step = store.latestStep(run.id, run.nextStep);
      store.put({ ...run, state: "review-required", updatedAt: this.iso(), detail }, "run"); pending.push({ id: run.id, requestId: step?.requestId, approvalToken: step?.outcome?.approvalToken });
    } }); } finally { store.close(); }
    for (const run of pending) { const key = this.key(brainId, run.id), timer = this.timers.get(key); if (timer) clearTimeout(timer); this.timers.delete(key);
      if (run.approvalToken) this.options.tools.invalidateApproval(brainId, run.approvalToken); if (run.requestId) this.options.tools.cancel(brainId, run.requestId); }
    this.emit(brainId);
  }
  async forget(brainId: string): Promise<void> {
    this.frozen.add(brainId); await this.options.rotateOwnerEpoch(brainId); await this.revokeRuns(brainId, undefined, "Instance recovery/replacement requires fresh action-file review.");
    await Promise.all([...this.pumping].filter(([key]) => JSON.parse(key)[0] === brainId).map(([, pending]) => pending)); this.owners.delete(brainId); this.frozen.delete(brainId);
  }
  async checkpointBeforeQuit(): Promise<void> {
    this.quitting = true;
    for (const timer of this.timers.values()) clearTimeout(timer); this.timers.clear();
    // Let the actual current effect retain its terminal receipt. No new step
    // starts; shutdown does not turn completed effects into Cancel/undo.
    for (const brainId of this.owners) { const store = await this.store(brainId); try { store.transaction(() => {
      for (const run of store.records<AutomationRun>("run", ["queued", "running", "waiting"])) {
        const step = store.latestStep(run.id, run.nextStep);
        if (step?.wait) { const remainingMs = step.wait.sessionId === this.source.sessionId ? Math.max(0, step.wait.monotonicDeadline - this.source.monotonicMs()) : Math.max(0, Date.parse(step.wait.dueAt) - this.source.wallMs());
          store.putStep({ ...step, wait: { ...step.wait, remainingMs }, updatedAt: this.iso() }); }
        store.put({ ...run, state: "paused", pausedFor: "shutdown", updatedAt: this.iso() }, "run");
      }
    }); } finally { store.close(); } }
    let timer: NodeJS.Timeout | undefined;
    try { await Promise.race([this.settle(), new Promise<never>((_resolve, reject) => { timer = setTimeout(() => reject(new Error("An active host effect has not acknowledged its terminal receipt. Studio will remain open; Stop or inspect that action.")), 30_000); timer.unref(); })]); }
    finally { if (timer) clearTimeout(timer); }
  }
  /** A refused application Quit reopens controls, never automatically resumes saved work. */
  resumeAdmissions(): void { this.quitting = false; }
}
function validExecutionState(state: AutomationRun["state"], approval: boolean): boolean { return ["queued", "running", "waiting", ...(approval ? ["approval-required"] : [])].includes(state); }
