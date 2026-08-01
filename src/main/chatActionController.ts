import { randomUUID } from "node:crypto";
import { EventEmitter } from "node:events";
import type {
  ActionEvent,
  ChatResult,
  ChatStreamEvent,
  EvolutionRun,
  EvolutionSourceEdit,
  EvolutionStartRequest,
  IdleCycleResult,
  ModalityPreview,
  RuntimeJob,
  StructuredAction,
  ToolExecutionResult,
  ToolInvocation
} from "../shared/types";
import type { NeuralChatStreamEvent } from "./brainService";

export interface ActionChatService {
  chat(
    brainId: string,
    input: string,
    signal?: AbortSignal,
    onStream?: (event: NeuralChatStreamEvent) => void,
    turnId?: string
  ): Promise<ChatResult>;
  idleCycle?(brainId: string, minimumIdleSeconds?: number): Promise<IdleCycleResult>;
}

export interface ActionToolExecutor {
  execute(
    invocation: ToolInvocation,
    onProgress?: (job: RuntimeJob) => void
  ): Promise<ToolExecutionResult>;
  cancel(brainId: string): number;
  hasPendingOrActive?(brainId: string): boolean;
}

export interface ActionEvolutionController {
  start(request: EvolutionStartRequest): Promise<EvolutionRun>;
}

function serializableToolExperience(action: StructuredAction, output: unknown): string {
  let serialized: string;
  try {
    serialized =
      JSON.stringify(
      output,
      (key, value) =>
        key === "dataUrl" && typeof value === "string"
          ? `[embedded media omitted; ${value.length} characters]`
          : value,
      2
      ) ?? JSON.stringify({ result: "No serializable output was returned." });
  } catch {
    serialized = JSON.stringify({ error: "Action output was not serializable." });
  }
  return [
    "[Visible structured action result]",
    `kind: ${action.kind}`,
    `tool: ${action.toolId ?? "internal"}`,
    `action: ${action.action ?? action.kind}`,
    `requested-by: ${action.source}`,
    "result:",
    serialized.slice(0, 48_000)
  ].join("\n");
}

function actionFingerprint(action: StructuredAction): string {
  // Assembly/concept identifiers are transient neural routing evidence. They
  // can legitimately change after an artifact is fed back into the same turn,
  // but that must not make an otherwise identical action recur forever. Keep
  // explicit human/model intent in the convergence key while ignoring those
  // volatile internal handles for imagination.
  if (action.kind === "imagine" || action.toolId === "modality.imagine") {
    const modality =
      typeof action.arguments.modality === "string"
        ? action.arguments.modality.trim().toLocaleLowerCase()
        : "";
    const prompt =
      typeof action.arguments.prompt === "string"
        ? action.arguments.prompt.replace(/\s+/g, " ").trim()
        : "";
    const inputPath =
      typeof action.arguments.inputPath === "string"
        ? action.arguments.inputPath.trim()
        : "";
    return JSON.stringify([
      action.kind,
      action.toolId,
      action.action,
      modality,
      prompt,
      inputPath
    ]);
  }
  return JSON.stringify([
    action.kind,
    action.toolId,
    action.action,
    action.arguments
  ]);
}

function neuralActionCorrelation(value?: string): string | undefined {
  const normalized = value?.trim().toLocaleLowerCase();
  return normalized && /^[a-f0-9]{32}$/.test(normalized)
    ? normalized
    : undefined;
}

function typedSourceEdits(value: unknown): EvolutionSourceEdit[] | undefined {
  if (value === undefined) return undefined;
  if (!Array.isArray(value)) {
    throw new Error("A source evolution action's sourceEdits must be an array.");
  }
  return value.map((entry) => {
    if (typeof entry !== "object" || entry === null || Array.isArray(entry)) {
      throw new Error("Every source edit must be a typed object.");
    }
    const edit = entry as Record<string, unknown>;
    if (
      typeof edit.path !== "string" ||
      typeof edit.content !== "string" ||
      !(
        edit.expectedSha256 === null ||
        typeof edit.expectedSha256 === "string"
      )
    ) {
      throw new Error(
        "Every source edit requires path, content, and expectedSha256 (or null for a new file)."
      );
    }
    return {
      path: edit.path,
      content: edit.content,
      expectedSha256: edit.expectedSha256
    };
  });
}

function eventFor(
  brainId: string,
  action: StructuredAction,
  neuralActionId?: string
): ActionEvent {
  const now = new Date().toISOString();
  return {
    id: randomUUID(),
    brainId,
    ...(neuralActionId ? { neuralActionId } : {}),
    action,
    state: "proposed",
    createdAt: now,
    updatedAt: now
  };
}

interface ActiveTurn {
  brainId: string;
  turnId: string;
  controller: AbortController;
  nextSequence: number;
}

interface ActionOutcome {
  event: ActionEvent;
  output?: unknown;
}

export class ChatActionController extends EventEmitter {
  private readonly activeTurns = new Map<string, ActiveTurn>();

  constructor(
    private readonly service: ActionChatService,
    private readonly tools: ActionToolExecutor,
    private readonly evolution: ActionEvolutionController
  ) {
    super();
  }

  isBusy(brainId: string): boolean {
    return (
      [...this.activeTurns.values()].some(
        (turn) => turn.brainId === brainId && !turn.controller.signal.aborted
      ) || this.tools.hasPendingOrActive?.(brainId) === true
    );
  }

  private publish(event: ActionEvent): void {
    this.emit("event", JSON.parse(JSON.stringify(event)) as ActionEvent);
  }

  private publishStream(
    turn: ActiveTurn,
    event:
      | Omit<Extract<ChatStreamEvent, { type: "chat-token" }>, "id" | "brainId" | "turnId" | "sequence" | "createdAt">
      | Omit<Extract<ChatStreamEvent, { type: "chat-action" }>, "id" | "brainId" | "turnId" | "sequence" | "createdAt">
      | Omit<Extract<ChatStreamEvent, { type: "modality-preview" }>, "id" | "brainId" | "turnId" | "sequence" | "createdAt">
      | Omit<Extract<ChatStreamEvent, { type: "chat-state" }>, "id" | "brainId" | "turnId" | "sequence" | "createdAt">
  ): void {
    const value = {
      ...event,
      id: randomUUID(),
      brainId: turn.brainId,
      turnId: turn.turnId,
      sequence: turn.nextSequence++,
      createdAt: new Date().toISOString()
    } as ChatStreamEvent;
    this.emit("stream", JSON.parse(JSON.stringify(value)) as ChatStreamEvent);
  }

  private publishAction(turn: ActiveTurn | undefined, event: ActionEvent): void {
    this.publish(event);
    if (turn) this.publishStream(turn, { type: "chat-action", actionEvent: event });
  }

  private async executeAction(
    event: ActionEvent,
    onUpdate: (event: ActionEvent) => void = (value) => this.publish(value)
  ): Promise<unknown> {
    event.state = "running";
    event.updatedAt = new Date().toISOString();
    onUpdate(event);
    const { action } = event;
    if (action.kind === "stop") {
      this.tools.cancel(event.brainId);
      event.state = "stopped";
      event.updatedAt = new Date().toISOString();
      return { stopped: true };
    }
    if (action.kind === "ponder") {
      if (!this.service.idleCycle) {
        throw new Error("The neural worker does not expose internal cognition.");
      }
      // A learned ponder action is an actual second recurrent computation,
      // not a decorative completed card. The worker performs prompt-free
      // liquid/LIF settling, rehearsal and any resulting plastic update.
      const cognition = await this.service.idleCycle(event.brainId, 0);
      event.state = "complete";
      event.updatedAt = new Date().toISOString();
      return { internal: true, kind: action.kind, cognition };
    }
    if (["talk", "learn"].includes(action.kind)) {
      event.state = "complete";
      event.updatedAt = new Date().toISOString();
      return { internal: true, kind: action.kind };
    }
    if (action.kind === "evolve") {
      const objective =
        typeof action.arguments.objective === "string"
          ? action.arguments.objective.trim()
          : "";
      if (!objective) throw new Error("An evolution action requires an objective.");
      const requestedKind = action.arguments.candidateKind;
      const requestedCandidateKind =
        requestedKind === "source" ||
        requestedKind === "neural" ||
        requestedKind === "data" ||
        requestedKind === "substrate" ||
        requestedKind === "architecture"
          ? requestedKind
          : undefined;
      const stringArray = (value: unknown): string[] | undefined =>
        Array.isArray(value) &&
        value.every((entry) => typeof entry === "string")
          ? value
          : undefined;
      const addExperts = action.arguments.addExperts;
      const sourceEdits = typedSourceEdits(action.arguments.sourceEdits);
      const hasTypedSourceEdits = Boolean(sourceEdits?.length);
      // Source changes are the one evolution route that must never be
      // synthesized from an objective alone. Without exact typed edits, use a
      // worker-owned neural overlay; a learned/organic action can therefore
      // improve itself without creating an empty Git candidate that is
      // guaranteed to fail evaluation.
      let candidateKind: NonNullable<EvolutionStartRequest["candidateKind"]> =
        hasTypedSourceEdits
          ? "source"
          : requestedCandidateKind === "source" || requestedCandidateKind === undefined
            ? "substrate"
            : requestedCandidateKind;
      const texts = stringArray(action.arguments.texts);
      const sourceIds = stringArray(action.arguments.sourceIds);
      if (
        candidateKind === "data" &&
        !texts?.length &&
        !sourceIds?.length &&
        action.arguments.latentReplay !== true
      ) {
        candidateKind = "substrate";
      }
      const latentReplay =
        typeof action.arguments.latentReplay === "boolean"
          ? action.arguments.latentReplay
          : candidateKind === "neural" || candidateKind === "substrate"
            ? true
            : undefined;
      const run = await this.evolution.start({
        brainId: event.brainId,
        objective,
        recursive: action.arguments.recursive !== false,
        candidateKind,
        texts,
        sourceIds,
        epochs:
          typeof action.arguments.epochs === "number"
            ? action.arguments.epochs
            : undefined,
        learningRate:
          typeof action.arguments.learningRate === "number"
            ? action.arguments.learningRate
            : undefined,
        latentReplay,
        objectives: stringArray(action.arguments.objectives),
        ...(hasTypedSourceEdits ? { sourceEdits } : {}),
        architectureChange:
          candidateKind === "architecture"
            ? {
                mutation: "grow-experts",
                addExperts:
                  typeof addExperts === "number" ? addExperts : undefined
              }
            : undefined
      });
      event.evolutionRunId = run.id;
      event.state = run.state === "failed" ? "failed" : "complete";
      event.error = run.error;
      event.updatedAt = new Date().toISOString();
      return run;
    }
    if (!action.toolId || !action.action) {
      throw new Error("The structured action is missing its tool protocol.");
    }
    const imagination =
      action.kind === "imagine" || action.toolId === "modality.imagine";
    const invocation = {
      brainId: event.brainId,
      toolId: action.toolId,
      action: action.action,
      arguments:
        imagination && event.neuralActionId
          ? { ...action.arguments, neuralActionId: event.neuralActionId }
          : action.arguments
    };
    const execution =
      imagination
        ? await this.tools.execute(invocation, (job) => {
            event.runtimeJobId = job.id;
            event.progress = Math.max(0, Math.min(1, job.progress));
            event.statusLabel = job.label;
            if (
              job.preview &&
              (!event.preview || job.preview.revision > event.preview.revision)
            ) {
              event.preview = job.preview;
            }
            event.updatedAt = job.updatedAt;
            onUpdate(event);
          })
        : await this.tools.execute(invocation);
    event.execution = execution;
    event.state = execution.state;
    event.error = execution.error;
    event.updatedAt = execution.finishedAt ?? new Date().toISOString();
    return execution.output;
  }

  async send(
    brainId: string,
    input: string,
    signal?: AbortSignal,
    requestedTurnId?: string
  ): Promise<ChatResult> {
    const turnId = requestedTurnId?.trim() || randomUUID();
    if (this.activeTurns.has(turnId)) {
      throw new Error("This chat turn is already active.");
    }
    const controller = new AbortController();
    const turn: ActiveTurn = {
      brainId,
      turnId,
      controller,
      nextSequence: 0
    };
    const abortFromCaller = (): void => controller.abort();
    signal?.addEventListener("abort", abortFromCaller, { once: true });
    if (signal?.aborted) controller.abort();
    this.activeTurns.set(turnId, turn);
    const seen = new Set<string>();
    const events: ActionEvent[] = [];
    const workerActions = new Map<string, ActionEvent>();
    const previewRevisions = new Map<string, number>();
    const executions: Array<Promise<ActionOutcome>> = [];
    let latestImagination: ActionEvent | undefined;

    const updateAction = (event: ActionEvent): void => {
      this.publishAction(turn, event);
      if (
        event.preview &&
        event.preview.revision > (previewRevisions.get(event.id) ?? -1)
      ) {
        previewRevisions.set(event.id, event.preview.revision);
        this.publishStream(turn, {
          type: "modality-preview",
          actionId: event.id,
          preview: event.preview
        });
      }
    };
    const queueAction = (
      action: StructuredAction,
      workerActionId?: string
    ): ActionEvent | undefined => {
      const fingerprint = actionFingerprint(action);
      if (seen.has(fingerprint)) {
        return workerActionId ? workerActions.get(workerActionId) : undefined;
      }
      seen.add(fingerprint);
      const event = eventFor(
        brainId,
        action,
        neuralActionCorrelation(workerActionId)
      );
      events.push(event);
      if (workerActionId) workerActions.set(workerActionId, event);
      if (action.kind === "imagine") latestImagination = event;
      this.publishAction(turn, event);
      const execution = (async (): Promise<ActionOutcome> => {
        let output: unknown;
        try {
          controller.signal.throwIfAborted();
          output = await this.executeAction(event, updateAction);
        } catch (error) {
          event.state = controller.signal.aborted ? "stopped" : "failed";
          event.error = error instanceof Error ? error.message : String(error);
          event.updatedAt = new Date().toISOString();
        }
        this.publishAction(turn, event);
        return { event, output };
      })();
      executions.push(execution);
      return event;
    };
    const consumeNeuralStream = (neural: NeuralChatStreamEvent): void => {
      if (controller.signal.aborted) return;
      if (neural.type === "chat-token") {
        this.publishStream(turn, { type: "chat-token", delta: neural.delta });
        return;
      }
      if (neural.type === "chat-action") {
        queueAction(neural.action, neural.actionId);
        return;
      }
      const action =
        (neural.actionId ? workerActions.get(neural.actionId) : undefined) ??
        latestImagination;
      if (!action) return;
      if (action.preview && neural.preview.revision <= action.preview.revision) return;
      action.preview = neural.preview;
      if (neural.preview.progress !== undefined) {
        action.progress = neural.preview.progress;
      }
      if (neural.preview.statusLabel) {
        action.statusLabel = neural.preview.statusLabel;
      }
      action.updatedAt = new Date().toISOString();
      updateAction(action);
    };

    this.publishStream(turn, { type: "chat-state", state: "started" });
    try {
      let result = await this.service.chat(
        brainId,
        input,
        controller.signal,
        consumeNeuralStream,
        turnId
      );
      for (const action of result.proposedActions ?? []) queueAction(action);

      // Every action starts as soon as its typed event arrives. Results are
      // integrated sequentially only after the initial neural response has
      // committed, avoiding concurrent mutation of one persistent identity.
      for (let index = 0; index < executions.length; index += 1) {
        controller.signal.throwIfAborted();
        const { event, output } = await executions[index]!;
        if (event.state !== "complete") continue;
        if (["talk", "ponder", "learn"].includes(event.action.kind)) continue;
        result = await this.service.chat(
          brainId,
          serializableToolExperience(event.action, output),
          controller.signal,
          consumeNeuralStream,
          turnId
        );
        for (const action of result.proposedActions ?? []) queueAction(action);
      }
      result.actionEvents = events;
      this.publishStream(turn, { type: "chat-state", state: "complete" });
      return result;
    } catch (error) {
      const cancelled = controller.signal.aborted;
      if (cancelled) {
        for (const event of events) {
          if (!["proposed", "running"].includes(event.state)) continue;
          event.state = "stopped";
          event.updatedAt = new Date().toISOString();
          this.publishAction(turn, event);
        }
      }
      this.publishStream(turn, {
        type: "chat-state",
        state: cancelled ? "cancelled" : "failed",
        error: error instanceof Error ? error.message : String(error)
      });
      throw error;
    } finally {
      signal?.removeEventListener("abort", abortFromCaller);
      this.activeTurns.delete(turnId);
    }
  }

  cancel(brainId: string, turnId?: string): number {
    let cancelled = 0;
    for (const turn of this.activeTurns.values()) {
      if (
        turn.brainId !== brainId ||
        (turnId !== undefined && turn.turnId !== turnId) ||
        turn.controller.signal.aborted
      ) {
        continue;
      }
      turn.controller.abort();
      cancelled += 1;
    }
    return cancelled + (cancelled > 0 || turnId === undefined
      ? this.tools.cancel(brainId)
      : 0);
  }

  async idle(
    brainId: string,
    minimumIdleSeconds = 45
  ): Promise<IdleCycleResult> {
    if (!this.service.idleCycle) {
      return {
        brainId,
        ran: false,
        reason: "idle-cognition-disabled",
        actions: []
      };
    }
    const cycle = await this.service.idleCycle(brainId, minimumIdleSeconds);
    if (!cycle.ran || cycle.actions.length === 0) return cycle;
    const events: ActionEvent[] = [];
    const seen = new Set<string>();
    // Organic activity is intentionally conservative: at most two typed
    // actions can leave one idle cycle, and every external action traverses
    // the same permission and audit path used by chat.
    for (const candidate of cycle.actions) {
      if (events.length >= 2) break;
      const action: StructuredAction = { ...candidate, source: "organic" };
      const fingerprint = actionFingerprint(action);
      if (seen.has(fingerprint)) continue;
      seen.add(fingerprint);
      const event = eventFor(brainId, action);
      events.push(event);
      this.publish(event);
      try {
        await this.executeAction(event);
      } catch (error) {
        event.state = "failed";
        event.error = error instanceof Error ? error.message : String(error);
        event.updatedAt = new Date().toISOString();
      }
      this.publish(event);
    }
    return { ...cycle, actionEvents: events };
  }
}
