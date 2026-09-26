import { randomUUID } from "node:crypto";
import { EventEmitter } from "node:events";
import type {
  ActionEvent,
  ApprovedChatActionRequest,
  ApprovedChatActionResult,
  ChatCancellationState,
  ChatQueueState,
  ChatResult,
  ChatStreamEvent,
  ChatTurnMetadata,
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
import type {
  ConfirmedToolRouteOutcome,
  NeuralChatStreamEvent,
  StructuredNeuralExperience,
  ToolRouteLearningResult
} from "./brainService";

export interface ActionChatService {
  chat(
    brainId: string,
    input: string,
    signal?: AbortSignal,
    onStream?: (event: NeuralChatStreamEvent) => void,
    turnId?: string
  ): Promise<ChatResult>;
  idleCycle?(brainId: string, minimumIdleSeconds?: number): Promise<IdleCycleResult>;
  learnStructuredExperience?(
    brainId: string,
    experience: StructuredNeuralExperience,
    signal?: AbortSignal
  ): Promise<{ brain?: ChatResult["brain"] }>;
  learnToolRouteOutcome?(
    brainId: string,
    outcome: ConfirmedToolRouteOutcome,
    signal?: AbortSignal
  ): Promise<ToolRouteLearningResult>;
  recordConversationActions?(brainId: string, actions: ActionEvent[]): Promise<void>;
}

export interface ActionToolExecutor {
  execute(
    invocation: ToolInvocation,
    onProgress?: (job: RuntimeJob) => void,
    requestId?: string
  ): Promise<ToolExecutionResult>;
  cancel(brainId: string, requestId?: string): number;
  hasPendingOrActive?(brainId: string): boolean;
}

export interface ActionEvolutionController {
  start(request: EvolutionStartRequest): Promise<EvolutionRun>;
}

function canonicalSystemToolId(toolId: string | undefined): string | undefined {
  return toolId === "windows.files"
    ? "system.files"
    : toolId === "windows.powershell"
      ? "system.shell"
      : toolId;
}

function serializableToolExperience(action: StructuredAction, output: unknown): string {
  let serialized: string;
  try {
    serialized =
      JSON.stringify(
      output,
      (key, value) =>
        (key === "dataUrl" || key === "mediaUrl") && typeof value === "string"
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
    `tool: ${canonicalSystemToolId(action.toolId) ?? "internal"}`,
    `action: ${action.action ?? action.kind}`,
    `requested-by: ${action.source}`,
    "result:",
    serialized.slice(0, 48_000)
  ].join("\n");
}

export function confirmedArgumentTrainingFields(
  action: StructuredAction
): Record<string, unknown> {
  const toolId = canonicalSystemToolId(action.toolId) ?? action.toolId;
  const credentialPattern = /\b(?:password|passwd|secret|api[_-]?key|access[_-]?token|authorization|bearer)\s*[:=]|\b(?:sk-[a-z0-9_-]{16,}|ghp_[a-z0-9]{16,})\b/i;
  if (toolId === "browser.automation" && action.action === "task") {
    // A completed host invocation may teach an operation kind. Never admit
    // typed page content or a multi-step program to autonomous supervision.
    const url = action.arguments.url;
    const steps = action.arguments.steps;
    if (typeof url !== "string" || url.length > 16_000 || /[\r\n\0]/.test(url) ||
        !url.startsWith("https://") || credentialPattern.test(url) ||
        (steps !== undefined && !Array.isArray(steps))) return {};
    let parsedUrl: URL;
    try {
      parsedUrl = new URL(url);
    } catch {
      return {};
    }
    if (parsedUrl.protocol !== "https:" || parsedUrl.username || parsedUrl.password) return {};
    const selected = Array.isArray(steps) ? steps : [];
    if (selected.length > 1) return {};
    if (selected.length === 0) {
      return {
        browserOperation: "none",
        ...(parsedUrl.search || parsedUrl.hash
          ? {} : { browserActionArguments: { url, steps: [] } })
      };
    }
    const step = selected[0];
    if (typeof step !== "object" || step === null || Array.isArray(step)) return {};
    const kind = (step as Record<string, unknown>).kind;
    if (typeof kind !== "string" || ![
      "click", "type", "press", "wait", "extract", "screenshot", "navigate"
    ].includes(kind)) return {};
    if (kind === "type") {
      // The operation label is safe to learn; entered text is not.
      return { browserOperation: kind };
    }
    // Keep the kind target independent of operands. For idle argument
    // training admit only a canonical selector or no-operand screenshot; a
    // URL query/fragment and arbitrary nested step fields are never copied.
    if (parsedUrl.search || parsedUrl.hash) return { browserOperation: kind };
    if (kind === "click") {
      const selector = (step as Record<string, unknown>).selector;
      if (typeof selector !== "string" || !selector || selector.length > 2_000 ||
          /[\r\n\0]/.test(selector) || credentialPattern.test(selector)) {
        return { browserOperation: kind };
      }
      return {
        browserOperation: kind,
        browserActionArguments: { url, steps: [{ kind, selector }] }
      };
    }
    if (kind === "screenshot") {
      return {
        browserOperation: kind,
        browserActionArguments: { url, steps: [{ kind }] }
      };
    }
    return { browserOperation: kind };
  }
  const selectedFields =
    toolId === "web.search" ? ["query"] :
      toolId === "agent.fork" ? ["objective"] :
        toolId === "source.self-modify" ? ["objective", "candidateKind"] : [];
  return Object.fromEntries(
    selectedFields.flatMap((key) => {
      const value = action.arguments[key];
      return typeof value === "string" &&
        value.trim().length > 0 && value.length <= 1024 &&
        !credentialPattern.test(value)
        ? [[key, value.trim()] as const]
        : [];
    })
  );
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
      inputPath,
      action.arguments.settings ?? null
    ]);
  }
  return JSON.stringify([
    action.kind,
    canonicalSystemToolId(action.toolId),
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
  turnMetadata?: ChatTurnMetadata;
  runtimePhase: "pending" | "queued" | "running";
  queue?: ChatQueueState;
  cancelledPhase?: ChatCancellationState["phase"];
  cancelledQueue?: ChatQueueState;
  runtimeCancellation?: ChatCancellationState;
  runtimeCancellationFailed?: boolean;
  settled: Promise<void>;
  resolveSettled(): void;
}

interface ActionOutcome {
  event: ActionEvent;
  output?: unknown;
}

async function waitForTurnSettlement(
  predecessor: ActiveTurn,
  signal: AbortSignal
): Promise<void> {
  signal.throwIfAborted();
  let rejectAbort!: (error: DOMException) => void;
  const aborted = new Promise<never>((_resolve, reject) => {
    rejectAbort = reject;
  });
  const abort = (): void => {
    rejectAbort(new DOMException("This queued chat turn was cancelled.", "AbortError"));
  };
  signal.addEventListener("abort", abort, { once: true });
  try {
    await Promise.race([predecessor.settled, aborted]);
    signal.throwIfAborted();
  } finally {
    signal.removeEventListener("abort", abort);
  }
}

/**
 * Active turns already have an ordered stream. Keep that transport light: a
 * progressive preview is delivered exactly once through `modality-preview`,
 * while the authoritative result (including any final tool output) returns
 * through the chat invocation. Idle/global actions still use the complete
 * legacy ActionEvent channel because they have no turn stream.
 */
export function actionEventForTurnStream(event: ActionEvent): ActionEvent {
  const { preview: _preview, execution, ...rest } = event;
  if (!execution) return rest;
  const { output: _output, ...executionWithoutOutput } = execution;
  return {
    ...rest,
    execution: executionWithoutOutput
  };
}

export class ChatActionController extends EventEmitter {
  private readonly activeTurns = new Map<string, ActiveTurn>();
  private readonly pendingApprovedActions = new Map<
    string,
    { event: ActionEvent; expiresAt: number; userUtterance?: string }
  >();
  private readonly preCancelledTurns = new Map<
    string,
    { brainId: string; expiresAt: number }
  >();

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
      | Omit<Extract<ChatStreamEvent, { type: "chat-phase" }>, "id" | "brainId" | "turnId" | "sequence" | "createdAt">
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
    if (turn) {
      this.publishStream(turn, {
        type: "chat-action",
        actionEvent: actionEventForTurnStream(event)
      });
      return;
    }
    this.publish(event);
  }

  private prunePendingApprovedActions(): void {
    const now = Date.now();
    for (const [eventId, pending] of this.pendingApprovedActions) {
      if (pending.expiresAt < now) this.pendingApprovedActions.delete(eventId);
    }
  }

  private rememberPendingApprovedAction(event: ActionEvent, userUtterance?: string): void {
    const execution = event.execution;
    if (
      event.state !== "approval-required" ||
      !execution?.approvalToken ||
      !execution.approvalExpiresAt
    ) {
      this.pendingApprovedActions.delete(event.id);
      return;
    }
    const expiresAt = Date.parse(execution.approvalExpiresAt);
    if (!Number.isFinite(expiresAt)) {
      this.pendingApprovedActions.delete(event.id);
      return;
    }
    this.prunePendingApprovedActions();
    this.pendingApprovedActions.set(event.id, { event, expiresAt, userUtterance });
    while (this.pendingApprovedActions.size > 256) {
      const oldest = this.pendingApprovedActions.keys().next().value;
      if (oldest === undefined) break;
      this.pendingApprovedActions.delete(oldest);
    }
  }

  private async learnConfirmedToolRoute(
    event: ActionEvent,
    userUtterance?: string,
    signal?: AbortSignal
  ): Promise<ToolRouteLearningResult | undefined> {
    const action = event.action;
    if (
      event.state !== "complete" ||
      !["tool", "agent", "evolve"].includes(action.kind) ||
      !action.toolId ||
      !action.action ||
      !userUtterance?.trim() ||
      !this.service.learnToolRouteOutcome
    ) {
      return undefined;
    }
    const toolId = canonicalSystemToolId(action.toolId) ?? action.toolId;
    // Only confirmed, bounded query/objective fields train autonomous
    // argument generation. Never pass file contents, shell commands, source
    // edits, or opaque MCP payloads into that head.
    const typedArguments = confirmedArgumentTrainingFields(action);
    return this.service.learnToolRouteOutcome(
      event.brainId,
      {
        eventId: event.id,
        utterance: userUtterance,
        toolId,
        action: action.action,
        arguments: typedArguments
      },
      signal
    );
  }

  private async executeAction(
    event: ActionEvent,
    onUpdate: (event: ActionEvent) => void = (value) => this.publish(value),
    requestId?: string,
    approvalToken?: string,
    userUtterance?: string
  ): Promise<unknown> {
    event.state = "running";
    event.error = undefined;
    event.updatedAt = new Date().toISOString();
    onUpdate(event);
    const { action } = event;
    if (action.kind === "stop") {
      this.tools.cancel(event.brainId, requestId);
      event.state = "stopped";
      event.updatedAt = new Date().toISOString();
      return { stopped: true };
    }
    if (action.kind === "ponder") {
      if (action.arguments.completedInTurn === true) {
        // The active OmniCortex can finish Ponder privately before it emits
        // the first answer token. Consume that typed result exactly once;
        // starting an idle cycle here would perform a redundant post-response
        // thought and misrepresent when the cognition affected the answer.
        event.state = "complete";
        event.updatedAt = new Date().toISOString();
        return {
          internal: true,
          kind: action.kind,
          cognition: action.arguments.ponderTrace,
          phase: "pre-speech",
          reused: false
        };
      }
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
      const modelOwned = action.source === "brain" || action.source === "organic";
      // Exact source edits are the only way to enter the source path. Every
      // other model-owned candidate kind must be selected explicitly by the
      // typed neural action; never silently reinterpret it as substrate.
      if (!hasTypedSourceEdits && requestedCandidateKind === "source") {
        throw new Error("Source evolution requires exact typed source edits.");
      }
      if (!hasTypedSourceEdits && modelOwned && !requestedCandidateKind) {
        throw new Error("The neural evolution action needs a selected candidate kind.");
      }
      const candidateKind: NonNullable<EvolutionStartRequest["candidateKind"]> =
        hasTypedSourceEdits
          ? "source"
          : requestedCandidateKind ?? "substrate";
      const texts = stringArray(action.arguments.texts);
      const sourceIds = stringArray(action.arguments.sourceIds);
      if (
        candidateKind === "data" &&
        !texts?.length &&
        !sourceIds?.length &&
        action.arguments.latentReplay !== true
      ) {
        throw new Error("A data evolution candidate needs source evidence or explicit latent replay.");
      }
      if (
        modelOwned && candidateKind === "architecture" &&
        (typeof addExperts !== "number" || !Number.isInteger(addExperts) || addExperts < 1)
      ) {
        throw new Error("An architecture candidate needs a typed positive expert-growth amount.");
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
          : action.arguments,
      ...(approvalToken ? { approvalToken } : {})
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
          }, requestId)
        : requestId
          ? await this.tools.execute(invocation, undefined, requestId)
          : await this.tools.execute(invocation);
    event.execution = execution;
    event.state = execution.state;
    event.error = execution.error;
    event.updatedAt = execution.finishedAt ?? new Date().toISOString();
    this.rememberPendingApprovedAction(event, userUtterance);
    return execution.output;
  }

  async approveAction(
    request: ApprovedChatActionRequest
  ): Promise<ApprovedChatActionResult> {
    this.prunePendingApprovedActions();
    const pending = this.pendingApprovedActions.get(request.actionEventId);
    if (!pending || pending.event.brainId !== request.brainId) {
      throw new Error("The approved chat action is unavailable or expired.");
    }
    const pendingExecution = pending.event.execution;
    if (
      pending.event.state !== "approval-required" ||
      pendingExecution?.approvalToken !== request.approvalToken ||
      pending.expiresAt < Date.now()
    ) {
      throw new Error("The approved chat action token is invalid or expired.");
    }

    // Claim the visible action before awaiting so double-clicks and replayed
    // IPC requests cannot execute or learn it twice.
    this.pendingApprovedActions.delete(request.actionEventId);
    const event = pending.event;
    let output: unknown;
    try {
      output = await this.executeAction(
        event,
        (value) => this.publish(value),
        event.id,
        request.approvalToken,
        pending.userUtterance
      );
    } catch (error) {
      event.state = "failed";
      event.error = error instanceof Error ? error.message : String(error);
      event.updatedAt = new Date().toISOString();
    }

    let brain: ChatResult["brain"] | undefined;
    let learned = false;
    let learningError: string | undefined;
    if (event.state === "complete") {
      event.statusLabel = "Action complete · integrating result";
      event.updatedAt = new Date().toISOString();
      this.publish(event);
      try {
        await this.learnConfirmedToolRoute(event, pending.userUtterance);
      } catch (error) {
        learningError = `Tool-route learning: ${error instanceof Error ? error.message : String(error)}`;
      }
      try {
        if (!this.service.learnStructuredExperience) {
          throw new Error("Structured action-result learning is unavailable.");
        }
        const result = await this.service.learnStructuredExperience(request.brainId, {
          content: serializableToolExperience(event.action, output),
          name: `Chat ${canonicalSystemToolId(event.action.toolId) ?? event.action.kind} result`,
          sourceLabel: "chat visible approved action evidence",
          license: "Locally observed tool result"
        });
        brain = result.brain;
        learned = true;
        event.statusLabel = learningError
          ? "Action complete · result learned; route learning pending"
          : "Action complete · result learned";
        event.error = learningError;
      } catch (error) {
        const resultError = error instanceof Error ? error.message : String(error);
        learningError = learningError ? `${learningError}; result: ${resultError}` : resultError;
        event.error = `Action completed, but its result could not enter neural learning: ${learningError}`;
        event.statusLabel = "Action complete · result learning failed";
      }
      event.updatedAt = new Date().toISOString();
    }

    await this.service.recordConversationActions?.(request.brainId, [event]);
    this.publish(event);
    return {
      actionEvent: JSON.parse(JSON.stringify(event)) as ActionEvent,
      learned,
      ...(brain ? { brain } : {}),
      ...(learningError ? { learningError } : {})
    };
  }

  async send(
    brainId: string,
    input: string,
    signal?: AbortSignal,
    requestedTurnId?: string,
    turnMetadata?: ChatTurnMetadata
  ): Promise<ChatResult> {
    const turnId = requestedTurnId?.trim() || randomUUID();
    this.prunePreCancelledTurns();
    const preCancelled = this.preCancelledTurns.get(turnId);
    if (preCancelled?.brainId === brainId) {
      this.preCancelledTurns.delete(turnId);
      throw new DOMException("This chat turn was cancelled before it started.", "AbortError");
    }
    if (this.activeTurns.has(turnId)) {
      throw new Error("This chat turn is already active.");
    }
    const steerPredecessor = turnMetadata?.kind === "steer"
      ? this.activeTurns.get(turnMetadata.replacesTurnId)
      : undefined;
    const warmHandoffPredecessor =
      steerPredecessor?.brainId === brainId ? steerPredecessor : undefined;
    const controller = new AbortController();
    let resolveSettled!: () => void;
    const settled = new Promise<void>((resolve) => {
      resolveSettled = resolve;
    });
    const turn: ActiveTurn = {
      brainId,
      turnId,
      controller,
      nextSequence: 0,
      turnMetadata,
      runtimePhase: "pending",
      settled,
      resolveSettled
    };
    const abortFromCaller = (): void => controller.abort();
    signal?.addEventListener("abort", abortFromCaller, { once: true });
    if (signal?.aborted) controller.abort();
    this.activeTurns.set(turnId, turn);
    const seen = new Set<string>();
    const events: ActionEvent[] = [];
    const workerActions = new Map<string, ActionEvent>();
    const previewRevisions = new Map<string, number>();
    const executions: Array<() => Promise<ActionOutcome>> = [];
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
      const execute = async (): Promise<ActionOutcome> => {
        let output: unknown;
        try {
          controller.signal.throwIfAborted();
          output = await this.executeAction(
            event,
            updateAction,
            turn.turnId,
            undefined,
            input
          );
        } catch (error) {
          event.state = controller.signal.aborted ? "stopped" : "failed";
          event.error = error instanceof Error ? error.message : String(error);
          event.updatedAt = new Date().toISOString();
        }
        this.publishAction(turn, event);
        return { event, output };
      };
      executions.push(execute);
      return event;
    };
    const consumeNeuralStream = (neural: NeuralChatStreamEvent): void => {
      if (neural.type === "runtime-activity") {
        if (neural.state === "queued" && neural.queue) {
          turn.runtimePhase = "queued";
          turn.queue = neural.queue;
          this.publishStream(turn, {
            type: "chat-state",
            state: "queued",
            queue: neural.queue
          });
        } else if (neural.state === "running") {
          const wasQueued = turn.runtimePhase === "queued";
          turn.runtimePhase = "running";
          turn.queue = undefined;
          if (wasQueued) {
            this.publishStream(turn, { type: "chat-state", state: "started" });
          }
        } else if (
          (neural.state === "cancelled" || neural.state === "failed") &&
          neural.cancellation
        ) {
          turn.runtimeCancellation = neural.cancellation;
          turn.runtimeCancellationFailed = neural.state === "failed";
        }
        return;
      }
      if (controller.signal.aborted) return;
      if (neural.type === "chat-token") {
        this.publishStream(turn, { type: "chat-token", delta: neural.delta });
        return;
      }
      if (neural.type === "chat-phase") {
        this.publishStream(turn, {
          type: "chat-phase",
          phase: neural.phase,
          replyComplete: neural.replyComplete,
          turnCommitted: neural.turnCommitted,
          learning: neural.learning,
          saving: neural.saving
        });
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
      // Auto/Full inline imagination is already decoding in the worker when
      // it publishes a preview. The trusted tool claim remains deferred until
      // the neural turn commits, but the visible state should still reflect
      // that real worker-side activity.
      if (action.state === "proposed") action.state = "running";
      action.updatedAt = new Date().toISOString();
      updateAction(action);
    };

    try {
      if (warmHandoffPredecessor) {
        // The Python neural runtime is deliberately serial. Keep the typed
        // Steer turn queued until its predecessor settles on the same warm
        // worker; steering must not cancel or overwrite the active user turn.
        const queue: ChatQueueState = {
          position: 1,
          queuedBehind: {
            requestId: warmHandoffPredecessor.turnId,
            owner: "chat",
            label: "Chat response",
            method: "chat",
            brainId,
            turnId: warmHandoffPredecessor.turnId
          }
        };
        turn.runtimePhase = "queued";
        turn.queue = queue;
        this.publishStream(turn, {
          type: "chat-state",
          state: "queued",
          queue,
          turnMetadata
        });
        await waitForTurnSettlement(warmHandoffPredecessor, controller.signal);
        turn.runtimePhase = "pending";
        turn.queue = undefined;
      }
      controller.signal.throwIfAborted();
      this.publishStream(turn, {
        type: "chat-state",
        state: "started",
        turnMetadata
      });
      let result = await this.service.chat(
        brainId,
        input,
        controller.signal,
        consumeNeuralStream,
        turnId
      );
      for (const action of result.proposedActions ?? []) queueAction(action);

      // Typed events and worker-side inline previews publish immediately, but
      // trusted tool execution/auditing starts only after the neural response
      // commits. This prevents two whole-brain persistence paths from racing
      // and losing the original turn while preserving real-time imagination.
      for (let index = 0; index < executions.length; index += 1) {
        controller.signal.throwIfAborted();
        const { event, output } = await executions[index]!();
        if (event.state !== "complete") continue;
        if (["talk", "ponder", "learn"].includes(event.action.kind)) continue;
        // The user-authored chat turn is already committed. Learn the visible
        // action result through the typed neural-ingestion path instead of
        // manufacturing a second human message and recursively running the
        // action policy. A future continuation must be an explicit typed
        // protocol; action-result learning never revises response prose.
        this.publishStream(turn, {
          type: "chat-phase",
          phase: "action-result-learning",
          replyComplete: true,
          turnCommitted: true,
          learning: true,
          saving: true
        });
        try {
          await this.learnConfirmedToolRoute(event, input, controller.signal);
        } catch (error) {
          if (controller.signal.aborted) throw error;
          event.error = `Action completed, but tool-route learning is pending: ${
            error instanceof Error ? error.message : String(error)
          }`;
          event.statusLabel = "Action complete · route learning pending";
          event.updatedAt = new Date().toISOString();
          this.publishAction(turn, event);
        }
        if (!this.service.learnStructuredExperience) continue;
        try {
          const learned = await this.service.learnStructuredExperience(
            brainId,
            {
              content: serializableToolExperience(event.action, output),
              name: `Chat ${canonicalSystemToolId(event.action.toolId) ?? event.action.kind} result`,
              sourceLabel: "chat visible action evidence",
              license: "Locally observed tool result"
            },
            controller.signal
          );
          if (learned?.brain) {
            result = {
              ...result,
              brain: {
                ...learned.brain,
                // Repository saves may return compact presentation mirrors;
                // keep the authoritative visible turn/trace from the chat
                // result while adopting the newly learned neural counters and
                // source ledger.
                messages: result.brain.messages,
                traces: result.brain.traces
              }
            };
          }
        } catch (error) {
          if (controller.signal.aborted) throw error;
          event.error = `Action completed, but its result could not enter neural learning: ${
            error instanceof Error ? error.message : String(error)
          }`;
          event.statusLabel = "Action complete · result learning failed";
          event.updatedAt = new Date().toISOString();
          this.publishAction(turn, event);
        }
      }
      const attentionEpoch = result.brainMessage.attentionEpoch ??
        result.humanMessage.attentionEpoch;
      if (attentionEpoch !== undefined) {
        events.forEach((event) => {
          event.attentionEpoch ??= attentionEpoch;
        });
      }
      result.actionEvents = events;
      await this.service.recordConversationActions?.(brainId, events);
      result.turnMetadata = turnMetadata;
      this.publishStream(turn, { type: "chat-state", state: "complete" });
      return result;
    } catch (error) {
      const cancelled = controller.signal.aborted;
      const cancellationFailed = cancelled && turn.runtimeCancellationFailed === true;
      const failureMessage = error instanceof Error ? error.message : String(error);
      // A worker/process failure can arrive after an organic inline preview
      // marked its action Running but before the deferred tool claim begins.
      // Finalize every unfinished visible action so recovery never leaves a
      // ghost "In progress" card attached to the abandoned turn.
      for (const event of events) {
        if (!["proposed", "running"].includes(event.state)) continue;
        event.state = cancelled ? "stopped" : "failed";
        if (!cancelled) event.error = failureMessage;
        event.updatedAt = new Date().toISOString();
        this.publishAction(turn, event);
      }
      this.publishStream(turn, {
        type: "chat-state",
        state: cancelled && !cancellationFailed ? "cancelled" : "failed",
        error: failureMessage,
        ...(cancelled
          ? {
              cancellation: {
                phase:
                  turn.runtimeCancellation?.phase ??
                  turn.cancelledPhase ??
                  turn.runtimePhase,
                unrelatedActivityContinues:
                  turn.runtimeCancellation?.unrelatedActivityContinues === true ||
                  Boolean(
                    (turn.cancelledPhase ?? turn.runtimePhase) === "queued" &&
                    (turn.cancelledQueue ?? turn.queue)?.queuedBehind
                  ),
                workerTerminationAcknowledged:
                  turn.runtimeCancellation?.workerTerminationAcknowledged === true
              }
            }
          : {})
      });
      throw error;
    } finally {
      signal?.removeEventListener("abort", abortFromCaller);
      this.activeTurns.delete(turnId);
      turn.resolveSettled();
    }
  }

  async cancelAndWait(brainId: string, turnId?: string): Promise<number> {
    const matching = [...this.activeTurns.values()].filter(
      (turn) =>
        turn.brainId === brainId &&
        (turnId === undefined || turn.turnId === turnId)
    );
    const cancelled = this.cancel(brainId, turnId);
    await Promise.all(matching.map((turn) => turn.settled));
    return cancelled;
  }

  cancel(brainId: string, turnId?: string): number {
    let cancelled = 0;
    const cancelledTurnIds: string[] = [];
    for (const turn of this.activeTurns.values()) {
      if (
        turn.brainId !== brainId ||
        (turnId !== undefined && turn.turnId !== turnId) ||
        turn.controller.signal.aborted
      ) {
        continue;
      }
      turn.cancelledPhase = turn.runtimePhase;
      turn.cancelledQueue = turn.queue;
      turn.controller.abort();
      cancelledTurnIds.push(turn.turnId);
      cancelled += 1;
    }
    if (cancelled > 0) {
      this.emit("neural-cancelled", { brainId, turnId });
    }
    if (turnId !== undefined && cancelled === 0) {
      this.prunePreCancelledTurns();
      this.preCancelledTurns.set(turnId, {
        brainId,
        expiresAt: Date.now() + 60_000
      });
      while (this.preCancelledTurns.size > 256) {
        const oldest = this.preCancelledTurns.keys().next().value;
        if (oldest === undefined) break;
        this.preCancelledTurns.delete(oldest);
      }
    }
    let cancelledTools = 0;
    for (const cancelledTurnId of cancelledTurnIds) {
      cancelledTools += this.tools.cancel(brainId, cancelledTurnId);
    }
    return cancelled + cancelledTools;
  }

  private prunePreCancelledTurns(): void {
    const now = Date.now();
    for (const [turnId, cancellation] of this.preCancelledTurns) {
      if (cancellation.expiresAt <= now) this.preCancelledTurns.delete(turnId);
    }
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
    // The worker transport is already byte-bounded. Preserve every distinct
    // neural choice from this cycle; each external action still traverses the
    // same visible permission, audit, and cancellation path used by chat.
    for (const candidate of cycle.actions) {
      const action: StructuredAction = { ...candidate, source: "organic" };
      const fingerprint = actionFingerprint(action);
      if (seen.has(fingerprint)) continue;
      seen.add(fingerprint);
      const event = eventFor(brainId, action);
      events.push(event);
      this.publish(event);
      try {
        const output = await this.executeAction(event);
        if (
          event.state === "complete" &&
          !["talk", "ponder", "learn", "stop"].includes(event.action.kind) &&
          this.service.learnStructuredExperience
        ) {
          const experience = serializableToolExperience(event.action, output);
          await this.service.learnStructuredExperience(brainId, {
            content: experience,
            name: `Organic ${event.action.toolId ?? event.action.kind} result`,
            sourceLabel: "organic visible action evidence",
            license: "Locally observed tool result"
          });
        }
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
