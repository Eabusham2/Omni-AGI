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
import { normalizeCompatibleArchitectureMutation } from "../shared/architectureMutation";
import { EngineRequestError } from "./engineSupervisor";

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
  cancelInlineImagination?(brainId: string, turnId: string, actionId: string): Promise<{
    requested: boolean; acknowledged: boolean;
  }>;
  onInlineImaginationCancelled?(listener: (event: {
    brainId: string; turnId: string; actionId: string;
  }) => void): () => void;
  onCodecRuntimeSetup?(listener: (event: { brainId: string; turnId: string; actionId: string; message: string }) => void): () => void;
  steerChat?(brainId: string, turnId: string, successorTurnId: string): Promise<void>;
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
  // Host-confirmed, actual typed payloads teach the native argument head.
  // Do not erase arbitrary user paths/content/experiences. Only explicit
  // external credential/transport fields and sensitive browser entry steps
  // are excluded from this particular autonomous argument-training channel.
  const transportKeys = new Set([
    "authorization", "proxyauthorization", "cookie", "setcookie", "headers",
    "password", "passwd", "apikey", "accesstoken", "refreshtoken", "bearertoken"
  ]);
  const clean = (value: unknown, depth = 0): unknown => {
    if (depth > 32) throw new Error("Typed host payload nesting exceeds the protocol budget.");
    if (Array.isArray(value)) return value.map((item) => clean(item, depth + 1));
    if (typeof value === "object" && value !== null) {
      const object = value as Record<string, unknown>;
      if (object.sensitive === true) throw new Error("Sensitive host entry is not an autonomous target.");
      return Object.fromEntries(Object.entries(object).flatMap(([key, item]) => {
        const normalized = key.replace(/[-_]/g, "").toLowerCase();
        if (transportKeys.has(normalized)) return [];
        return [[key, clean(item, depth + 1)]];
      }));
    }
    if (typeof value === "number" && !Number.isFinite(value)) throw new Error("Nonfinite host payload.");
    return value;
  };
  try {
    const payload = clean(action.arguments) as Record<string, unknown>;
    if (!action.toolId?.startsWith("mcp.")) {
      for (const key of ["assemblyIds", "conceptIds", "organic", "recursive", "localPackEnabled", "trainedPackAvailable", "neuralRoute"]) delete payload[key];
    }
    // Keep user-authored ordinary strings untouched; URL authority credentials
    // are transport material, not a parameter-learning target.
    if (typeof payload.url === "string") {
      const parsed = new URL(payload.url);
      if (parsed.username || parsed.password) return {};
    }
    return {
      typedActionArguments: payload,
      ...(action.inputSchema ? { inputSchema: action.inputSchema } : {})
    };
  } catch {
    return {};
  }
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
  actions: Map<string, ActionEvent>;
  neuralBoundary: Promise<void>;
  resolveNeuralBoundary(): void;
  neuralCallStarted: boolean;
  neuralEnded: boolean;
  steerSuccessor?: string;
  earlySteeredYield?: boolean;
  pendingDrain?: Promise<void>;
}

interface ActionOutcome {
  event: ActionEvent;
  output?: unknown;
}

async function waitForTurnNeuralBoundary(
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
    await Promise.race([predecessor.neuralBoundary, aborted]);
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
  private readonly inlineCancellations = new Map<string, { turn: ActiveTurn; event: ActionEvent }>();

  constructor(
    private readonly service: ActionChatService,
    private readonly tools: ActionToolExecutor,
    private readonly evolution: ActionEvolutionController
  ) {
    super();
    service.onInlineImaginationCancelled?.((ack) => {
      const key = `${ack.brainId}:${ack.turnId}:${ack.actionId}`;
      const pending = this.inlineCancellations.get(key);
      if (pending) this.settleInlineCancellation(key, pending);
    });
    service.onCodecRuntimeSetup?.((setup) => {
      const turn = this.activeTurns.get(setup.turnId);
      if (!turn || turn.brainId !== setup.brainId) return;
      const action = [...turn.actions.values()].find((event) => event.neuralActionId === setup.actionId);
      if (!action || action.cancellationRequested || !["proposed", "running"].includes(action.state)) return;
      action.statusLabel = setup.message;
      action.updatedAt = new Date().toISOString();
      this.publishAction(turn.neuralEnded ? undefined : turn, action);
    });
  }

  private settleInlineCancellation(key: string, pending: { turn: ActiveTurn; event: ActionEvent }): void {
    const { turn, event } = pending;
    this.inlineCancellations.delete(key);
    event.cancellationRequested = false;
    event.inlineGenerationOwned = false;
    event.state = "stopped";
    event.statusLabel = "Artifact cancelled · decoder cleanup acknowledged";
    event.updatedAt = new Date().toISOString();
    event.error = undefined;
    this.publishAction(this.activeTurns.has(turn.turnId) ? turn : undefined, event);
    void this.service.recordConversationActions?.(turn.brainId, [event]).catch(() => undefined);
  }

  async cancelInlineAction(brainId: string, turnId: string, actionEventId: string): Promise<{
    actionEvent: ActionEvent; acknowledged: boolean;
  }> {
    const turn = this.activeTurns.get(turnId);
    const event = turn?.actions.get(actionEventId);
    if (!turn || turn.brainId !== brainId || !event || !event.neuralActionId || !event.inlineGenerationOwned ||
        event.action.kind !== "imagine" || event.action.toolId !== "modality.imagine" ||
        !["proposed", "running"].includes(event.state) || !this.service.cancelInlineImagination) {
      throw new Error("This inline artifact is not owned by the requested active chat turn.");
    }
    const key = `${brainId}:${turnId}:${event.neuralActionId}`;
    event.cancellationRequested = true;
    event.statusLabel = "Cancelling artifact · waiting for decoder cleanup";
    event.updatedAt = new Date().toISOString();
    this.inlineCancellations.set(key, { turn, event });
    this.publishAction(turn, event);
    let result: { requested: boolean; acknowledged: boolean };
    try {
      result = await this.service.cancelInlineImagination(brainId, turnId, event.neuralActionId);
    } catch (error) {
      // A typed ownership refusal means no cancellation was admitted. A
      // transport timeout is different: its control may still be in flight.
      if (typeof error === "object" && error !== null && "code" in error &&
          (error.code === -32602 || error.code === -32600)) {
        this.inlineCancellations.delete(key);
        event.cancellationRequested = false;
        event.updatedAt = new Date().toISOString();
        this.publishAction(this.activeTurns.has(turnId) ? turn : undefined, event);
      }
      throw error;
    }
    if (!result.requested) {
      // A missing correlation is not a cancellation acknowledgement. Keep the
      // direction/receipt intact and never kill or cancel the containing turn.
      this.inlineCancellations.delete(key);
      event.cancellationRequested = false;
      throw new Error("The worker has not admitted this exact inline artifact cancellation.");
    }
    if (result.acknowledged && this.inlineCancellations.has(key)) {
      this.settleInlineCancellation(key, { turn, event });
    }
    return { actionEvent: JSON.parse(JSON.stringify(event)) as ActionEvent,
      acknowledged: event.state === "stopped" };
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
      | Omit<Extract<ChatStreamEvent, { type: "chat-reply-committed" }>, "id" | "brainId" | "turnId" | "sequence" | "createdAt">
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
    if (turn && !turn.earlySteeredYield) {
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
    // Actual host-confirmed typed payloads, including complete plans, train
    // the same native head. Credential/transport material alone is omitted;
    // ordinary user-authored paths and content are not silently redacted.
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
      const architectureChange = action.arguments.architectureChange === undefined
        ? undefined : normalizeCompatibleArchitectureMutation(action.arguments.architectureChange);
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
        !architectureChange && (typeof addExperts !== "number" || !Number.isInteger(addExperts) || addExperts < 1)
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
        ...(event.neuralActionId ? {
          provenance: { source: "same-native-cortex-action", neuralActionId: event.neuralActionId, actionEventId: event.id, chatTurnId: requestId }
        } : {}),
        ...(hasTypedSourceEdits ? { sourceEdits } : {}),
        architectureChange:
          candidateKind === "architecture"
            ? architectureChange ?? {
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
    if (event.cancellationRequested || (event as ActionEvent).state === "stopped") return execution.output;
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
    let resolveNeuralBoundary!: () => void;
    const neuralBoundary = new Promise<void>((resolve) => { resolveNeuralBoundary = resolve; });
    const turn: ActiveTurn = {
      brainId,
      turnId,
      controller,
      nextSequence: 0,
      turnMetadata,
      runtimePhase: "pending",
      settled,
      resolveSettled,
      actions: new Map(), neuralBoundary, resolveNeuralBoundary,
      neuralCallStarted: false, neuralEnded: false
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
      workerActionId ??= action.actionId;
      if (workerActionId && workerActions.has(workerActionId)) return workerActions.get(workerActionId);
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
      turn.actions.set(event.id, event);
      if (workerActionId) workerActions.set(workerActionId, event);
      if (action.kind === "imagine") latestImagination = event;
      this.publishAction(turn, event);
      const perform = async (): Promise<ActionOutcome> => {
        // An inline cancellation must not be restarted as the later trusted
        // host action after the containing text turn commits.
        if (event.cancellationRequested || event.state === "stopped") return { event };
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
          if (!event.cancellationRequested && (event as ActionEvent).state !== "stopped") {
            event.state = controller.signal.aborted ? "stopped" : "failed";
            event.error = error instanceof Error ? error.message : String(error);
            event.updatedAt = new Date().toISOString();
          }
        }
        this.publishAction(turn, event);
        return { event, output };
      };
      let execution: Promise<ActionOutcome> | undefined;
      const execute = (): Promise<ActionOutcome> => execution ??= perform();
      executions.push(execute);
      // External tool execution uses the trusted tool executor's grants and
      // durable pre-effect intent gate. Native brain mutations remain queued
      // on its write boundary; action-result learning below waits for commit.
      // Imagination retains its worker-owned progressive preview lifecycle.
      if (workerActionId && action.kind === "tool") void execute();
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
      if (neural.type === "inline-imagination-started") {
        const action = workerActions.get(neural.actionId);
        if (action) {
          action.inlineGenerationOwned = true;
          action.state = "running";
          action.statusLabel = "Preparing inline artifact";
          action.updatedAt = new Date().toISOString();
          this.publishAction(turn, action);
        }
        return;
      }
      const action =
        (neural.actionId ? workerActions.get(neural.actionId) : undefined) ??
        latestImagination;
      if (!action) return;
      if (action.cancellationRequested || action.state === "stopped") return;
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
        // Steer interrupts only the old generation at its next native safe
        // boundary. Its committed partial reply and independent actions remain.
        const queue: ChatQueueState = {
          position: 1,
          queuedBehind: {
            requestId: warmHandoffPredecessor.turnId,
            owner: "chat",
            label: "Warm neural steering boundary",
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
        if (!warmHandoffPredecessor.neuralEnded) {
          warmHandoffPredecessor.steerSuccessor = turnId;
          if (warmHandoffPredecessor.neuralCallStarted) {
            await this.service.steerChat?.(brainId, warmHandoffPredecessor.turnId, turnId);
          }
        }
        await waitForTurnNeuralBoundary(warmHandoffPredecessor, controller.signal);
        turn.runtimePhase = "pending";
        turn.queue = undefined;
      }
      controller.signal.throwIfAborted();
      if (turn.steerSuccessor) {
        throw new EngineRequestError("Chat yielded to a newer warm steering direction before dispatch.", -32801,
          { brainId, turnId, steered: true, zeroTokenYield: true, safeBoundary: true, warm: true });
      }
      this.publishStream(turn, {
        type: "chat-state",
        state: "started",
        turnMetadata
      });
      turn.neuralCallStarted = true;
      let result = await this.service.chat(
        brainId,
        input,
        controller.signal,
        consumeNeuralStream,
        turnId
      );
      for (const action of result.proposedActions ?? []) queueAction(action);

      // service.chat has returned only after its atomic neural/host turn save
      // (or exact durable receipt reconciliation). Optional artifacts and their
      // learning must not keep the completed text owned by generation controls.
      this.publishStream(turn, {
        type: "chat-reply-committed",
        humanMessage: result.humanMessage,
        brainMessage: result.brainMessage,
        pendingActions: executions.length
      });
      turn.neuralEnded = true;
      turn.resolveNeuralBoundary();

      // Streamed external tools may already be running after their independent
      // durable intent/permission gate. Await the same memoized execution once;
      // result learning and whole-brain auditing still use the committed turn.
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
      this.publishStream(turn, { type: "chat-state", state: result.generationEnd === "steered" ? "steered" :
        result.generationEnd === "native-stop" ? "stopped" : result.generationEnd === "no-reply" ? "no-reply" : "complete" });
      return result;
    } catch (error) {
      if (error instanceof EngineRequestError && [-32801, -32802].includes(error.code ?? 0) &&
          typeof error.data === "object" && error.data !== null &&
          ((error.data as Record<string, unknown>).steered === true ||
           (error.data as Record<string, unknown>).nativeStopped === true) &&
          (error.data as Record<string, unknown>).brainId === brainId &&
          (error.data as Record<string, unknown>).turnId === turnId) {
        turn.earlySteeredYield = true;
        turn.neuralEnded = true;
        this.publishStream(turn, { type: "chat-state", state: error.code === -32801 ? "steered" : "stopped" });
        turn.resolveNeuralBoundary();
        // Already-started permitted actions keep their own receipts. Their
        // outcome learning remains serialized, but never delays the successor.
        turn.pendingDrain = (async () => {
          for (const execution of executions) {
            const { event, output } = await execution();
            if (event.state !== "complete") continue;
            await this.learnConfirmedToolRoute(event, input, controller.signal).catch(() => undefined);
            if (this.service.learnStructuredExperience && !["talk", "ponder", "learn"].includes(event.action.kind)) {
              await this.service.learnStructuredExperience(brainId, {
                content: serializableToolExperience(event.action, output),
                name: `Interrupted chat ${event.action.kind} result`,
                sourceLabel: "chat visible action evidence", license: "Locally observed tool result"
              }, controller.signal).catch((learningError: unknown) => {
                event.error = `Action result learning failed: ${String(learningError)}`;
                event.updatedAt = new Date().toISOString();
                this.publishAction(undefined, event);
              });
            }
          }
          await this.service.recordConversationActions?.(brainId, events);
        })().catch(() => undefined).finally(() => {
          this.activeTurns.delete(turnId);
          turn.resolveSettled();
        });
        throw error;
      }
      const cancelled = controller.signal.aborted;
      const cancellationFailed = cancelled && turn.runtimeCancellationFailed === true;
      const failureMessage = error instanceof Error ? error.message : String(error);
      // A worker/process failure can arrive after an organic inline preview
      // marked its action Running but before the deferred tool claim begins.
      // Finalize every unfinished visible action so recovery never leaves a
      // ghost "In progress" card attached to the abandoned turn.
      for (const event of events) {
        if (event.cancellationRequested) continue;
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
      turn.neuralEnded = true;
      turn.resolveNeuralBoundary();
      if (!turn.pendingDrain) {
        this.activeTurns.delete(turnId);
        turn.resolveSettled();
      }
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
