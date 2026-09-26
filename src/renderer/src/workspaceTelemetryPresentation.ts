import type { WorkspaceSnapshot } from "../../shared/types";

export type CurrentExperienceState = "idle" | "responding" | "saving";

export interface WorkspaceTelemetryDelta {
  fromAt: string;
  toAt: string;
  temporaryTokens: number;
  recentTokens: number;
  evictions: number;
  connectionUpdates: number;
  parameterSteps: number;
}

export interface WorkspaceTelemetryPresentation {
  contextText: string;
  contextAriaLabel: string;
  experienceText: string;
  experienceAriaLabel: string;
  backgroundText: string;
  backgroundAriaLabel: string;
  corticalWeightsText: string;
  backgroundErrorText?: string;
  deltaText: string;
  measuredText: string;
  freshAttentionText?: string;
}

function integer(value: unknown): number {
  return typeof value === "number" && Number.isSafeInteger(value) && value >= 0
    ? value
    : 0;
}

function signed(value: number): string {
  return `${value >= 0 ? "+" : ""}${value.toLocaleString()}`;
}

function measuredTime(value: string): string {
  const parsed = Date.parse(value);
  if (!Number.isFinite(parsed)) return "time unavailable";
  return new Date(parsed).toLocaleTimeString([], {
    hour: "numeric",
    minute: "2-digit",
    second: "2-digit"
  });
}

function visibleError(value: unknown): string | undefined {
  if (typeof value !== "string") return undefined;
  const normalized = value.replace(/\s+/g, " ").trim();
  return normalized ? normalized.slice(0, 240) : undefined;
}

/** Deltas compare two committed worker snapshots, never renderer estimates. */
export function workspaceTelemetryDelta(
  previous: WorkspaceSnapshot | null | undefined,
  next: WorkspaceSnapshot
): WorkspaceTelemetryDelta | undefined {
  if (!previous || previous.brainId !== next.brainId) return undefined;
  const previousLearning = previous.learning?.fastNeuralMemory;
  const nextLearning = next.learning?.fastNeuralMemory;
  return {
    fromAt: previous.learning?.measuredAt ?? previous.contextWindow.updatedAt,
    toAt: next.learning?.measuredAt ?? next.contextWindow.updatedAt,
    temporaryTokens:
      next.contextWindow.tokenCount - previous.contextWindow.tokenCount,
    recentTokens:
      integer(next.contextWindow.recentTokenCount) -
      integer(previous.contextWindow.recentTokenCount),
    evictions:
      integer(next.contextWindow.evictions) -
      integer(previous.contextWindow.evictions),
    connectionUpdates:
      integer(nextLearning?.connectionUpdatesTotal) -
      integer(previousLearning?.connectionUpdatesTotal),
    parameterSteps:
      integer(nextLearning?.parameterStepsTotal) -
      integer(previousLearning?.parameterStepsTotal)
  };
}

export function workspaceTelemetryPresentation(input: {
  workspace: WorkspaceSnapshot;
  delta?: WorkspaceTelemetryDelta;
  currentExperience: CurrentExperienceState;
}): WorkspaceTelemetryPresentation {
  const { workspace, delta, currentExperience } = input;
  const context = workspace.contextWindow;
  const recent = integer(context.recentTokenCount);
  const evictions = integer(context.evictions);
  const capacity = Math.max(1, context.capacityTokens);
  const occupancy = Math.min(100, (context.tokenCount / capacity) * 100);
  const measuredAt = workspace.learning?.measuredAt ?? context.updatedAt;
  const fast = workspace.learning?.fastNeuralMemory;
  const background = workspace.learning?.backgroundParameters;
  const contextText =
    `${context.tokenCount.toLocaleString()} / ${context.capacityTokens.toLocaleString()} temporary tokens` +
    ` · ${recent.toLocaleString()} recent · ${evictions.toLocaleString()} evicted`;
  const deltaText = delta
    ? ` Since the prior committed measurement: ${signed(delta.temporaryTokens)} temporary, ` +
      `${signed(delta.recentTokens)} recent, ${signed(delta.evictions)} evictions, ` +
      `${signed(delta.connectionUpdates)} connection updates, and ` +
      `${signed(delta.parameterSteps)} optimizer steps across all training.`
    : "";
  const visibleDeltaText = delta
    ? `${signed(delta.temporaryTokens)} temporary tokens · ` +
      `${signed(delta.evictions)} evictions · ` +
      `${signed(delta.connectionUpdates)} connection updates · ` +
      `${signed(delta.parameterSteps)} optimizer steps (all training)`
    : "Waiting for the next committed measurement";
  const contextAriaLabel =
    `Temporary context uses ${context.tokenCount.toLocaleString()} of ` +
    `${context.capacityTokens.toLocaleString()} tokens, ${occupancy.toFixed(1)} percent. ` +
    `${recent.toLocaleString()} tokens are recent and ${evictions.toLocaleString()} ` +
    `capacity or decay evictions have been measured. Updated ${measuredTime(context.updatedAt)}.` +
    deltaText;

  const experienceText = currentExperience === "responding"
    ? "Response in progress · fast synapse receipt pending"
    : currentExperience === "saving"
      ? "Response complete · saving fast synapses and episodes"
      : fast?.state === "learned" && fast.safelyStored
        ? "Latest turn saved into fast synapses and episodes"
        : "No completed experience measured yet";
  const experienceAriaLabel = currentExperience === "idle" && fast?.committedAt
    ? `${experienceText}. Committed ${measuredTime(fast.committedAt)}. ` +
      `${fast.connectionUpdatesTotal.toLocaleString()} cumulative local connection updates. ` +
      `${fast.parameterStepsTotal.toLocaleString()} cumulative optimizer steps across all training; this count does not prove a dense-weight update for this turn.`
    : experienceText;

  const pending = integer(background?.pending);
  const completed = integer(background?.completed);
  const backgroundError = visibleError(background?.lastError) ??
    visibleError(background?.error);
  const backgroundErrorText = backgroundError
    ? `Last slow-replay error: ${backgroundError}`
    : undefined;
  const phase = background?.state === "running"
    ? " · running"
    : background?.state === "pausing"
      ? " · stopping current update"
      : background?.state === "paused"
        ? background.pauseReason === "launch-override"
          ? " · suspended for this app launch; pending updates retained"
          : " · paused; pending updates retained"
    : background?.state === "pending" || background?.state === "failed"
      ? " · waiting to retry"
      : "";
  const moduleWeightStatus = background?.corticalParametersUpdated === true && completed > 0
    ? " · module weights changed"
    : background?.corticalParametersUpdated === false && completed > 0
      ? " · module weights unchanged"
      : " · module weights unverified";
  const backgroundText = !background
    ? "Slow optimizer replay: counts not measured"
    : `Slow optimizer replay: ${pending.toLocaleString()} pending · ${completed.toLocaleString()} completed${phase}${moduleWeightStatus}` +
      (backgroundError ? ` · last error: ${backgroundError}` : "");
  const corticalWeightsText = background?.corticalParametersUpdated === true && completed > 0
    ? "Trainable module weights changed in the last completed slow replay"
    : background?.corticalParametersUpdated === false && completed > 0
      ? "No trainable module weight change in the last completed slow replay"
      : "Trainable module weight change not independently verified";
  const backgroundAriaLabel = background
    ? `${backgroundText}. Updated ${measuredTime(background.updatedAt)}.` +
      (background.completedAt
        ? ` Last completion ${measuredTime(background.completedAt)}.`
        : "") + ` ${corticalWeightsText}.` +
      (background.parameterChanged !== undefined && background.corticalParametersUpdated === undefined
        ? " The composite parameter checksum includes associative memory and is not proof of module weight training."
        : "")
    : backgroundText;

  const boundary = workspace.freshAttentionBoundary;
  const freshAttentionText =
    boundary && context.tokenCount === 0 && recent === 0
      ? `Fresh attention reached 0 temporary tokens · ` +
        `${boundary.messagesPreserved.toLocaleString()} messages and learned neural state preserved`
      : undefined;
  return {
    contextText,
    contextAriaLabel,
    experienceText,
    experienceAriaLabel,
    backgroundText,
    backgroundAriaLabel,
    corticalWeightsText,
    ...(backgroundErrorText ? { backgroundErrorText } : {}),
    deltaText: visibleDeltaText,
    measuredText: `Neural state measured ${measuredTime(measuredAt)}`,
    ...(freshAttentionText ? { freshAttentionText } : {})
  };
}
