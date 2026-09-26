const MAX_UI_MESSAGE_CHARACTERS = 320;

function isStackFrame(line: string): boolean {
  return (
    /^\s*at\s+/i.test(line) ||
    /^\s*File\s+"[^"]+",\s+line\s+\d+/i.test(line) ||
    /^\s*Traceback\s+\(most recent call last\):?\s*$/i.test(line) ||
    /^\s*(?:Caused by|During handling of the above exception)/i.test(line)
  );
}

/**
 * Renderer diagnostics should explain the failure without turning a compact
 * toast, job card, or status row into a raw stack-trace console. The complete
 * diagnostic remains available in the runtime logs and operational trace.
 */
export function conciseUiMessage(
  value: unknown,
  fallback = "Something went wrong."
): string {
  const raw = value instanceof Error
    ? value.message || value.stack || ""
    : typeof value === "string"
      ? value
      : "";
  const normalized = raw
    .replaceAll("\0", "")
    .replace(/^Error invoking remote method ['"][^'"]+['"]:\s*/i, "")
    .replace(/^Worker traceback:\s*/i, "")
    .trim();
  const lines = normalized
    .split(/\r?\n/)
    .map((line) => line.trim())
    .filter(Boolean);
  const containsTrace = lines.some((line) =>
    isStackFrame(line) || /(?:traceback|worker traceback)/i.test(line)
  );
  const meaningfulLines = lines.filter((line) =>
    !isStackFrame(line) && !/(?:traceback|worker traceback)/i.test(line)
  );
  const selected = containsTrace
    ? meaningfulLines.at(-1) ?? meaningfulLines[0] ?? ""
    : meaningfulLines[0] ?? "";
  const withoutType = selected
    .replace(/^Error invoking remote method ['"][^'"]+['"]:\s*/i, "")
    .replace(/^Worker traceback:\s*/i, "")
    .replace(/^(?:[A-Za-z_$][\w.$]*(?:Error|Exception)|Error):\s*/u, "")
    .replace(/\s+at\s+(?:[\w.$<>]+\s+)?\(?[^\s()]+:\d+:\d+\)?.*$/u, "")
    .replace(/\s+/g, " ")
    .trim();
  const message = withoutType || fallback;
  return message.length > MAX_UI_MESSAGE_CHARACTERS
    ? `${message.slice(0, MAX_UI_MESSAGE_CHARACTERS - 1).trimEnd()}…`
    : message;
}

const RECOVERY_POINT_CANCELLED_COPY =
  "Recovery point creation cancelled; current brain unchanged.";

/** Expected checkpoint cancellation is control flow, not an IPC diagnostic. */
export function recoveryPointCreationErrorMessage(value: unknown): string {
  const message = conciseUiMessage(
    value,
    "The recovery point could not be created."
  );
  return /^(?:Worker request\s+)?["']?checkpoint["']?.*cancelled\.?$/i.test(
    message
  ) || /^Worker request\s+["']checkpoint["']\s+was canceled\.?$/i.test(message)
    ? RECOVERY_POINT_CANCELLED_COPY
    : message;
}

export function livePerceptionErrorMessage(value: unknown): string {
  const message = conciseUiMessage(
    value,
    "Live Perception could not complete the request."
  );
  return /^Live observation maxInFlight exceeds the current resource envelope(?:\s*\(\d+\))?\.?$/i.test(
    message
  )
    ? "Device resources changed before Live Perception started. Try Start again; Auto will adapt safely."
    : message;
}

/** Omni's stable tokenizer boundary is UTF-8 bytes, so this is exact for the
 * draft itself instead of a word-based estimate. */
export function utf8DraftTokenCount(value: string): number {
  return new TextEncoder().encode(value).length;
}

export function isPristineBrainSummary(summary: {
  concepts: number;
  synapses: number;
  generation: number;
  neuralUpdates?: number;
  inferenceCount?: number;
  trainingSources?: number;
  substrateTotals?: { neurons: number; assemblies: number; synapses: number };
}): boolean {
  return (
    (summary.substrateTotals?.assemblies ?? summary.concepts) === 0 &&
    (summary.substrateTotals?.synapses ?? summary.synapses) === 0 &&
    summary.generation === 0 &&
    (summary.neuralUpdates ?? 0) === 0 &&
    (summary.inferenceCount ?? 0) === 0 &&
    (summary.trainingSources ?? 0) === 0
  );
}

type ConnectionCountSource =
  | "queried-substrate"
  | "persisted-substrate"
  | "compatibility-mirror";

export interface ConnectionCountPresentation {
  /** Current unique edges, never the number of times those edges were updated. */
  uniqueConnections: number;
  uniqueSource: ConnectionCountSource;
  /** Historical activity; one edge may contribute many updates. */
  cumulativeUpdates: number;
  /** True only when activity exists but no authoritative topology was readable. */
  topologyPending: boolean;
}

function uiCount(value: number | undefined): number | undefined {
  return typeof value === "number" && Number.isFinite(value) && value >= 0
    ? Math.trunc(value)
    : undefined;
}

/**
 * Keep the two connection quantities shown by the renderer semantically
 * separate. A queried page is freshest, the checksum-validated persisted
 * overview is the durable fallback, and the bounded BrainDocument map is only
 * a compatibility fallback. Cumulative update activity is never substituted
 * for a unique edge count.
 */
export function connectionCountPresentation(input: {
  queriedConnections?: number;
  persistedConnections?: number;
  mirroredConnections?: number;
  cumulativeUpdates?: number;
}): ConnectionCountPresentation {
  const queried = uiCount(input.queriedConnections);
  const persisted = uiCount(input.persistedConnections);
  const mirrored = uiCount(input.mirroredConnections) ?? 0;
  const uniqueConnections = queried ?? persisted ?? mirrored;
  const uniqueSource: ConnectionCountSource = queried !== undefined
    ? "queried-substrate"
    : persisted !== undefined
      ? "persisted-substrate"
      : "compatibility-mirror";
  const cumulativeUpdates = uiCount(input.cumulativeUpdates) ?? 0;
  return {
    uniqueConnections,
    uniqueSource,
    cumulativeUpdates,
    topologyPending:
      uniqueSource === "compatibility-mirror" &&
      uniqueConnections === 0 &&
      cumulativeUpdates > 0
  };
}

/**
 * Return a truthful represented-node count for the Brain Map's summary node.
 * A nonzero plasticity counter proves activity even while the compatibility
 * mirror and an authoritative first page are briefly empty.
 */
export function substrateOverviewCount(input: {
  neurons?: number;
  assemblies?: number;
  mirroredEndpointCount?: number;
  plasticityEvents: number;
}): number {
  const measured = Math.max(
    0,
    Math.trunc(input.neurons ?? 0),
    Math.trunc(input.assemblies ?? 0),
    Math.trunc(input.mirroredEndpointCount ?? 0)
  );
  return measured || (input.plasticityEvents > 0 ? 1 : 0);
}
