import type {
  ActionKind,
  ActionSource,
  StructuredAction
} from "../shared/types";

const ACTION_KINDS = new Set<ActionKind>([
  "talk",
  "tool",
  "imagine",
  "agent",
  "ponder",
  "learn",
  "evolve",
  "stop"
]);
const TOOL_ID = /^[a-z][a-z0-9.-]{1,79}$/;
const TOOL_ACTION = /^[a-z][a-z0-9_-]{0,79}$/;

function record(value: unknown): Record<string, unknown> | undefined {
  return typeof value === "object" && value !== null && !Array.isArray(value)
    ? (value as Record<string, unknown>)
    : undefined;
}

function argumentsObject(value: unknown): Record<string, unknown> {
  return record(value) ?? {};
}

function boundedText(value: unknown, maximum: number): string {
  return typeof value === "string"
    ? value.replace(/\0/g, "").trim().slice(0, maximum)
    : "";
}

function naturalToolAction(
  kind: "imagine" | "agent" | "evolve",
  source: ActionSource,
  argumentsValue: Record<string, unknown>
): StructuredAction {
  if (kind === "imagine") {
    return {
      kind,
      source,
      toolId: "modality.imagine",
      action: "generate",
      arguments: argumentsValue
    };
  }
  if (kind === "agent") {
    return {
      kind,
      source,
      toolId: "agent.fork",
      action: "start",
      arguments: argumentsValue
    };
  }
  return {
    kind,
    source,
    toolId: "source.self-modify",
    action: "propose",
    arguments: argumentsValue
  };
}

export function normalizeStructuredAction(
  value: unknown,
  source: ActionSource
): StructuredAction | undefined {
  const candidate = record(value);
  if (!candidate) return undefined;
  const rawKind = boundedText(candidate.kind, 32).toLocaleLowerCase();
  const rawToolId = boundedText(candidate.toolId ?? candidate.tool_id, 80).toLocaleLowerCase();
  const rawAction = boundedText(candidate.action, 80).toLocaleLowerCase();
  const inferredKind =
    rawKind ||
    (rawToolId === "modality.imagine"
      ? "imagine"
      : rawToolId === "agent.fork"
        ? "agent"
        : rawToolId === "source.self-modify"
          ? "evolve"
          : rawToolId
            ? "tool"
            : "");
  if (!ACTION_KINDS.has(inferredKind as ActionKind)) return undefined;
  const kind = inferredKind as ActionKind;
  const confidence =
    typeof candidate.confidence === "number" && Number.isFinite(candidate.confidence)
      ? Math.max(0, Math.min(1, candidate.confidence))
      : undefined;

  if (["talk", "ponder", "learn", "stop"].includes(kind)) {
    return {
      kind,
      source,
      arguments: argumentsObject(candidate.arguments),
      confidence
    };
  }

  if (["imagine", "agent", "evolve"].includes(kind) && !rawToolId) {
    return {
      ...naturalToolAction(
        kind as "imagine" | "agent" | "evolve",
        source,
        argumentsObject(candidate.arguments)
      ),
      confidence
    };
  }
  if (!TOOL_ID.test(rawToolId) || !TOOL_ACTION.test(rawAction)) return undefined;
  return {
    kind,
    source,
    toolId: rawToolId,
    action: rawAction,
    arguments: argumentsObject(candidate.arguments),
    confidence
  };
}

/**
 * Accept only the worker's dedicated typed action channel. Response prose is
 * deliberately ignored so an organic action cannot be manufactured by a
 * hidden prompt, slash command, or tagged-text convention.
 */
export function parseModelActions(
  _text: string,
  rawActions?: unknown
): StructuredAction[] {
  const values: unknown[] = [];
  if (Array.isArray(rawActions)) values.push(...rawActions.slice(0, 8));
  else if (rawActions !== undefined) values.push(rawActions);
  const actions = values
    .map((value) => normalizeStructuredAction(value, "brain"))
    .filter((value): value is StructuredAction => Boolean(value));
  const seen = new Set<string>();
  return actions.filter((action) => {
    const fingerprint = JSON.stringify([
      action.kind,
      action.toolId,
      action.action,
      action.arguments
    ]);
    if (seen.has(fingerprint)) return false;
    seen.add(fingerprint);
    return true;
  });
}
