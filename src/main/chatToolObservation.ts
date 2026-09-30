import { createHash } from "node:crypto";
import type { ActionEvent } from "../shared/types";

export interface ChatToolObservation {
  format: "omni-chat-tool-observation-v1";
  brainId: string;
  turnId: string;
  neuralActionId: string;
  actionEventId: string;
  executionId: string;
  toolId: string;
  action: string;
  completedAt: string;
  payloadJson: string;
  payloadSha256: string;
  observationId: string;
}

/** Admission to a reader-side inbox is not proof of neural consumption. */
export interface ChatToolObservationReceipt {
  brainId: string;
  turnId: string;
  observationId: string;
  accepted: boolean;
  duplicate?: boolean;
  reason?: string;
}

export class ChatObservationResourcePause extends Error {
  constructor(readonly estimatedAllocationBytes: number) {
    super("Live tool observation awaits admitted payload/transport RAM; the exact result remains eligible for durable learning.");
    this.name = "ChatObservationResourcePause";
  }
}

const hash = (value: string): string => createHash("sha256").update(value, "utf8").digest("hex");
const identity = (value: unknown): value is string => typeof value === "string" && value.length > 0 && !value.includes("\0");
const canonicalToolId = (value: string | undefined): string | undefined => value === "windows.files" ? "system.files" :
  value === "windows.powershell" ? "system.shell" : value;

function jsonEstimate(value: unknown, path = new Set<object>()): number {
  if (value === null || typeof value === "boolean") return 5;
  if (typeof value === "number") {
    if (!Number.isFinite(value)) throw new Error("Live tool observation contains a nonfinite number.");
    return 32;
  }
  if (typeof value === "string") return 2 + value.length * 6; // worst UTF-8/JSON escaping, without a full encoded copy
  if (typeof value !== "object") throw new Error("Live tool observation requires explicit JSON data.");
  if (path.has(value)) throw new Error("Live tool observation contains a cycle.");
  if (!Array.isArray(value) && ![Object.prototype, null].includes(Object.getPrototypeOf(value))) {
    throw new Error("Live tool observation cannot invoke implicit object conversion.");
  }
  if (Object.getOwnPropertySymbols(value).length) throw new Error("Live tool observation contains non-JSON keys.");
  path.add(value);
  let bytes = 2;
  try {
    if (Array.isArray(value)) {
      for (let index = 0; index < value.length; index += 1) {
        const field = Object.getOwnPropertyDescriptor(value, String(index));
        if (!field || !("value" in field)) throw new Error("Live tool observation contains a sparse/accessor array.");
        bytes += 1 + jsonEstimate(field.value, path);
      }
    } else {
      for (const key in value) {
        if (!Object.hasOwn(value, key)) continue;
        const field = Object.getOwnPropertyDescriptor(value, key)!;
        if (!("value" in field)) throw new Error("Live tool observation cannot evaluate accessors.");
        bytes += 4 + key.length * 6 + jsonEstimate(field.value, path);
      }
    }
  } finally { path.delete(value); }
  return bytes;
}

function canonicalJson(value: unknown): string {
  if (Array.isArray(value)) return `[${value.map(canonicalJson).join(",")}]`;
  if (typeof value === "object" && value !== null) {
    const object = value as Record<string, unknown>;
    return `{${Object.keys(object).sort().map((key) => `${JSON.stringify(key)}:${canonicalJson(object[key])}`).join(",")}}`;
  }
  return JSON.stringify(value);
}

/** Actual host receipt/output only. No answer snippets, prompts or new rows. */
export function createChatToolObservation(
  brainId: string, turnId: string, event: ActionEvent, output: unknown,
  admit: (estimatedAllocationBytes: number) => boolean
): ChatToolObservation | undefined {
  const execution = event.execution;
  if (event.brainId !== brainId || event.action.kind !== "tool" || event.state !== "complete" ||
      event.cancellationRequested || !event.neuralActionId || !/^[a-f0-9]{32}$/.test(event.neuralActionId) ||
      execution?.state !== "complete") return undefined;
  const toolId = event.action.toolId, action = event.action.action;
  const completedAt = execution.finishedAt ?? event.updatedAt;
  if (![brainId, turnId, event.id, execution.id, toolId, action, completedAt].every(identity) ||
      canonicalToolId(execution.toolId) !== canonicalToolId(toolId) || execution.action !== action || !Number.isFinite(Date.parse(completedAt)) ||
      execution.output !== output) throw new Error("Live tool observation lost its exact completed execution binding.");
  const data = output === undefined ? { outputPresent: false } : { outputPresent: true, output };
  // Both canonical payload and escaped RPC frame coexist briefly. Resource
  // admission precedes either copy; no fixed result-size product ceiling and
  // no silently shortened output are introduced.
  const estimated = 131072 + jsonEstimate(data) * 10;
  if (!Number.isSafeInteger(estimated) || !admit(estimated)) throw new ChatObservationResourcePause(estimated);
  const payloadJson = canonicalJson(data), payloadSha256 = hash(payloadJson);
  const observation = { format: "omni-chat-tool-observation-v1" as const, brainId, turnId,
    neuralActionId: event.neuralActionId, actionEventId: event.id, executionId: execution.id,
    toolId: toolId!, action: action!, completedAt, payloadJson, payloadSha256 };
  return { ...observation, observationId: hash([observation.format, brainId, turnId, event.neuralActionId,
    event.id, execution.id, toolId!, action!, completedAt, payloadSha256].join("\0")) };
}

export function observationReceipt(value: unknown, expected: ChatToolObservation): ChatToolObservationReceipt {
  if (!value || typeof value !== "object" || Array.isArray(value)) throw new Error("Invalid live observation receipt.");
  const receipt = value as Record<string, unknown>;
  if (receipt.brainId !== expected.brainId || receipt.turnId !== expected.turnId ||
      receipt.observationId !== expected.observationId || typeof receipt.accepted !== "boolean" ||
      (receipt.duplicate !== undefined && typeof receipt.duplicate !== "boolean") ||
      (receipt.reason !== undefined && typeof receipt.reason !== "string")) {
    throw new Error("Live observation receipt belongs to a different turn/result.");
  }
  return receipt as unknown as ChatToolObservationReceipt;
}
