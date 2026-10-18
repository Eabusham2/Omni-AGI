import type { NativeMcpSamplingInput, NativeMcpSamplingResult } from "./nativeMcpSampling";
import type { McpAppCsp } from "./mcpApps";
import type { McpAppDataReview } from "./mcpAppTaskData";

export interface McpRequestOrigin {
  brainId: string;
  serverId: string;
  serverLabel: string;
  endpoint: string;
  connectionEpoch: number;
  originatingTurnId?: string;
  actionEventId?: string;
}

export interface McpFormProperty {
  type: "string" | "number" | "integer" | "boolean" | "array";
  title?: string;
  description?: string;
  default?: string | number | boolean | string[];
  minLength?: number;
  maxLength?: number;
  pattern?: string;
  format?: "email" | "uri" | "date" | "date-time";
  minimum?: number;
  maximum?: number;
  enum?: string[];
  oneOf?: Array<{ const: string; title?: string }>;
  minItems?: number;
  maxItems?: number;
  items?: { type?: "string"; enum?: string[]; anyOf?: Array<{ const: string; title?: string }> };
}
export interface McpFormSchema {
  type: "object";
  properties: Record<string, McpFormProperty>;
  required?: string[];
}
export type McpFormValues = Record<string, string | number | boolean | string[]>;
export type McpElicitationRequest =
  | { mode: "form"; message: string; requestedSchema: McpFormSchema }
  | { mode: "url"; message: string; url: string; elicitationId: string };
export type McpElicitationResult = { action: "accept"; content?: McpFormValues } | { action: "decline" | "cancel" };
export interface McpSamplingMediaReview {
  messageIndex: number;
  blockIndex: number;
  kind: "image" | "audio";
  mimeType: string;
  decodedBytes: number;
  /** Hash of the exact canonical base64, without decoding untrusted media. */
  encodedSha256: string;
}
export interface McpAppPermissionReview {
  category: "tool" | "network" | "link" | "roots";
  message: string;
  toolId?: string;
  arguments?: Record<string, unknown>;
  csp?: McpAppCsp;
  url?: string;
}

/** No host capabilities are granted by these reviews; each action retains its own gate. */
export type McpInteractionRecord = {
  format: "omni-mcp-interaction";
  version: 1;
  id: string;
  origin: McpRequestOrigin;
  createdAt: string;
  updatedAt: string;
  status: "pending" | "accepted" | "declined" | "cancelled";
  reason?: string;
  submittedContentSha256?: string;
} & (
  | { kind: "form"; request: Extract<McpElicitationRequest, { mode: "form" }> }
  | { kind: "url"; request: Extract<McpElicitationRequest, { mode: "url" }>; outOfBandStatus?: "unknown" | "complete";
      navigation?: { state: "dispatching" | "opened" | "failed"; at: string } }
  | { kind: "sampling-input"; request: NativeMcpSamplingInput; systemPromptOmitted?: string; modelPreferences?: Record<string, unknown>; media?: McpSamplingMediaReview[] }
  | { kind: "sampling-output"; request: NativeMcpSamplingResult }
  | { kind: "app-permission"; request: McpAppPermissionReview }
  | { kind: "app-data"; request: McpAppDataReview }
);

export interface McpInteractionResponse {
  brainId: string;
  id: string;
  action: "accept" | "decline" | "cancel";
  values?: McpFormValues;
  /** Explicitly edited request only; host binds identities and hashes it again. */
  sampling?: Pick<NativeMcpSamplingInput, "messages" | "maxTokens" | "temperature" | "stopSequences">;
  editedText?: string;
}

export interface McpInteractionEvent { record: McpInteractionRecord; }
export interface McpInteractionsApi {
  list(brainId: string): Promise<McpInteractionRecord[]>;
  respond(response: McpInteractionResponse): Promise<McpInteractionRecord>;
  onEvent(listener: (event: McpInteractionEvent) => void): () => void;
}
