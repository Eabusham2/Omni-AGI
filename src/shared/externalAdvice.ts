export type ExternalAdviceProvider =
  | "openai"
  | "anthropic"
  | "gemini"
  | "openrouter"
  | "ollama"
  | "openai-compatible-local";

export type ExternalAdviceChannel = "assistant" | "reviewer";

export interface ExternalAdviceContextSelection {
  includeTask: boolean;
  includeActivity: boolean;
  /** Exact visible character budget for host-selected context. */
  maxCharacters: number;
}

export interface ExternalAdviceChannelConfiguration {
  /** Portable preference only. Current installation authority must also allow the channel. */
  enabled: boolean;
  provider: ExternalAdviceProvider;
  /** Manual provider model ID; the advice path never substitutes another model. */
  model: string;
  /** Accepted only for explicit loopback providers; Ollama defaults to 127.0.0.1:11434/v1. */
  endpoint?: string;
  maxOutputTokens: number;
  context: ExternalAdviceContextSelection;
}

export interface ExternalAdviceSettings {
  format: "omni-external-advice-settings";
  version: 1;
  brainId: string;
  revision: number;
  updatedAt: string;
  assistant: ExternalAdviceChannelConfiguration;
  reviewer: ExternalAdviceChannelConfiguration;
  /** Current installation projection. It is never written to the portable workspace. */
  deviceAuthority?: { assistant: boolean; reviewer: boolean };
  deviceAuthorityRequired: true;
  credentialsStoredInBrain: false;
}

export interface ExternalAdviceConfigureInput {
  expectedRevision: number;
  channel: ExternalAdviceChannel;
  configuration: ExternalAdviceChannelConfiguration;
}

export interface ExternalAdviceContextItem {
  id: string;
  kind: "task" | "activity";
  label: string;
  text: string;
  contentSha256: string;
  /** Original source hash retained when a configured credential was redacted before provider transmission. */
  sourceContentSha256?: string;
  createdAt?: string;
}

/** Returned by the trusted host selector, never assembled by the provider client. */
export interface ExternalAdviceSelectedContext {
  format: "omni-external-advice-selected-context";
  version: 1;
  brainId: string;
  channel: ExternalAdviceChannel;
  selection: ExternalAdviceContextSelection;
  items: ExternalAdviceContextItem[];
  includedCharacters: number;
  omittedItems: number;
  truncated: boolean;
  redactedCredentials?: number;
}

export interface ExternalAdvicePreviewInput {
  channel: ExternalAdviceChannel;
  prompt: string;
}

export interface ExternalAdvicePreview {
  format: "omni-external-advice-preview";
  version: 1;
  brainId: string;
  channel: ExternalAdviceChannel;
  provider: ExternalAdviceProvider;
  model: string;
  endpoint?: string;
  prompt: string;
  /** Exact visible text sent to the configured provider. */
  requestText: string;
  context: ExternalAdviceSelectedContext;
  settingsRevision: number;
  authorityRevision: number;
  authorityOwnerEpoch: string;
  contextSha256: string;
  sha256: string;
  maxOutputTokens: number;
  advisoryOnly: true;
  automaticLearning: false;
}

export interface ExternalAdviceStartInput extends ExternalAdvicePreviewInput {
  requestId: string;
  expectedSettingsRevision: number;
  previewSha256: string;
}

export type ExternalAdviceRequestState =
  | "queued"
  | "streaming"
  | "complete"
  | "cancelled"
  | "failed";

export interface ExternalAdviceRequest {
  format: "omni-external-advice-request";
  version: 1;
  id: string;
  brainId: string;
  channel: ExternalAdviceChannel;
  provider: ExternalAdviceProvider;
  model: string;
  endpoint?: string;
  state: ExternalAdviceRequestState;
  prompt: string;
  requestText: string;
  context: ExternalAdviceSelectedContext;
  response: string;
  responseCredentialRedactions?: number;
  error?: string;
  createdAt: string;
  updatedAt: string;
  settingsRevision: number;
  authorityRevision: number;
  authorityOwnerEpoch: string;
  contextSha256: string;
  previewSha256: string;
  requestSha256: string;
  maxOutputTokens: number;
  providerRequestId?: string;
  finishReason?: string;
  advisoryOnly: true;
  automaticLearning: false;
  hostActionAuthority: false;
  credentialsStoredInBrain: false;
  inheritedFrom?: { brainId: string; requestId: string; sha256: string };
}

export interface ExternalAdviceHistoryQuery {
  after?: string;
  limit?: number;
  channel?: ExternalAdviceChannel;
}

export interface ExternalAdviceHistoryPage {
  entries: ExternalAdviceRequest[];
  total: number;
  nextAfter?: string;
  /** Only requests backed by a live controller in this process. */
  activeRequestIds?: string[];
  retirementBlocked?: { requestIds: string[]; error: string };
}

export interface ExternalAdviceEvent {
  brainId: string;
  request: ExternalAdviceRequest;
  /** True only while this main-process service owns a live request controller. */
  active?: boolean;
  /** Streaming UI may be ahead of the last resource-admitted portable checkpoint. */
  persistence?: { receivedCharacters: number; persistedCharacters: number };
}

export interface ExternalAdviceCredentialStatus {
  provider: ExternalAdviceProvider;
  configured: boolean;
  required: boolean;
  persistence: "encrypted" | "session" | "none";
  revision: number;
}

export interface ExternalAdviceCredentialSaveInput {
  provider: ExternalAdviceProvider;
  credential: string;
}

export interface ExternalAdviceCancellation {
  brainId: string;
  requestId: string;
  cancelled: boolean;
}

export interface ExternalAdviceReviewReceipt {
  request: ExternalAdviceRequest;
  text: string;
  advisoryOnly: true;
  hostActionAuthority: false;
}
export interface ExternalAdviceModelDiscoveryInput {
  channel: ExternalAdviceChannel;
  requestId: string;
  expectedSettingsRevision: number;
  expectedCredentialRevision: number;
}
export interface ExternalAdviceModelDiscoveryResult extends ApiTeacherModelDiscoveryResult {
  brainId: string;
  channel: ExternalAdviceChannel;
  settingsRevision: number;
  credentialRevision: number;
  /** Exact saved endpoint, present only for Ollama local. */
  endpoint?: string;
}
export interface ExternalAdviceModelDiscoveryCancellation {
  brainId: string;
  channel: ExternalAdviceChannel;
  requestId: string;
  cancelled: boolean;
}

export interface ExternalAdviceApi {
  settings(brainId: string): Promise<ExternalAdviceSettings>;
  configure(brainId: string, input: ExternalAdviceConfigureInput): Promise<ExternalAdviceSettings>;
  preview(brainId: string, input: ExternalAdvicePreviewInput): Promise<ExternalAdvicePreview>;
  /** Public panel requests are assistant-only; reviewer calls use the trusted host service method. */
  start(brainId: string, input: ExternalAdviceStartInput): Promise<ExternalAdviceRequest>;
  cancel(brainId: string, requestId: string): Promise<ExternalAdviceCancellation>;
  history(brainId: string, query?: ExternalAdviceHistoryQuery): Promise<ExternalAdviceHistoryPage>;
  credentialStatus(): Promise<ExternalAdviceCredentialStatus[]>;
  saveCredential(input: ExternalAdviceCredentialSaveInput): Promise<ExternalAdviceCredentialStatus[]>;
  removeCredential(provider: ExternalAdviceProvider): Promise<ExternalAdviceCredentialStatus[]>;
  discoverModels(brainId: string, input: ExternalAdviceModelDiscoveryInput): Promise<ExternalAdviceModelDiscoveryResult>;
  cancelModelDiscovery(brainId: string, input: Pick<ExternalAdviceModelDiscoveryInput, "channel" | "requestId">): Promise<ExternalAdviceModelDiscoveryCancellation>;
  onEvent(listener: (event: ExternalAdviceEvent) => void): () => void;
}
import type { ApiTeacherModelDiscoveryResult } from "./types";
