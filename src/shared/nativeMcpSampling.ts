/** Reviewed current-task data, never persona policy or a source-file grant. */
export type NativeMcpSamplingContent = { type: "text"; text: string } |
  { type: "image" | "audio"; data: string; mimeType: string };

export const NATIVE_MCP_IMAGE_MIME_TYPES = ["image/png", "image/jpeg", "image/webp", "image/gif"] as const;
export const NATIVE_MCP_AUDIO_MIME_TYPES = ["audio/wav", "audio/x-wav", "audio/wave", "audio/flac", "audio/x-flac",
  "audio/ogg", "audio/mpeg", "audio/mp3", "audio/aac", "audio/mp4", "audio/x-m4a"] as const;

export interface NativeMcpSamplingMediaReceipt {
  messageIndex: number;
  blockIndex: number;
  kind: "image" | "audio";
  mimeType: string;
  sourceBytes: number;
  sourceSha256: string;
  decodedWindows: number;
  sourceFrames?: number;
  sourceSampleFrames?: number;
  sourceSampleRate?: number;
  modelSamples?: number;
  modelSampleRate?: number;
  wholeInputTraversed: true;
  nativeSensory: true;
}

export interface NativeMcpSamplingInput {
  brainId: string;
  requestId: string;
  originatingTurnId?: string;
  messages: Array<{ role: "user" | "assistant"; content: NativeMcpSamplingContent | NativeMcpSamplingContent[] }>;
  maxTokens: number;
  temperature?: number;
  stopSequences?: string[];
}

export interface ApprovedNativeMcpSamplingInput extends NativeMcpSamplingInput {
  inputSha256: string;
}

export interface NativeMcpSamplingResult {
  role: "assistant";
  content: { type: "text"; text: string };
  model: string;
  stopReason: "endTurn" | "maxTokens" | "stopSequence";
  inputSha256: string;
  requestId: string;
  brainId: string;
  privateWorkspace: true;
  parametersMutated: false;
  mediaInput?: NativeMcpSamplingMediaReceipt[];
}

export type NativeMcpSamplingControl = "mcp_sampling_enqueue" | "mcp_sampling_status" |
  "mcp_sampling_cancel" | "mcp_sampling_consume";

export interface NativeMcpSamplingStatus {
  brainId: string;
  requestId: string;
  inputSha256: string;
  state: "queued" | "running" | "complete" | "cancelled" | "failed";
  result?: NativeMcpSamplingResult;
  error?: { code: number; message: string; reason: string };
  accepted?: boolean;
  duplicate?: boolean;
  boundary?: "turn" | "serial";
  consumed?: boolean;
}
