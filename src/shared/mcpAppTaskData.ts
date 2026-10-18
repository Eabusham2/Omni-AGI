import type { McpContentBlock } from "./mcpCapabilities";
import type { McpAppDescriptor } from "./mcpApps";
import type { NativeAppObservationResult } from "./nativeAppObservation";

/** Remote content remains data even when its text contains imperative prose. */
export type McpAppTaskData =
  | { method: "ui/message"; params: { role: "user"; content: McpContentBlock | McpContentBlock[]; _meta?: Record<string, unknown> } }
  | { method: "ui/update-model-context"; params: { content?: McpContentBlock[]; structuredContent?: Record<string, unknown>; _meta?: Record<string, unknown> } };
export interface McpAppDataReview {
  message: string;
  view: McpAppDescriptor;
  captureSha256: string;
  dataSha256: string;
  data: McpAppTaskData;
  untrustedTaskData: true;
  hiddenPromptUsed: false;
}
export interface McpAppObservationReceipt {
  id: string;
  artifactId: string;
  recordSha256: string;
  streamId: string;
  ordinal: number;
  state: "pending-native-admission" | "processing-native-admission" | "superseded" | "dormant" | "admitted";
  observedAt: string;
  method: McpAppTaskData["method"];
  /** This is a host queue receipt, never proof of neural learning. */
  neuralAdmission: "not-yet-admitted" | "admitted" | "advanced-without-observation";
  nativeReceipt?: NativeAppObservationResult;
}
