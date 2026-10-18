import type { McpAppDescriptor, McpAppCsp } from "./mcpApps";
import type { McpToolSummary } from "./types";
import type { McpAppObservationReceipt, McpAppTaskData } from "./mcpAppTaskData";

export interface McpAppSource {
  brainId: string;
  serverId: string;
  serverLabel: string;
  endpoint: string;
}
export interface McpAppBlob { sha256: string; bytes: number; mimeType: string; }
export interface McpAppCapture {
  format: "omni-mcp-app-capture";
  version: 1;
  artifactId: string;
  source: McpAppSource;
  observedAt: string;
  resourceUri: string;
  tool: McpToolSummary;
  input: McpAppBlob;
  result: McpAppBlob;
  untrusted: true;
}
export interface McpAppResourceArtifact {
  format: "omni-mcp-app-resource";
  version: 1;
  artifactId: string;
  captureSha256: string;
  source: McpAppSource;
  observedAt: string;
  uri: string;
  content: McpAppBlob;
  metadata: Record<string, unknown>;
  declaredCsp: McpAppCsp;
  untrusted: true;
}
export interface McpAppObservationArtifact {
  format: "omni-mcp-app-observation";
  version: 1;
  id: string;
  artifactId: string;
  captureSha256: string;
  source: McpAppSource;
  observedAt: string;
  reviewId: string;
  method: McpAppTaskData["method"];
  content: McpAppBlob;
  streamId: string;
  ordinal: number;
  previousSha256: string;
  untrusted: true;
}
export type McpAppArtifactRecord = McpAppCapture | McpAppResourceArtifact | McpAppObservationArtifact;
export interface McpAppViewProjection extends McpAppDescriptor {
  artifactSha256: string;
  originalSource: McpAppSource;
  /** Presentation only; this never restores a session, token or tool approval. */
  savedArtifact: true;
  needsMappingReview: boolean;
}
export interface McpAppArtifactSummary {
  brainId: string;
  artifactId: string;
  captures: Array<{ sha256: string; source: McpAppSource; observedAt: string; resourceUri: string; remoteToolName: string }>;
  resources: Array<{ sha256: string; captureSha256: string; source: McpAppSource; observedAt: string; content: McpAppBlob }>;
  selectedCapture?: string;
  selectedResource?: string;
  needsMappingReview: boolean;
  observationQueue?: { pending: number; dormant: number; superseded: number; admitted: number; latest?: McpAppObservationReceipt };
}
