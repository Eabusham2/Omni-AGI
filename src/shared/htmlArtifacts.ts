export const HTML_ARTIFACT_CONTENT_FORMAT = "omni-chat-html-artifact-content" as const;
export const HTML_ARTIFACT_CONTENT_VERSION = 1 as const;
export const HTML_ARTIFACT_RESULT_FORMAT = "omni-chat-html-artifact-result" as const;
export const HTML_ARTIFACT_SCHEME = "omni-html-artifact" as const;
export const HTML_ARTIFACT_MESSAGE_FORMAT = "omni-html-artifact-message" as const;

export interface HtmlArtifactToolDeclaration {
  toolId: string;
  actions: string[];
}

export interface HtmlArtifactContent {
  format: typeof HTML_ARTIFACT_CONTENT_FORMAT;
  version: typeof HTML_ARTIFACT_CONTENT_VERSION;
  title: string;
  html: string;
  css?: string;
  script?: string;
  tools: HtmlArtifactToolDeclaration[];
  height: number;
}

/** Renderer-safe identity only. Raw HTML/script never crosses preload. */
export interface HtmlArtifactProjection {
  format: "omni-chat-html-artifact-projection";
  version: 1;
  brainId: string;
  originalBrainId: string;
  artifactId: string;
  sourceActionId: string;
  contentSha256: string;
  title: string;
  interactive: boolean;
  operationalState: "enabled" | "dormant";
  revision: number;
  createdAt: string;
  savedArtifact: true;
  untrusted: true;
}

export interface HtmlArtifactResult {
  format: typeof HTML_ARTIFACT_RESULT_FORMAT;
  version: 1;
  artifact: HtmlArtifactProjection;
}

export interface HtmlArtifactOpenRequest {
  brainId: string;
  originalBrainId: string;
  artifactId: string;
  sourceActionId: string;
  contentSha256: string;
}

export interface HtmlArtifactLaunch {
  format: "omni-chat-html-artifact-launch";
  version: 1;
  viewId: string;
  channelNonce: string;
  artifact: HtmlArtifactProjection;
  url: string;
  height: number;
  scriptEnabled: boolean;
}

export interface HtmlArtifactToolRequest {
  format: typeof HTML_ARTIFACT_MESSAGE_FORMAT;
  version: 1;
  viewId: string;
  channelNonce: string;
  requestId: string;
  toolId: string;
  action: string;
  arguments: Record<string, unknown>;
}

export interface HtmlArtifactApprovalReview {
  kind: "html-artifact-operation";
  artifactId: string;
  contentSha256: string;
  resolved: { toolId: string; action: string; arguments: Record<string, unknown> };
}

export interface HtmlArtifactToolReceipt {
  format: "omni-chat-html-artifact-receipt";
  version: 1;
  viewId: string;
  requestId: string;
  executionId: string;
  toolId: string;
  action: string;
  state: "approval-required" | "complete" | "failed" | "denied";
  startedAt: string;
  finishedAt?: string;
  dispatchStarted?: true;
  output?: unknown;
  error?: string;
  review?: HtmlArtifactApprovalReview;
  reviewId?: string;
}

export interface HtmlArtifactReviewRequest {
  viewId: string;
  channelNonce: string;
  requestId: string;
  reviewId: string;
  decision: "approve" | "deny";
}

export interface HtmlArtifactActivationRequest extends HtmlArtifactOpenRequest {
  expectedRevision: number;
  acknowledged: true;
}

export interface HtmlArtifactsApi {
  open(request: HtmlArtifactOpenRequest): Promise<HtmlArtifactLaunch>;
  activate(request: HtmlArtifactActivationRequest): Promise<HtmlArtifactProjection>;
  request(request: HtmlArtifactToolRequest): Promise<HtmlArtifactToolReceipt>;
  review(request: HtmlArtifactReviewRequest): Promise<HtmlArtifactToolReceipt>;
  close(brainId: string, viewId: string, channelNonce: string): Promise<void>;
}
