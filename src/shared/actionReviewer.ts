/** Optional external/local advice, never tool authority or Omni neural state. */
export interface ActionReviewerBudget {
  timeoutMs: number;
  maxRequestBytes: number;
  maxContextBytes: number;
  maxResponseBytes: number;
}

export interface ActionReviewerConfiguration {
  enabled: boolean;
  /** Settings revision supplied by trusted main, not by proposed tool operands. */
  revision: string;
  /** Current device/instance access-grant revision; copies do not inherit it. */
  ownerRevision: string;
  reviewerId: string;
  budget: ActionReviewerBudget;
}

export interface ActionReviewInvocation {
  brainId: string;
  toolId: string;
  action: string;
  arguments: Record<string, unknown>;
}

export interface ActionReviewerObservedItem {
  id: string;
  kind: "task" | "work" | "self-report";
  /** Explicitly selected observed content; no assertion of hidden intention. */
  text: string;
}

export interface ActionReviewerEvidence {
  sourceId: string;
  /** Exact text present in the observed action or selected context. */
  quote: string;
}

export interface ActionReviewerVerdict {
  verdict: "allow" | "ask";
  summary: string;
  evidence: ActionReviewerEvidence[];
}

/** Immutable historical advice. No approval token, credential or executable grant. */
export interface ActionReviewerApprovalReview extends ActionReviewerVerdict {
  kind: "action-review";
  formatVersion: 1;
  reviewId: string;
  brainId: string;
  requestId: string;
  chatTurnId?: string;
  actionEventId?: string;
  configurationRevision: string;
  ownerRevision: string;
  reviewerId: string;
  permissionRevision: string;
  proposalSha256: string;
  proposedInvocation: ActionReviewInvocation;
  resolvedInvocation?: ActionReviewInvocation;
  disposition: "reviewed" | "unavailable" | "unparseable" | "timeout" | "budget-refused";
  startedAt: string;
  finishedAt: string;
  elapsedMs: number;
  observedContext: ActionReviewerObservedItem[];
  requestBytes: number;
  responseBytes: number;
}

export interface ActionReviewerRequest {
  brainId: string;
  requestId: string;
  reviewerId: string;
  configurationRevision: string;
  ownerRevision: string;
  proposalSha256: string;
  /** Strict reviewer-only instructions plus exactly bounded observed data. */
  prompt: string;
  budget: ActionReviewerBudget;
}
