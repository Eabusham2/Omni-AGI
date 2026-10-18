import type { ChatInputAcceptedReceipt, ChatGenerationEnd } from "./types";

/** UI history only. None of these records are model input or dispatch authority. */
export interface SteerDirectionSource {
  brainId: string;
  turnId: string;
  predecessorTurnId: string;
  createdAt: string;
  content: string;
  submittedContent?: string;
  attachmentIds: string[];
  inputKind: "text" | "native-audio";
  audio?: {
    inputSha256: string; attachmentId: string; sourceSha256: string; sourceBytes: number;
    mimeType: string; utteranceId: string; sequence: number; capturedAt: string;
  };
}
export type SteerDirectionState = "pending" | "received" | "complete" | "failed" | "stopped" | "steered";
export interface SteerDirectionRecord {
  format: "omni-steer-direction";
  version: 1;
  id: string;
  brainId: string;
  sequence: number;
  turnId: string;
  predecessorTurnId: string;
  bindingSha256: string;
  source: SteerDirectionSource;
  state: SteerDirectionState;
  revision: number;
  createdAt: string;
  updatedAt: string;
  dismissedAt?: string;
  /** Main-normalized actual input receipt, never inferred from Started/Queue. */
  inputReceipt?: ChatInputAcceptedReceipt;
  acceptedAttachmentIds?: string[];
  /** Current host-history projection; immutable native receipt IDs stay above. */
  humanMessageId?: string;
  humanTurnId?: string;
  brainMessageId?: string;
  reply?: { humanMessageId: string; brainMessageId: string; createdAt: string; generationEnd?: ChatGenerationEnd };
  failure?: "dispatch-failed" | "stopped" | "post-reply-work-failed";
  dormant?: { reason: "copied-instance" | "merged-history" | "recovered-history" | "process-interrupted" | "saved-continuation"; fromBrainId: string; at: string };
  provenance?: Array<{ brainId: string; id: string }>;
}
export interface SteerDirectionPageQuery { beforeSequence?: number; limit?: number; includeDismissed?: boolean }
export interface SteerDirectionPage { entries: SteerDirectionRecord[]; total: number; nextBeforeSequence?: number }
export interface SteerDirectionDismissRequest { brainId: string; id: string; expectedRevision: number; dismissed: boolean }
export interface SteerDirectionEvent {
  brainId: string; id: string; sequence: number; revision: number; state: SteerDirectionState;
  type: "changed" | "dismissed" | "unavailable"; at: string;
  detail?: string;
}
export interface SteerDirectionsApi {
  list(brainId: string, query?: SteerDirectionPageQuery): Promise<SteerDirectionPage>;
  get(brainId: string, id: string): Promise<SteerDirectionRecord | null>;
  /** Bounded exact lookup for currently rendered human turns, including old pages. */
  forTurns(brainId: string, turnIds: string[]): Promise<SteerDirectionRecord[]>;
  dismiss(request: SteerDirectionDismissRequest): Promise<SteerDirectionRecord>;
  onEvent(listener: (event: SteerDirectionEvent) => void): () => void;
}
