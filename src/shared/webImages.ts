import type { ChatTaskAttachmentResolved } from "./taskAttachments";

/** An observed navigation reference is not a pixel observation or an instruction. */
export interface WebImageReference {
  format: "omni-web-image-reference"; version: 1; id: string; brainId: string; turnId: string;
  url: string; pageUrl: string; label?: string; fetched: false;
  source: { kind: "search-response" | "html-response" | "image-response"; url: string; sha256: string; fetchedAt: string };
}
export interface WebImageDownload {
  reference: WebImageReference; requestId: string; ownerTurnId: string; scopeTurnId: string;
  requestedUrl: string; finalUrl: string; redirects: string[]; fetchedAt: string;
  permissionRevision: string; learningPolicySha256?: string;
  mimeType: "image/png" | "image/jpeg" | "image/webp" | "image/gif"; sha256: string; bytes: number;
}
export interface WebImageSelection {
  format: "omni-web-image-selection"; version: 1; brainId: string; turnId: string;
  reference: WebImageReference; download: WebImageDownload; attachment: ChatTaskAttachmentResolved;
  selectionSha256: string;
  /** Only a later actual native receipt can attest pixel consumption or learning. */
  observation: { state: "pending-native-perception"; pixelsRead: false; parametersChanged: false; rawSourceTextInjected: false };
}
export interface WebImagePerceptionReceipt {
  format: "omni-web-image-perception"; version: 1; selectionSha256: string; referenceId: string;
  sourceSha256: string; sourceBytes: number; sourceFrames: number; decodedWindows: number;
  wholeInputTraversed: true; nativeSensory: true; rawSourceTextInjected: false; parametersChanged: false;
}
export interface WebImageLearningReceipt {
  format: "omni-web-image-learning"; version: 1; selectionSha256: string; referenceId: string;
  sourceSha256: string; sourceBytes: number; committed: true; duplicate: boolean;
  sourceIdentity: string; transactionId: string;
}
