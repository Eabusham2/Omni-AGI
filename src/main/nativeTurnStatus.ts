import type { InstanceResumeAttachment } from "../shared/instanceLifecycleApi";

/** Main-owned scheduling evidence. This is not exposed as a renderer grant. */
export interface NativeTurnStatus {
  format: "omni-native-turn-status";
  formatVersion: 1;
  brainId: string;
  turnId: string;
  inputSha256: string;
  status: "committed" | "retained" | "active" | "accepted-unfinished" | "absent";
  loaded: boolean;
  admissionOpen: boolean;
  receipt?: {
    humanMessageId: string;
    brainMessageId: string;
    traceId: string;
    inferenceCount: number;
    parameterChecksumAfter: string;
    committedAt: string;
  };
  attachment?: InstanceResumeAttachment;
}
