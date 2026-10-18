/** Attribution only. Labels are untrusted source data, never instructions. */
export interface ChatInputProvenance {
  kind: "mcp-app";
  /** Original observation owner, not rewritten on copy/merge. Current owner is the enclosing message/brain. */
  sourceBrainId: string;
  serverId: string;
  serverLabel: string;
  endpoint: string;
  resourceUri: string;
  artifactId: string;
  captureSha256: string;
  originalCallBrainId: string;
  toolId: string;
  reviewId: string;
  observationId: string;
  streamId: string;
  ordinal: number;
  recordSha256: string;
  dataSha256: string;
  nativeFrontierSha256: string;
  inputSha256: string;
}

/** Main-only proof input; never an accepted renderer/model argument. */
export interface NativeAppChatInput {
  recordJson: string;
  recordSha256: string;
  dataJson: string;
  dataSha256: string;
  captureJson: string;
  captureSha256: string;
  nativeFrontierSha256: string;
  provenance: ChatInputProvenance;
}

const ID = /^[A-Za-z0-9][A-Za-z0-9._:-]{0,255}$/, SHA = /^[a-f0-9]{64}$/;
export function checkedChatInputProvenance(value: unknown): ChatInputProvenance {
  if (!value || typeof value !== "object" || Array.isArray(value)) throw new Error("Invalid task source attribution.");
  const row = value as Record<string, unknown>;
  if (Object.keys(row).sort().join(",") !== "artifactId,captureSha256,dataSha256,endpoint,inputSha256,kind,nativeFrontierSha256,observationId,ordinal,originalCallBrainId,recordSha256,resourceUri,reviewId,serverId,serverLabel,sourceBrainId,streamId,toolId" ||
    row.kind !== "mcp-app" || !Number.isSafeInteger(row.ordinal) || Number(row.ordinal) < 1 ||
    ![row.sourceBrainId, row.serverId, row.artifactId, row.originalCallBrainId, row.reviewId, row.observationId, row.streamId, row.toolId].every(item => typeof item === "string" && ID.test(item)) ||
    ![row.captureSha256, row.recordSha256, row.dataSha256, row.nativeFrontierSha256, row.inputSha256].every(item => typeof item === "string" && SHA.test(item)) ||
    ![row.serverLabel, row.endpoint, row.resourceUri].every(item => typeof item === "string" && !/[\0\r\n]/.test(item))) throw new Error("Task source attribution lost its exact owner/hash binding.");
  return Object.fromEntries(Object.keys(row).sort().map(key => [key, row[key]])) as unknown as ChatInputProvenance;
}
