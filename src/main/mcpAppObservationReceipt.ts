import type { McpAppObservationArtifact } from "../shared/mcpAppArtifacts";
import type { NativeAppObservationDisposition, NativeAppObservationFrontier, NativeAppObservationResult } from "../shared/nativeAppObservation";
import { nativeAppObservationFrontierSha256 } from "./nativeAppObservation";
import { checkedAppObservationContent } from "./mcpAppObservationAudit";

const SHA = /^[a-f0-9]{64}$/;
// CURRENT native app_observation._ID uses ASCII opaque1..128 + Python \Z.
// The negative all-character lookahead is true EOF; JS $ alone would accept
// a trailing newline. Queue/artifact/review UUID contracts stay separate.
const NATIVE_STREAM_ID = /^[A-Za-z0-9][A-Za-z0-9._-]{0,127}(?![\s\S])/;
const object = (value: unknown): Record<string, unknown> => value && typeof value === "object" && !Array.isArray(value) ? value as Record<string, unknown> : {};

export function checkedAppObservationFrontier(value: unknown, brainId: string): NativeAppObservationFrontier {
  const row = object(value);
  if (row.format !== "omni-native-app-observation-frontier" || row.version !== 1 || row.brainId !== brainId || !SHA.test(String(row.frontierSha256))) throw new Error("Native App observation returned another instance/frontier.");
  if (row.head !== null) {
    const head = object(row.head), learning = head.learning === null ? null : object(head.learning);
    const content = head.content === undefined ? undefined : checkedAppObservationContent(head.content, String(head.recordSha256), String(head.dataSha256));
    if (Object.keys(head).sort().join(",") !== (content ? "baseFrontierSha256,committedAt,content,dataSha256,disposition,format,learning,ordinal,previousSha256,recordSha256,state,streamId,version" : "baseFrontierSha256,committedAt,dataSha256,disposition,format,learning,ordinal,previousSha256,recordSha256,state,streamId,version") ||
      head.format !== "omni-native-app-observation-head" || head.version !== 1 || typeof head.streamId !== "string" || !NATIVE_STREAM_ID.test(head.streamId) || !Number.isSafeInteger(head.ordinal) || Number(head.ordinal) < 1 ||
      ![head.recordSha256, head.dataSha256, head.previousSha256, head.baseFrontierSha256].every(item => SHA.test(String(item))) ||
      typeof head.committedAt !== "string" || !Number.isFinite(Date.parse(head.committedAt)) ||
      !(head.disposition === "observe" && head.state === "observed" && learning || head.disposition === "superseded" && head.state === "advanced-without-observation" && learning === null && !content)) throw new Error("Native App observation returned an invalid committed head.");
    if (learning && (Object.keys(learning).sort().join(",") !== "afterimageId,assemblyId,experienceCountAfter,experienceCountBefore,kind,source,sourceLabel,steps,textBytes,textSha256" ||
      learning.kind !== "experience" || learning.source !== "mcp-app" || learning.steps !== 0 || typeof learning.sourceLabel !== "string" ||
      !Number.isSafeInteger(learning.experienceCountBefore) || Number(learning.experienceCountBefore) < 0 ||
      learning.experienceCountAfter !== Number(learning.experienceCountBefore) + 1 || typeof learning.assemblyId !== "string" || !learning.assemblyId || typeof learning.afterimageId !== "string" || learning.afterimageId.includes("\0") ||
      !SHA.test(String(learning.textSha256)) || !Number.isSafeInteger(learning.textBytes) || Number(learning.textBytes) < 0 ||
      Number(learning.textBytes) === 0 && (!content?.summary.nativeSensory || learning.textSha256 !== "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"))) throw new Error("App observation did not return an ordinary native experience receipt.");
  }
  const frontier = row as unknown as NativeAppObservationFrontier;
  if (nativeAppObservationFrontierSha256(frontier.head) !== frontier.frontierSha256) throw new Error("Native App frontier checksum changed.");
  return frontier;
}
export function checkedAppObservationResult(value: unknown, brainId: string, record: McpAppObservationArtifact,
  recordSha256: string, disposition: NativeAppObservationDisposition, expectedFrontierSha256: string): NativeAppObservationResult {
  const frontier = checkedAppObservationFrontier(value, brainId), row = object(value), head = frontier.head;
  if (Object.keys(row).some(key => !["format", "version", "brainId", "frontierSha256", "head", "duplicate", "reloadRequired", "nativeOwnerReleased"].includes(key)) ||
    (row.reloadRequired !== undefined || row.nativeOwnerReleased !== undefined) && (row.reloadRequired !== true || row.nativeOwnerReleased !== true) ||
    !head || typeof row.duplicate !== "boolean" || head.streamId !== record.streamId || head.ordinal !== record.ordinal ||
    head.recordSha256 !== recordSha256 || head.dataSha256 !== record.content.sha256 || head.previousSha256 !== record.previousSha256 ||
    head.disposition !== disposition || head.baseFrontierSha256 !== expectedFrontierSha256) throw new Error("Native App acknowledgement does not match the exact reviewed queued observation.");
  return row as unknown as NativeAppObservationResult;
}
