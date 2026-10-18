import { checkedNativeComputeCandidate, nativeComputeCanonical as canonical, nativeComputeSha256 as hash } from "./cachedDecoderExecutionService";
import type { NativeMixedRegionCandidate, NativeOperationAutoCandidate } from "../shared/nativeMixedRegions";

const SHA = /^[a-f0-9]{64}$/;
const record = (value: unknown): value is Record<string, unknown> => Boolean(value) && typeof value === "object" && !Array.isArray(value);
function exact(value: unknown, fields: string[]): Record<string, unknown> {
  if (!record(value) || Object.keys(value).sort().join("\0") !== fields.sort().join("\0")) throw new Error("Mixed region candidate shape differs.");
  return value;
}
const integer = (value: unknown): value is number => typeof value === "number" && Number.isSafeInteger(value) && value >= 0;

export function checkedNativeOperationCandidate(value: unknown): NativeOperationAutoCandidate {
  if (!record(value) || value.backendMode !== "native-mixed-sequential") return checkedNativeComputeCandidate(value);
  const row = exact(value, ["planSha256", "executionDevice", "deviceId", "memoryGroupId", "backendMode", "asyncDevices", "supported", "fallbackAcknowledged", "regionSchedule"]);
  const schedule = exact(row.regionSchedule, ["version", "registrySha256", "regionCount", "outerCandidate", "segments"]);
  if (schedule.version !== 1 || typeof schedule.registrySha256 !== "string" || !SHA.test(schedule.registrySha256) ||
      !integer(schedule.regionCount) || schedule.regionCount < 1 || !Array.isArray(schedule.segments) || !schedule.segments.length)
    throw new Error("Mixed region candidate has no exact actual registry/ranges.");
  const outer = checkedNativeComputeCandidate(schedule.outerCandidate);
  let end = 0;
  const segments = schedule.segments.map(value => {
    const segment = exact(value, ["first", "count", "candidate"]);
    if (!integer(segment.first) || !integer(segment.count) || segment.count < 1 || segment.first < end || segment.first + segment.count > Number(schedule.regionCount))
      throw new Error("Mixed region schedule overlaps or exceeds its registered regions.");
    end = segment.first + segment.count;
    const candidate = checkedNativeComputeCandidate(segment.candidate);
    if (candidate.executionDevice === outer.executionDevice) throw new Error("Mixed region schedule contains a redundant ordinary segment.");
    return { first: segment.first, count: segment.count, candidate };
  });
  const participants = [...new Set([outer, ...segments.map(row => row.candidate)].flatMap(candidate => candidate.asyncDevices))].sort();
  const fallbackAcknowledged = [outer, ...segments.map(row => row.candidate)].some(candidate => candidate.fallbackAcknowledged);
  const supported = [outer, ...segments.map(row => row.candidate)].every(candidate => candidate.supported);
  const payload = { executionDevice: outer.executionDevice, deviceId: outer.deviceId, memoryGroupId: outer.memoryGroupId,
    backendMode: "native-mixed-sequential", asyncDevices: participants, supported, fallbackAcknowledged,
    regionSchedule: { version: 1, registrySha256: schedule.registrySha256, regionCount: schedule.regionCount, outerCandidate: outer, segments } };
  if (canonical(row.asyncDevices) !== canonical(participants) || row.fallbackAcknowledged !== fallbackAcknowledged || row.supported !== supported ||
      row.executionDevice !== outer.executionDevice || row.deviceId !== outer.deviceId || row.memoryGroupId !== outer.memoryGroupId ||
      row.planSha256 !== hash(payload)) throw new Error("Mixed region plan changes its home, completion participants, fallback or exact descriptor hash.");
  return { ...payload, planSha256: String(row.planSha256) } as NativeMixedRegionCandidate;
}
