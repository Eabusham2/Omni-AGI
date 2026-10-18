/** Measured at the native byte encoder, not inferred from prompt or delivery time. */
export type NativeMediaTimeline = {
  format: "omni-native-media-timeline";
  version: 1;
  producer: "native-byte-encoder";
  layout: "snapshot" | "replacement-from-zero" | "complete-from-zero";
  unit: "image" | "samples" | "frames";
  start: 0;
  count: number;
  timebaseNumerator: number;
  timebaseDenominator: number;
  /** Only present when the actual producer guarantees this prefix is stable. */
  stableCount?: number;
  /** Present only for actual audio successfully encoded into this same container. */
  containerAudio?: { sampleRate: number; sampleCount: number; startSample: 0 };
};

export function nativeMediaTimeline(value: unknown): NativeMediaTimeline | undefined {
  if (!value || typeof value !== "object" || Array.isArray(value)) return undefined;
  const row = value as Record<string, unknown>;
  if (Object.keys(row).some(key => !["format", "version", "producer", "layout", "unit", "start", "count", "timebaseNumerator", "timebaseDenominator", "stableCount", "containerAudio"].includes(key)) ||
      row.format !== "omni-native-media-timeline" || row.version !== 1 || row.producer !== "native-byte-encoder" || row.start !== 0 ||
      !["snapshot", "replacement-from-zero", "complete-from-zero"].includes(String(row.layout)) || !["image", "samples", "frames"].includes(String(row.unit))) return undefined;
  for (const key of ["count", "timebaseNumerator", "timebaseDenominator"] as const) if (!Number.isSafeInteger(row[key]) || Number(row[key]) < 1) return undefined;
  if (row.unit === "image" && (row.layout !== "snapshot" || row.count !== 1 || row.timebaseNumerator !== 1 || row.timebaseDenominator !== 1) || row.unit !== "image" && row.layout === "snapshot") return undefined;
  if (row.stableCount !== undefined && (!Number.isSafeInteger(row.stableCount) || Number(row.stableCount) < 0 || Number(row.stableCount) > Number(row.count))) return undefined;
  if (row.containerAudio !== undefined) {
    const audio = row.containerAudio as Record<string, unknown>;
    if (!audio || typeof audio !== "object" || Array.isArray(audio) || Object.keys(audio).sort().join(",") !== "sampleCount,sampleRate,startSample" || row.unit !== "frames" || row.layout !== "complete-from-zero" || audio.startSample !== 0 ||
        !Number.isSafeInteger(audio.sampleRate) || Number(audio.sampleRate) < 1 || !Number.isSafeInteger(audio.sampleCount) || Number(audio.sampleCount) < 1) return undefined;
  }
  return value as NativeMediaTimeline;
}

/** Main presentation clock; this is not a neural generation or native frame clock. */
export interface ImaginationDeliveryAnchor {
  version: 1;
  kind: "first-verified-main-delivery";
  firstObservedAt: string;
  elapsedMs: number;
  truncatedStart: boolean;
}
export function imaginationDeliveryAnchor(value: unknown): ImaginationDeliveryAnchor | undefined {
  if (!value || typeof value !== "object" || Array.isArray(value)) return undefined; const row = value as Record<string, unknown>;
  if (Object.keys(row).sort().join(",") !== "elapsedMs,firstObservedAt,kind,truncatedStart,version" || row.version !== 1 || row.kind !== "first-verified-main-delivery" || typeof row.firstObservedAt !== "string" || !Number.isFinite(Date.parse(row.firstObservedAt)) || typeof row.elapsedMs !== "number" || !Number.isFinite(row.elapsedMs) || row.elapsedMs < 0 || typeof row.truncatedStart !== "boolean") return undefined;
  return value as ImaginationDeliveryAnchor;
}
