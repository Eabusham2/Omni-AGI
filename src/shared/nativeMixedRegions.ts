import type { DecoderExecutionCandidate } from "./decoderExecution";

/** Complete-operation hypothesis/result metadata, never an imported dispatch grant. */
export interface NativeMixedRegionCandidate extends Omit<DecoderExecutionCandidate, "backendMode"> {
  backendMode: "native-mixed-sequential";
  regionSchedule: {
    version: 1;
    registrySha256: string;
    regionCount: number;
    outerCandidate: DecoderExecutionCandidate;
    segments: Array<{ first: number; count: number; candidate: DecoderExecutionCandidate }>;
  };
}
export type NativeOperationAutoCandidate = DecoderExecutionCandidate | NativeMixedRegionCandidate;
