import { createHash } from "node:crypto";
import type { DeviceResourceAllocation } from "../shared/deviceResources";

/** Float64 bits make 50 and 50.0 identical across Python/JS without rounding. */
export function spillSelectionSignature(value: DeviceResourceAllocation): string {
  const bytes = Buffer.alloc(8); bytes.writeDoubleBE(value.percent);
  const body = { deviceId: value.deviceId, enabled: value.enabled, mode: value.mode, percentBits: bytes.toString("hex") };
  return `native-spill-selection-1:${createHash("sha256").update(JSON.stringify(body)).digest("hex")}`;
}
