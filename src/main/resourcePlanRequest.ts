import type { WorkingMemoryPlanRequest } from "../shared/types";

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

/**
 * Validate renderer input at the privileged boundary and return only fields
 * understood by the authoritative main-process resource planner.
 */
export function requireWorkingMemoryPlanRequest(
  value: unknown,
): WorkingMemoryPlanRequest {
  if (
    !isRecord(value) ||
    !["auto", "extended", "manual"].includes(String(value.mode)) ||
    (value.hardwareTier !== undefined &&
      !["micro", "personal", "gpu", "workstation"].includes(
        String(value.hardwareTier),
      )) ||
    (value.requestedItems !== undefined &&
      typeof value.requestedItems !== "string") ||
    (value.requestedContextTokens !== undefined &&
      typeof value.requestedContextTokens !== "string") ||
    (value.acceleratorAvailable !== undefined &&
      typeof value.acceleratorAvailable !== "boolean") ||
    (value.systemRamMode !== undefined &&
      !["auto", "manual"].includes(String(value.systemRamMode))) ||
    (value.systemRamSharePercent !== undefined &&
      (typeof value.systemRamSharePercent !== "number" ||
        !Number.isFinite(value.systemRamSharePercent))) ||
    (value.storagePoolMode !== undefined &&
      !["auto", "manual"].includes(String(value.storagePoolMode))) ||
    (value.storagePoolBytes !== undefined &&
      typeof value.storagePoolBytes !== "string") ||
    (value.trainingSourceBytes !== undefined &&
      (typeof value.trainingSourceBytes !== "number" ||
        !Number.isSafeInteger(value.trainingSourceBytes) ||
        value.trainingSourceBytes < 0))
  ) {
    throw new Error("Invalid working-memory resource request.");
  }

  return {
    mode: value.mode as WorkingMemoryPlanRequest["mode"],
    hardwareTier:
      value.hardwareTier as WorkingMemoryPlanRequest["hardwareTier"],
    requestedItems: value.requestedItems as string | undefined,
    requestedContextTokens: value.requestedContextTokens as string | undefined,
    acceleratorAvailable: value.acceleratorAvailable as boolean | undefined,
    systemRamMode:
      value.systemRamMode as WorkingMemoryPlanRequest["systemRamMode"],
    systemRamSharePercent: value.systemRamSharePercent as number | undefined,
    storagePoolMode:
      value.storagePoolMode as WorkingMemoryPlanRequest["storagePoolMode"],
    storagePoolBytes: value.storagePoolBytes as string | undefined,
    trainingSourceBytes: value.trainingSourceBytes as number | undefined,
  };
}
