/** Read the saved geometry without loading a neural model or accepting a renderer shape. */
import { lstat, readFile } from "node:fs/promises";
import { join } from "node:path";
import type { BrainConfig } from "../shared/types";
import { validateNativeArchitectureDescriptor } from "./nativeCoreInventory";

export async function savedRuntimeShape(brainDirectory: string, brainId: string, fallback: BrainConfig):
Promise<Pick<BrainConfig, "workingMemorySlots" | "nativeArchitecture">> {
  const path = join(brainDirectory, "engine", "brain.json");
  let info;
  try { info = await lstat(path); } catch (error) {
    if ((error as NodeJS.ErrnoException).code === "ENOENT") {
      return { workingMemorySlots: fallback.workingMemorySlots, nativeArchitecture: fallback.nativeArchitecture };
    }
    throw error;
  }
  if (!info.isFile() || info.isSymbolicLink()) throw new Error("Saved neural geometry metadata must be a regular managed file.");
  const metadata = JSON.parse(await readFile(path, "utf8")) as Record<string, unknown>;
  if (typeof metadata !== "object" || metadata === null || Array.isArray(metadata)) throw new Error("Saved neural geometry metadata is invalid.");
  if (metadata.brain_id !== undefined && metadata.brain_id !== brainId) throw new Error("Saved neural geometry belongs to a different brain.");
  if (metadata.format === "omni-engine-unmaterialized") {
    return { workingMemorySlots: fallback.workingMemorySlots, nativeArchitecture: fallback.nativeArchitecture };
  }
  const raw = metadata.config;
  if (typeof raw !== "object" || raw === null || Array.isArray(raw)) {
    if (metadata.format !== "omni-engine-unmaterialized") throw new Error("Saved neural geometry has no configuration.");
    return { workingMemorySlots: fallback.workingMemorySlots, nativeArchitecture: fallback.nativeArchitecture };
  }
  const config = raw as Record<string, unknown>;
  const slots = config.working_memory_slots ?? fallback.workingMemorySlots;
  if (typeof slots !== "number" || !Number.isSafeInteger(slots) || slots < 1) throw new Error("Saved neural workspace geometry is invalid.");
  const descriptor = config.native_architecture === undefined ? undefined : validateNativeArchitectureDescriptor(config.native_architecture);
  if (descriptor && descriptor.shape.workingMemoryItems !== slots) throw new Error("Saved native descriptor and workspace geometry disagree.");
  return { workingMemorySlots: slots, nativeArchitecture: descriptor };
}
