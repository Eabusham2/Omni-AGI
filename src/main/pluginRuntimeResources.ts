import type { PluginRuntimeResourceLease } from "../shared/pluginRuntime";
import { requireAutomationHeadroom } from "./automationResources";

export interface PluginRuntimeMemoryReading {
  residentBytes: number;
  ceilingBytes: number;
  freeBytes: number;
  heapAvailableBytes: number;
}

/** Cooperative whole-family admission, not an OS-enforced memory sandbox. */
export class PluginRuntimeResources {
  private tail: Promise<unknown> = Promise.resolve();
  private readonly reservations = new Set<{ bytes: number; released: boolean; started: boolean }>();
  constructor(private readonly measure: (brainId: string) => Promise<PluginRuntimeMemoryReading>,
    private readonly startupEstimateBytes = 64 * 1024 * 1024) {}

  private serialize<T>(work: () => Promise<T>): Promise<T> {
    const result = this.tail.then(work);
    this.tail = result.catch(() => undefined);
    return result;
  }
  private reserved(): number { return [...this.reservations].reduce((total, value) => total + value.bytes, 0); }

  admitTransfer(brainId: string, bytes: number): Promise<{ release(): Promise<void> }> {
    return this.serialize(async () => {
      if (!Number.isSafeInteger(bytes) || bytes < 0 || !Number.isSafeInteger(bytes * 6)) throw new Error("Invalid runtime transfer byte accounting.");
      const entry = { bytes: bytes * 6 + 4096, released: false, started: true };
      requireAutomationHeadroom(this.reserved() + entry.bytes, await this.measure(brainId));
      this.reservations.add(entry);
      return { release: () => this.serialize(async () => { entry.released = true; this.reservations.delete(entry); }) };
    });
  }

  admit(brainId: string, scriptBytes: number): Promise<PluginRuntimeResourceLease> {
    return this.serialize(async () => {
      if (!Number.isSafeInteger(scriptBytes) || scriptBytes < 0) throw new Error("Invalid runtime script byte accounting.");
      const readings = await this.measure(brainId), reserved = this.reserved();
      const available = Math.min(readings.ceilingBytes - readings.residentBytes - reserved,
        readings.freeBytes - reserved, readings.heapAvailableBytes - reserved);
      const maxMessageBytes = Math.floor(available / 64);
      if (!Number.isSafeInteger(maxMessageBytes) || maxMessageBytes < 4096) throw new Error("Runtime awaits physical transfer headroom.");
      const entry = { bytes: this.startupEstimateBytes + scriptBytes * 8 + maxMessageBytes * 6, released: false, started: false };
      requireAutomationHeadroom(reserved + entry.bytes, readings);
      this.reservations.add(entry);
      return {
        maxMessageBytes,
        check: processId => this.serialize(async () => {
          if (entry.released) throw new Error("Runtime resource owner was released.");
          if (processId !== undefined) {
            if (!Number.isSafeInteger(processId) || processId < 1) throw new Error("Runtime renderer measurement needs its actual OS identity.");
            // The real renderer is now included in whole-family RSS; retain
            // transfer reserves, not a second copy of the startup estimate.
            entry.started = true; entry.bytes = maxMessageBytes * 6;
          }
          requireAutomationHeadroom(this.reserved(), await this.measure(brainId));
        }),
        release: () => this.serialize(async () => { entry.released = true; this.reservations.delete(entry); })
      };
    });
  }
}
