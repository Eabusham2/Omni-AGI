import { constants } from "node:fs";
import { open, opendir, realpath } from "node:fs/promises";
import type { SystemTemperatureReading } from "../shared/systemStats";

/** Trusted fixture seam. No root/path is accepted from a neural tool call. */
export interface SystemTemperatureIo {
  entries(): AsyncIterable<string>;
  read(zone: string, field: "temp" | "type", signal?: AbortSignal): Promise<string>;
}
const check = (signal?: AbortSignal): void => { if (signal?.aborted) throw new DOMException("Temperature read cancelled.", "AbortError"); };
const defaultIo: SystemTemperatureIo = {
  async *entries() {
    const directory = await opendir("/sys/class/thermal", { bufferSize: 32 });
    for await (const entry of directory) yield entry.name;
  },
  async read(zone, field, signal) {
    check(signal);
    const target = await realpath(`/sys/class/thermal/${zone}/${field}`);
    if (!target.startsWith("/sys/devices/")) throw new Error("Thermal sensor is outside the kernel device namespace.");
    const file = await open(target, constants.O_RDONLY | constants.O_NOFOLLOW);
    try {
      check(signal);
      if (!(await file.stat()).isFile()) throw new Error("Thermal value is not a sysfs file.");
      const bytes = Buffer.alloc(129), reading = await file.read(bytes, 0, bytes.length, 0);
      check(signal);
      if (reading.bytesRead > 128) throw new Error("Thermal value exceeds the bounded sysfs scalar format.");
      return new TextDecoder("utf-8", { fatal: true }).decode(bytes.subarray(0, reading.bytesRead));
    } finally { await file.close(); }
  }
};

/** Linux thermal-zone ABI temp is millidegrees Celsius, read-only here.
 * https://www.kernel.org/doc/html/v5.7/driver-api/thermal/sysfs-api.html
 * Do not guess hwmon voltage/thermistor conversions or install sensor drivers.
 */
export async function readSystemTemperatures(platform: NodeJS.Platform, signal?: AbortSignal, io: SystemTemperatureIo = defaultIo): Promise<SystemTemperatureReading | undefined> {
  check(signal);
  if (platform !== "linux") return undefined; // No passive adapter here, not a claim the OS cannot expose sensors.
  const readings: SystemTemperatureReading["readings"] = [], unavailableSensors: SystemTemperatureReading["unavailableSensors"] = [];
  try {
    for await (const zone of io.entries()) {
      check(signal);
      if (!/^thermal_zone[0-9]+$/.test(zone)) continue;
      try {
        const text = (await io.read(zone, "temp", signal)).trim();
        check(signal);
        const milliCelsius = /^-?[0-9]+$/.test(text) ? Number(text) : Number.NaN;
        if (!Number.isSafeInteger(milliCelsius) || milliCelsius < -273150) throw new Error("Invalid absolute temperature.");
        let label: string | null = null;
        try { const value = (await io.read(zone, "type", signal)).trim(); if (value && value.length <= 128 && !value.includes("\0")) label = value; }
        catch { check(signal); }
        check(signal); readings.push({ sensorId: zone, label, celsius: milliCelsius / 1000 });
      } catch {
        check(signal); unavailableSensors.push({ sensorId: zone, reason: "unreadable-or-invalid-reading" });
      }
    }
  } catch (error) {
    check(signal);
    if ((error as NodeJS.ErrnoException).code === "ENOENT" && !readings.length && !unavailableSensors.length) return undefined;
    throw error;
  }
  check(signal);
  if (!readings.length) {
    if (unavailableSensors.length) throw new Error("Present thermal sensors did not provide readable temperature observations.");
    return undefined;
  }
  return { interface: "linux-sysfs-thermal-zones", unit: "celsius", readings, unavailableSensors, traversalComplete: true };
}
