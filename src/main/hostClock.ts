import { randomUUID } from "node:crypto";
import type { HostClockSnapshot } from "../shared/clock";

export interface HostClockSource { wallMs(): number; monotonicMs(): number; sessionId: string }
export const realHostClock: HostClockSource = {
  wallMs: Date.now,
  monotonicMs: () => Number(process.hrtime.bigint()) / 1_000_000,
  sessionId: randomUUID()
};

export function validTimeZone(value: unknown): string {
  const zone = value === undefined ? Intl.DateTimeFormat().resolvedOptions().timeZone : value;
  if (typeof zone !== "string" || !zone || zone.includes("\0")) throw new Error("Invalid time zone.");
  try { return new Intl.DateTimeFormat("en", { timeZone: zone }).resolvedOptions().timeZone; }
  catch { throw new Error("Use a supported IANA time zone, such as Europe/London."); }
}

export function hostClockSnapshot(source: HostClockSource = realHostClock, timeZone?: string): HostClockSnapshot {
  const epochMilliseconds = source.wallMs(), time = new Date(epochMilliseconds), zone = validTimeZone(timeZone);
  if (!Number.isFinite(epochMilliseconds) || !Number.isFinite(time.getTime())) throw new Error("The host clock is unavailable.");
  const formatter = new Intl.DateTimeFormat("en-CA", { timeZone: zone, year: "numeric", month: "2-digit", day: "2-digit",
    hour: "2-digit", minute: "2-digit", second: "2-digit", hourCycle: "h23" });
  const parts = Object.fromEntries(formatter.formatToParts(time).map(part => [part.type, part.value]));
  const asUtc = Date.UTC(Number(parts.year), Number(parts.month) - 1, Number(parts.day), Number(parts.hour), Number(parts.minute), Number(parts.second));
  const utcOffsetMinutes = Math.round((asUtc - Math.floor(epochMilliseconds / 1000) * 1000) / 60_000);
  return { epochMilliseconds, utc: time.toISOString(), local: formatter.format(time), timeZone: zone,
    utcOffsetMinutes, monotonicMilliseconds: source.monotonicMs(), sessionId: source.sessionId };
}
