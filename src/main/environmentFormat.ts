import { createHash } from "node:crypto";

export function environmentCanonical(value: unknown): string {
  const ordered = (input: unknown): unknown => {
    if (Array.isArray(input)) return input.map(ordered);
    if (!input || typeof input !== "object") return input;
    return Object.fromEntries(Object.entries(input as Record<string, unknown>)
      .sort(([left], [right]) => left < right ? -1 : left > right ? 1 : 0)
      .map(([key, item]) => [key, ordered(item)]));
  };
  return JSON.stringify(ordered(value));
}

export function environmentHash(value: string | Uint8Array): string {
  return createHash("sha256").update(value).digest("hex");
}

export function environmentObject(value: unknown, label = "environment value"): Record<string, unknown> {
  if (!value || typeof value !== "object" || Array.isArray(value)) throw new Error(`${label} must be an object.`);
  return value as Record<string, unknown>;
}

export function environmentId(value: unknown, label = "environment id"): string {
  if (typeof value !== "string" || !/^[a-f0-9]{32}$/.test(value)) throw new Error(`Invalid ${label}.`);
  return value;
}

export function environmentRequestId(value: unknown): string {
  if (typeof value !== "string" || !/^[A-Za-z0-9][A-Za-z0-9._:-]{0,127}$/.test(value)) throw new Error("Invalid environment request id.");
  return value;
}

export function environmentSha(value: unknown, label = "environment checksum"): string {
  if (typeof value !== "string" || !/^[a-f0-9]{64}$/.test(value)) throw new Error(`Invalid ${label}.`);
  return value;
}
