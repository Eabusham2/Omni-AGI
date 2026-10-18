/** A routing observation, never authority to execute a parser or tool. */
export interface SourceMediaType {
  value: string;
  source: "file-signature" | "http-content-type";
}

export function normalizeSourceMediaType(value: unknown): string | undefined {
  if (typeof value !== "string") return undefined;
  const normalized = value.split(";", 1)[0]?.trim().toLowerCase();
  return normalized && normalized.length <= 255 && /^[a-z0-9!#$&^_.+-]+\/[a-z0-9!#$&^_.+-]+$/.test(normalized)
    ? normalized : undefined;
}

export function validSourceMediaType(value: unknown): value is SourceMediaType {
  if (!value || typeof value !== "object" || Array.isArray(value)) return false;
  const item = value as Record<string, unknown>;
  return Object.keys(item).sort().join(",") === "source,value" &&
    ["file-signature", "http-content-type"].includes(String(item.source)) &&
    typeof item.value === "string" && normalizeSourceMediaType(item.value) !== undefined && normalizeSourceMediaType(item.value) === item.value;
}
