import type { EvolutionStartRequest } from "./types";

export function normalizeCompatibleArchitectureMutation(value: unknown): NonNullable<EvolutionStartRequest["architectureChange"]> {
  if (typeof value !== "object" || value === null || Array.isArray(value)) throw new Error("Architecture change must be typed.");
  const input = value as Record<string, unknown>;
  const fields: Record<string, readonly string[]> = {
    "grow-experts": ["addExperts"], "grow-depth": ["addLayers"],
    "grow-router": ["addNeurons"], "grow-regions": ["addRegions", "neuronsPerRegion"]
  };
  const mutation = typeof input.mutation === "string" ? input.mutation : "";
  const allowed = fields[mutation];
  if (!allowed || Object.keys(input).some((key) => key !== "mutation" && !allowed.includes(key))) throw new Error("Unsupported compatible architecture mutation; width/head geometry is not blindly padded.");
  const result: Record<string, unknown> = { mutation };
  for (const field of allowed) {
    const amount = input[field] ?? (field === "neuronsPerRegion" ? undefined : 1);
    if (typeof amount !== "number" || !Number.isSafeInteger(amount) || amount < 1) throw new Error("Architecture needs a positive " + mutation + " " + field + " count.");
    result[field] = amount;
  }
  return result as NonNullable<EvolutionStartRequest["architectureChange"]>;
}
