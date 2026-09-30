/** Structural parameter IDs only. This is neither text context nor answer memory. */
export interface ConceptIdView {
  format: "omni-structural-concept-id-view";
  version: 1;
  brainId: string;
  turnId: string;
  path: string;
  sha256: string;
  bytes: number;
  count: number;
}

export interface ConceptIdPage {
  brainId: string;
  ids: string[];
  totalCount: number;
  returnedCount: number;
  offset: number;
  nextOffset: number | null;
  viewTruncated: boolean;
  byteBudget: number;
  coverage: string;
  sourceBrainId?: string;
  sourceTurnId?: string;
  ownership?: string;
  historicalInspection?: boolean;
  executionAuthorized?: false;
  ancestryProvenance?: string;
}

export function normalizeConceptIdView(value: unknown, brainId: string, turnId: unknown): ConceptIdView {
  if (typeof value !== "object" || value === null || Array.isArray(value)) throw new Error("Invalid concept ID view.");
  const record = value as Record<string, unknown>;
  const fields = ["format", "version", "brainId", "turnId", "path", "sha256", "bytes", "count"];
  if (Object.keys(record).length !== fields.length || Object.keys(record).some((key) => !fields.includes(key)) ||
    record.format !== "omni-structural-concept-id-view" || record.version !== 1 ||
    record.brainId !== brainId || record.turnId !== turnId ||
    !/^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$/.test(brainId) ||
    typeof turnId !== "string" || !/^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$/.test(turnId) ||
    typeof record.sha256 !== "string" || !/^[a-f0-9]{64}$/.test(record.sha256) ||
    record.path !== `state/concept-id-views/${record.sha256}.jsonl` ||
    !Number.isSafeInteger(record.bytes) || (record.bytes as number) < 0 ||
    !Number.isSafeInteger(record.count) || (record.count as number) < 0
  ) throw new Error("Invalid concept ID view ownership, path or coverage.");
  return { ...record } as unknown as ConceptIdView;
}
