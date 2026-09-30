export type CortexEntity = "modules" | "rows" | "elements" | "links" | "boundaries";
export interface CortexQuery {
  entity?: CortexEntity; moduleId?: string; group?: string; search?: string;
  row?: number; offset?: number; pageSize?: number; cursor?: string;
}
export interface CortexModule {
  id: string; module: string; field: "weight" | "bias"; role: "core" | "plasticity";
  group: string; rows: number; columns: number | null; packedColumns: number;
  packedBytes: number; logicalShape: number[] | null; logicalParameters: number | null;
  layout: string; shapeEvidence: string; activationObserved: false; activation: null;
}
export interface CortexElement { row: number; column: number; value: -1 | 0 | 1; activationObserved: false; activation: null; }
export interface CortexRow { row: number; columns: number | null; activationObserved: false; activation: null; }
export interface CortexLink { sourceAxis: "input-column"; sourceIndex: number; targetAxis: "output-row";
  targetIndex: number; moduleId: string; relationship: string; semanticExplanationVerified: false; }
export interface CortexBoundary { position: number; tokenId: number; kind: string; byteValue: number | null;
  embeddingRow: number; languageHeadRow: number; observedInput: true; firingObserved: false; activation: null; }
export interface CortexPage {
  brainId: string; revision: string; source: "committed-packed-native-cortex"; entity: CortexEntity;
  total: number; offset: number; returned: number; hasMore: boolean; nextCursor?: string | null;
  records: Array<CortexModule | CortexElement | CortexRow | CortexLink | CortexBoundary>;
  selectedModule: CortexModule | null; groups: string[]; moduleCount: number; packedBytes: number;
  logicalParameters: number; logicalInventoryComplete: boolean;
  activity: { observed: false; reason: string };
  integrity: { generationManifestVerified: true; selectedTernaryCodesValidated: boolean; wholeRolePayloadScanned: false };
  relationships: { kind: "computational-dependencies-not-semantic-explanation"; ideaInputs: string[]; tokenInputs: string[] };
}
export interface CortexActivityQuery { module: string; enabled: boolean; start?: number; count?: number; }
export interface CortexActivity {
  available: boolean; observed: boolean; enabled?: boolean; module?: string; reason?: string;
  admittedCount?: number; requestedStart?: number; requestedCount?: number; evidence?: string;
  observation: null | { observedAtUnix: number; axis: string; start: number; end: number;
    values: Array<number | null>; embeddingRowObserved: number | null; sample: string;
    fullPopulationObserved: false; firingClassification: null };
}
