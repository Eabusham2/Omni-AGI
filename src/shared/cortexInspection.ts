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
  storageKind?: "base-projection" | "repeated-contact-page";
  connectionStateBytes?: number; structuralBytes?: number;
  /** Exact owning forward module; repeated-contact pages do not forward independently. */
  activityOwner?: string;
}
export interface CortexElement { row: number; column: number; value: -1 | 0 | 1; activationObserved: false; activation: null;
  strength?: number; resistance?: number; effectiveContribution?: number; isRepeatedContact?: boolean; slot?: number; ordinal?: number; }
export interface CortexRow { row: number; columns: number | null; activationObserved: false; activation: null; }
export interface CortexLink { sourceAxis: "tensor-column" | "constant-one"; sourceIndex: number; targetAxis: "tensor-row" | "output-row";
  targetIndex: number; moduleId: string; relationship: string; semanticExplanationVerified: false;
  value?: -1 | 0 | 1; strength?: number; resistance?: number; effectiveContribution?: number; isRepeatedContact?: boolean; ordinal?: number; }
export interface CortexBoundary { position: number; tokenId: number; kind: string; byteValue: number | null;
  embeddingRow: number; languageHeadRow: number; observedInput: true; firingObserved: false; activation: null; }
export interface CortexPage {
  brainId: string; revision: string; source: "committed-packed-native-cortex"; entity: CortexEntity;
  total: number; offset: number; returned: number; hasMore: boolean; nextCursor?: string | null;
  records: Array<CortexModule | CortexElement | CortexRow | CortexLink | CortexBoundary>;
  selectedModule: CortexModule | null; groups: string[]; moduleCount: number; packedBytes: number;
  logicalParameters: number; logicalInventoryComplete: boolean;
  connectionStateBytes?: number; structuralBytes?: number; scannedThrough?: number;
  inventory?: { baseLogicalParameters: number; baseLogicalInventoryComplete: boolean;
    repeatedContactPages: number; allocatedRepeatedContactSlots: number; repeatedLiveContacts: null };
  activity: { observed: false; reason: string };
  integrity: { generationManifestVerified: true; selectedTernaryCodesValidated: boolean; wholeRolePayloadScanned: false };
  relationships: { kind: "computational-dependencies-not-semantic-explanation"; ideaInputs: string[]; tokenInputs: string[] };
}
export interface CortexActivityQuery { module: string; enabled: boolean; start?: number; count?: number; captureId?: string; }
export interface CortexActivity {
  available: boolean; observed: boolean; enabled?: boolean; module?: string; reason?: string;
  admittedCount?: number; requestedStart?: number; requestedCount?: number; evidence?: string;
  /** Described by the actual loaded forward owner, never guessed from its path. */
  outputAxis?: "embedding-column" | "output-row" | "output-channel";
  outputSize?: number;
  observation: null | { observedAtUnix: number; axis: string; start: number; end: number;
    values: Array<number | null>; embeddingRowObserved: number | null; sample: string;
    fullPopulationObserved: false; firingClassification: null };
}
