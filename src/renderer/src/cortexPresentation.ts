import type { CortexActivity, CortexActivityQuery, CortexElement, CortexModule, CortexPage, CortexQuery } from "../../shared/cortexInspection";

export function cortexRequestKey(brainId: string, updatedAt: string, query: CortexQuery): string {
  return JSON.stringify([brainId, updatedAt, query.entity ?? "modules", query.moduleId ?? "", query.group ?? "",
    query.search ?? "", query.row ?? 0, query.offset ?? 0, query.pageSize ?? 64, query.cursor ?? ""]);
}

export function cortexRecordLabel(record: CortexPage["records"][number]): string {
  if ("module" in record) return `${record.module} · ${record.field} · ${record.role}`;
  if ("tokenId" in record) return `${record.position}: token ${record.tokenId} · ${record.kind}`;
  if ("sourceIndex" in record) {
    const from = record.sourceAxis === "constant-one" ? "constant 1" : `tensor column ${record.sourceIndex}`;
    return `${from} → ${record.targetAxis === "output-row" ? "output" : "tensor row"} ${record.targetIndex}${contactLabel(record)}`;
  }
  if ("column" in record) return `row ${record.row}, column ${record.column}${contactLabel(record)}`;
  return `Stored row ${record.row}`;
}

function contactLabel(record: Pick<CortexElement, "strength" | "resistance" | "effectiveContribution" | "isRepeatedContact" | "ordinal"> & { value?: number }): string {
  const sign = record.value === undefined ? "" : `: ${record.value > 0 ? "+1" : record.value}`;
  return sign + (record.strength === undefined ? "" : ` × ${record.strength}`) +
    (record.resistance === undefined ? "" : ` · resistance ${record.resistance}`) +
    (record.effectiveContribution === undefined ? "" : ` · contribution ${record.effectiveContribution}`) +
    (record.isRepeatedContact ? ` · repeated contact ${record.ordinal}` : "");
}

export function cortexInventoryLabel(page: CortexPage): string {
  if (page.logicalInventoryComplete) return `${page.logicalParameters.toLocaleString()} logical base parameter slots`;
  const inventory = page.inventory;
  if (!inventory) return `${page.logicalParameters.toLocaleString()} known parameter slots · incomplete inventory`;
  const prefix = inventory.baseLogicalInventoryComplete ? "base parameter slots" : "known base parameter slots";
  return `${inventory.baseLogicalParameters.toLocaleString()} ${prefix} · live repeated contacts paged separately`;
}

/** The runtime describes its real axis. Storage names cannot establish forward geometry. */
export function cortexActivityViewport(
  selected: CortexModule, query: CortexQuery, page: CortexPage, descriptor: CortexActivity, enabled: boolean
): CortexActivityQuery | null {
  const size = descriptor.outputSize;
  if (!descriptor.available || !descriptor.outputAxis || !Number.isSafeInteger(size) || size! < 1) return null;
  const contacts = page.records.filter((record): record is CortexElement => "row" in record && "column" in record);
  const link = page.records.find((record) => "sourceIndex" in record);
  let start = 0;
  let count = page.returned || query.pageSize || 64;
  if (query.entity === "elements" || query.entity === "links") {
    if (descriptor.outputAxis === "embedding-column" || selected.field === "bias") {
      start = selected.storageKind === "repeated-contact-page"
        ? contacts[0]?.column ?? (link && "sourceIndex" in link ? selected.field === "bias" ? link.targetIndex : link.sourceIndex : 0)
        : page.offset;
    } else if (descriptor.outputAxis === "output-row") {
      start = selected.storageKind === "repeated-contact-page"
        ? contacts[0]?.row ?? (link && "targetIndex" in link ? link.targetIndex : 0)
        : query.row ?? 0;
      count = 1;
    }
    // A flattened convolution kernel row is not an output channel. Its actual
    // output-channel viewport therefore starts at zero rather than guessing.
  } else if (query.entity === "rows" && descriptor.outputAxis === "output-row") {
    start = page.offset;
  }
  if (start < 0 || start >= size!) return null;
  return { module: selected.activityOwner ?? selected.module, enabled, start, count: Math.min(count, size! - start) };
}
