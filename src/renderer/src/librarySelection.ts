import type { BrainSummary } from "../../shared/types";

export function librarySelection(summaries: readonly BrainSummary[], selectedIds: ReadonlySet<string>,
  visible: readonly BrainSummary[] = summaries): { selected: BrainSummary[]; hiddenSelected: number; allVisibleSelected: boolean; canMerge: boolean; mergeReason: string } {
  const seen = new Set<string>();
  const selected = summaries.filter(brain => {
    if (!selectedIds.has(brain.id) || seen.has(brain.id)) return false;
    seen.add(brain.id); return true;
  });
  const visibleIds = new Set(visible.map(brain => brain.id));
  const mergeReason = selected.length < 2 ? "Select at least two brains to merge."
    : selected.some(brain => brain.runtime !== "adaptive-core") ? "Only native adaptive-core brains can merge."
      : "";
  return { selected, hiddenSelected: selected.filter(brain => !visibleIds.has(brain.id)).length,
    allVisibleSelected: visible.length > 0 && visible.every(brain => selectedIds.has(brain.id)),
    canMerge: mergeReason === "", mergeReason };
}

export function toggleLibrarySelection(selectedIds: ReadonlySet<string>, id: string): Set<string> {
  const next = new Set(selectedIds);
  if (next.has(id)) next.delete(id); else next.add(id);
  return next;
}

export function selectVisibleLibraryBrains(selectedIds: ReadonlySet<string>, visible: readonly BrainSummary[]): Set<string> {
  return new Set([...selectedIds, ...visible.map(brain => brain.id)]);
}
