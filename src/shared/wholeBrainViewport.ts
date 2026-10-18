import type { WholeBrainInspectionApi, WholeBrainInspectionPage, WholeBrainInspectionQuery } from "./wholeBrainInspection";

/** Transport windows, not limits on any learned owner/record cardinality. */
export interface WholeBrainViewport {
  parameters?: WholeBrainInspectionPage;
  parameterDetail?: WholeBrainInspectionPage;
  aggregates?: WholeBrainInspectionPage;
  ideas?: WholeBrainInspectionPage;
  sparseDetail?: WholeBrainInspectionPage;
  members?: WholeBrainInspectionPage;
}
export function sameWholeBrainSnapshot(left: WholeBrainInspectionPage, right: WholeBrainInspectionPage): boolean {
  return left.brainId === right.brainId && (Object.keys(left.snapshot) as Array<keyof typeof left.snapshot>)
    .every(key => left.snapshot[key] === right.snapshot[key]);
}
export function retainWholeBrainWindow(current: WholeBrainViewport, page: WholeBrainInspectionPage): WholeBrainViewport {
  if (page.entity === "cortex") return { ...current, [page.cortex?.entity === "modules" ? "parameters" : "parameterDetail"]: page };
  if (page.entity === "substrate") return { ...current, [page.substrate?.membershipPage ? "members" : page.substrate?.clusters.length || page.substrate?.entity === "overview"
    ? "aggregates" : page.substrate?.entity === "assemblies" ? "ideas" : "sparseDetail"]: page };
  return current;
}
export function wholeBrainWindowPages(viewport: WholeBrainViewport): WholeBrainInspectionPage[] {
  return Object.values(viewport).filter((page): page is WholeBrainInspectionPage => Boolean(page));
}
export const initialWholeBrainQueries: readonly WholeBrainInspectionQuery[] = [
  { entity: "catalog", catalogFacet: "overview", pageSize: 128 },
  { entity: "cortex", pageSize: 32, cortex: { entity: "modules" } },
  { entity: "substrate", pageSize: 64, substrate: { entity: "overview", zoom: 0 } },
  { entity: "substrate", pageSize: 8, substrate: { entity: "assemblies", zoom: 1 } }
];
/** One sender's worker requests are intentionally sequential: parallel reads
 * cancel each other. Publish nothing until the entire new viewport is fenced. */
export async function hydrateWholeBrainViewport(api: WholeBrainInspectionApi, brainId: string, isCurrent: () => boolean) {
  const pages: WholeBrainInspectionPage[] = [];
  let windows: WholeBrainViewport = {};
  for (const query of initialWholeBrainQueries) {
    if (!isCurrent()) throw new Error("Whole-brain refresh was superseded");
    const page = await api.query(brainId, query);
    if (!isCurrent()) throw new Error("Whole-brain refresh was superseded");
    if (page.brainId !== brainId || page.entity !== query.entity || pages[0] && !sameWholeBrainSnapshot(pages[0], page))
      throw new Error("The brain changed while refreshing. The previous consistent map was kept.");
    pages.push(page); windows = retainWholeBrainWindow(windows, page);
  }
  return { overview: pages[0]!, windows, active: pages[2]!, query: initialWholeBrainQueries[2]! };
}
