import type {
  BrainProvenance,
  BrainSummary,
  TrainingSource
} from "../../shared/types";

export interface BrainProvenancePresentation {
  originLabel: "Locally initialized" | "Merged native instance";
  compactLabel: string;
  ariaLabel: string;
}

function record(value: unknown): Record<string, unknown> | undefined {
  return typeof value === "object" && value !== null && !Array.isArray(value)
    ? value as Record<string, unknown>
    : undefined;
}

export function brainProvenancePresentation(input: {
  provenance?: BrainProvenance;
  runtimeCard?: Record<string, unknown> | null;
  trainingSources?: ReadonlyArray<Pick<TrainingSource, "name" | "learnedRecords">>;
  adaptation?: BrainSummary["adaptation"];
}): BrainProvenancePresentation | undefined {
  const runtime = input.runtimeCard;
  const cortex = record(runtime?.pretrained_text_cortex);
  const nativeRuntime = runtime &&
    runtime.origin_kind === "ground-up" &&
    runtime.pretrained === false &&
    runtime.baseFrozen === false &&
    !Object.hasOwn(runtime, "foundationModelId") &&
    !cortex;
  if (runtime && !nativeRuntime) return undefined;
  if (input.provenance && !["ground-up", "merged-native"].includes(input.provenance.originKind)) {
    return undefined;
  }
  if (input.provenance?.originKind === "merged-native" && input.provenance.merge) {
    const compactLabel = `OmniCortex · ${input.provenance.merge.parentIds.length} native parents`;
    return { originLabel: "Merged native instance", compactLabel,
      ariaLabel: `${compactLabel}. Complete parent state retained; no initial curriculum was rerun.` };
  }
  if (!nativeRuntime && input.provenance?.originKind !== "ground-up") {
    return undefined;
  }
  const compactLabel = "OmniCortex · locally initialized native core";
  return {
    originLabel: "Locally initialized",
    compactLabel,
    ariaLabel: `${compactLabel}. Its capabilities and adaptations are learned locally.`
  };
}
