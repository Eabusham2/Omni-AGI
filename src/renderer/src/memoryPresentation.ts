import type { ThoughtTrace, TraceStep } from "../../shared/types";

/**
 * Simplify implementation vocabulary without changing the reported mechanism.
 * A stored fact or key/value lookup must never be presented as neural learning.
 */
export function presentMemoryOperation(value: string): string {
  return value
    .replace(/\bgrammar[- ]selected fact lookup\b/gi, "fact lookup")
    .replace(/\b(?:raw )?key\s*[/:-]\s*value lookup\b/gi, "key/value lookup")
    .replace(/\blatent replay\b/gi, "memory rehearsal")
    .replace(/\blatents?\b/gi, "neural memory")
    .replace(/\bplasticity\b/gi, "connection learning")
    .replace(/\bhuman consolidation\b/gi, "adaptive retention")
    .replace(/\b(?:background\s+)?consolidations?\b/gi, "adaptive retention");
}

export function presentJournalKind(value: string): string {
  const normalized = value.trim().toLocaleLowerCase().replaceAll("-", " ");
  return /^(?:human |background )?consolidation$/.test(normalized)
    ? "adaptive retention"
    : normalized;
}

export type TraceMechanismKind =
  | "activation"
  | "recall"
  | "timing"
  | "branching"
  | "adaptation"
  | "operation";

export interface TraceMechanismPresentation {
  id: string;
  kind: TraceMechanismKind;
  title: string;
  detail: string;
  measure?: string;
}

export interface TraceEvidencePresentation {
  mechanisms: TraceMechanismPresentation[];
  activatedConcepts: ThoughtTrace["activatedConcepts"];
  recalledIdeas: ThoughtTrace["recalledIdeas"];
  candidates: Array<{
    id: string;
    label: string;
    outcome: "selected" | "compared";
  }>;
}

/**
 * Preserve worker-reported signal/vector measurements as their own quantity.
 * These values are not interchangeable with the optional labeled concept and
 * recalled-preview arrays carried by a trace.
 */
export function presentTraceSignalMeasures(
  trace: Pick<ThoughtTrace, "steps">
): string[] {
  return [...new Set(trace.steps.flatMap((step) => {
    const measure = step.value?.trim();
    if (!measure || !/\b(?:signals?|vectors?)\b/i.test(measure)) return [];
    const evidence = `${step.stage} ${step.detail} ${measure}`;
    if (!/activ|inhibit|recall|associat|spread|signal|vector/i.test(evidence)) return [];
    return [presentMemoryOperation(measure)];
  }))];
}

function traceMechanismKind(step: TraceStep): TraceMechanismKind {
  const evidence = `${step.stage} ${step.detail} ${step.value ?? ""}`.toLocaleLowerCase();
  if (/plastic|synap|hebb|stdp|strengthen|weaken|weight (?:change|update)|connection learning|adapt(?:ed|ation)/.test(evidence)) {
    return "adaptation";
  }
  if (/time constant|timing|liquid|temporal integration|latency/.test(evidence)) {
    return "timing";
  }
  if (/branch|ponder|continuation|candidate|alternative/.test(evidence)) {
    return "branching";
  }
  if (/recall|retriev|rehears|replay|remember|memory|retention|consolidat/.test(evidence)) {
    return "recall";
  }
  if (/activat|signal|feature|association|connect(?:ed)? ideas|spread/.test(evidence)) {
    return "activation";
  }
  return "operation";
}

/**
 * Present the evidence that the trace actually recorded. Array position never
 * assigns meaning: a missing, repeated, or reordered worker event therefore
 * cannot masquerade as a fixed memory pipeline in the UI. The raw trace stays
 * untouched for factual JSON export.
 */
export function presentTraceEvidence(
  trace: Pick<
    ThoughtTrace,
    "id" | "steps" | "activatedConcepts" | "recalledIdeas" | "branches" | "selectedBranch"
  >
): TraceEvidencePresentation {
  const mechanisms = trace.steps.map((step, sourceIndex) => {
    const measure = step.value?.trim();
    return {
      id: `${trace.id}:${sourceIndex}:${step.stage}`,
      kind: traceMechanismKind(step),
      title: presentMemoryOperation(step.stage).trim() || "Recorded operation",
      detail: presentMemoryOperation(step.detail).trim() || "No detail was recorded.",
      ...(measure ? { measure: presentMemoryOperation(measure) } : {})
    };
  });
  const branchCount = Number.isSafeInteger(trace.branches)
    ? Math.max(0, trace.branches)
    : 0;
  const selectedBranch = Number.isSafeInteger(trace.selectedBranch)
    ? trace.selectedBranch
    : 0;
  return {
    mechanisms,
    activatedConcepts: trace.activatedConcepts,
    recalledIdeas: trace.recalledIdeas,
    candidates: Array.from({ length: branchCount }, (_, index) => {
      const candidate = index + 1;
      return {
        id: `${trace.id}:candidate:${candidate}`,
        label: `Candidate ${candidate}`,
        outcome: candidate === selectedBranch ? "selected" : "compared"
      };
    })
  };
}
