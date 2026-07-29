import type {
  EvolutionCandidate,
  EvolutionRun,
  EvolutionRunState,
  EvolutionStartRequest
} from "../../shared/types";

export type EvolutionCandidateKind = NonNullable<
  EvolutionStartRequest["candidateKind"]
>;

export interface EvolutionRunView {
  id: string;
  objective: string;
  state: EvolutionRunState;
  recursive: boolean;
  generation: number;
  createdAt: string;
  updatedAt: string;
  candidates: EvolutionCandidate[];
}

export interface EvolutionStartOptions {
  brainId: string;
  objective: string;
  recursive: boolean;
  candidateKind: EvolutionCandidateKind;
  sourceIds: string[];
}

export function buildEvolutionStartRequest(
  options: EvolutionStartOptions
): EvolutionStartRequest {
  const request: EvolutionStartRequest = {
    brainId: options.brainId,
    objective: options.objective.trim(),
    recursive: options.recursive,
    candidateKind: options.candidateKind
  };

  if (options.candidateKind === "data") {
    request.sourceIds = [...options.sourceIds];
  } else if (
    options.candidateKind === "neural" ||
    options.candidateKind === "substrate"
  ) {
    request.latentReplay = true;
  } else if (options.candidateKind === "architecture") {
    request.architectureChange = {
      mutation: "grow-experts",
      addExperts: 1
    };
  }
  return request;
}

export function groupEvolutionRuns(
  candidates: EvolutionCandidate[],
  knownRuns: EvolutionRun[] = []
): EvolutionRunView[] {
  const groups = new Map<string, EvolutionRunView>();
  for (const run of knownRuns) {
    groups.set(run.id, {
      id: run.id,
      objective: run.objective,
      state: run.state,
      recursive: run.recursive,
      generation: run.generation,
      createdAt: run.createdAt,
      updatedAt: run.updatedAt,
      candidates: []
    });
  }

  const orderedCandidates = [...candidates].sort((left, right) =>
    left.createdAt.localeCompare(right.createdAt)
  );
  for (const candidate of orderedCandidates) {
    const existing = groups.get(candidate.runId);
    const group: EvolutionRunView = existing ?? {
      id: candidate.runId,
      objective: candidate.objective,
      state: candidate.state,
      recursive: false,
      generation: candidate.generation,
      createdAt: candidate.createdAt,
      updatedAt: candidate.updatedAt,
      candidates: []
    };
    group.objective = candidate.objective || group.objective;
    group.generation = Math.max(group.generation, candidate.generation);
    group.createdAt =
      group.createdAt.localeCompare(candidate.createdAt) <= 0
        ? group.createdAt
        : candidate.createdAt;
    if (candidate.updatedAt.localeCompare(group.updatedAt) >= 0) {
      group.updatedAt = candidate.updatedAt;
      group.state = candidate.state;
    }
    group.candidates.push(candidate);
    groups.set(candidate.runId, group);
  }

  return [...groups.values()].sort((left, right) =>
    right.updatedAt.localeCompare(left.updatedAt)
  );
}

export function canStopEvolution(state: EvolutionRunState): boolean {
  return state === "experimenting" || state === "awaiting-review";
}

export function canApproveEvolution(state: EvolutionRunState): boolean {
  return state === "experimenting" || state === "awaiting-review";
}

export function canRollbackEvolution(state: EvolutionRunState): boolean {
  return state === "promoted";
}
