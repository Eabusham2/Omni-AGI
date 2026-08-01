import {
  useCallback,
  useEffect,
  useMemo,
  useState
} from "react";
import type {
  BrainDocument,
  EvolutionCandidate,
  EvolutionRun,
  ToolPermissionLevel
} from "@shared/types";
import { Icon } from "./icons";
import {
  buildEvolutionStartRequest,
  canApproveEvolution,
  canRollbackEvolution,
  canStopEvolution,
  groupEvolutionRuns,
  type EvolutionCandidateKind
} from "./evolutionView";

interface EvolutionWorkspaceProps {
  brain: BrainDocument;
  onOpenPermissions: () => void;
  onToast: (message: string) => void;
}

const candidateKinds: Array<{
  id: EvolutionCandidateKind;
  label: string;
  description: string;
}> = [
  {
    id: "substrate",
    label: "Neural substrate",
    description: "Assemblies, fast synapses, slow weights, and latent replay"
  },
  {
    id: "neural",
    label: "Neural weights",
    description: "Isolated safe-tensor overlay and retention evaluation"
  },
  {
    id: "data",
    label: "Learned sources",
    description: "Re-evaluate retained training sources in an isolated overlay"
  },
  {
    id: "architecture",
    label: "Compatible growth",
    description: "Add one ternary residual expert without changing tensor shapes"
  }
];

function shortId(value?: string): string {
  return value ? `${value.slice(0, 8)}…` : "origin";
}

function relativeTime(value: string): string {
  const delta = Math.max(0, Date.now() - new Date(value).getTime());
  const minutes = Math.round(delta / 60_000);
  if (minutes < 2) return "just now";
  if (minutes < 60) return `${minutes}m ago`;
  const hours = Math.round(minutes / 60);
  if (hours < 24) return `${hours}h ago`;
  return `${Math.round(hours / 24)}d ago`;
}

function stateLabel(value: string): string {
  return value.replaceAll("-", " ");
}

function candidateKind(candidate: EvolutionCandidate): string {
  const value = (candidate as EvolutionCandidate & {
    candidateKind?: string;
  }).candidateKind;
  return value ?? "source";
}

function policyCopy(permission: ToolPermissionLevel): string {
  if (permission === "ask") {
    return "Isolated experiments may run. Evaluation and promotion wait for your explicit review here.";
  }
  if (permission === "auto") {
    return "Passing neural, data, and substrate candidates may promote automatically. Source and architecture promotion still wait for review.";
  }
  if (permission === "full") {
    return "Passing candidates may promote under the immutable evaluator, with exact rollback retained.";
  }
  return "Recursive experiments are disabled. Enable their tool permission to start a candidate.";
}

function EvolutionCandidateCard({
  candidate,
  permission,
  busyAction,
  onApprove,
  onRollback
}: {
  candidate: EvolutionCandidate;
  permission: ToolPermissionLevel;
  busyAction: string;
  onApprove: (candidate: EvolutionCandidate) => void;
  onRollback: (candidate: EvolutionCandidate) => void;
}) {
  const latestEvaluation = candidate.evaluations.at(-1);
  const checksPassed =
    latestEvaluation?.checks.filter((check) => check.passed).length ?? 0;
  const parent = candidate.parentCandidateId
    ? `parent ${shortId(candidate.parentCandidateId)}`
    : "immutable origin";
  const busy = busyAction === candidate.id;

  return (
    <article
      className={`evolution-candidate evolution-candidate--${candidate.state}`}
      data-testid={`evolution-candidate-${candidate.id}`}
    >
      <div className="evolution-candidate__head">
        <span className="evolution-candidate__identity">
          <i><Icon name="pulse" size={15} /></i>
          <span>
            <small>{candidateKind(candidate).toUpperCase()} · G{candidate.generation}</small>
            <strong>{shortId(candidate.id)}</strong>
          </span>
        </span>
        <em className={`evolution-state evolution-state--${candidate.state}`}>
          <i /> {stateLabel(candidate.state)}
        </em>
      </div>
      <p>{candidate.objective}</p>
      <div className="evolution-lineage">
        <span><Icon name="fork" size={13} /> {parent}</span>
        <span>run {shortId(candidate.runId)}</span>
        <span>{relativeTime(candidate.updatedAt)}</span>
      </div>
      {latestEvaluation ? (
        <div className="evolution-evaluation">
          <span className={latestEvaluation.passed ? "is-passed" : "is-failed"}>
            <Icon name={latestEvaluation.passed ? "check" : "warning"} size={14} />
            {latestEvaluation.passed ? "Evaluation passed" : "Evaluation failed"}
          </span>
          <span>
            {checksPassed}/{latestEvaluation.checks.length} checks
          </span>
          {latestEvaluation.diffSha256 ? (
            <code>diff {shortId(latestEvaluation.diffSha256)}</code>
          ) : null}
          <details>
            <summary>Measured checks</summary>
            <ul>
              {latestEvaluation.checks.map((check) => (
                <li key={check.name} className={check.passed ? "is-passed" : "is-failed"}>
                  <Icon name={check.passed ? "check" : "close"} size={12} />
                  <span>{check.name}</span>
                  <em>{check.passed ? "pass" : "fail"}</em>
                </li>
              ))}
            </ul>
          </details>
        </div>
      ) : (
        <div className="evolution-evaluation evolution-evaluation--pending">
          <Icon name="activity" size={14} />
          Immutable evaluation has not completed yet.
        </div>
      )}
      {candidate.error ? (
        <p className="evolution-candidate__error">
          <Icon name="warning" size={13} /> {candidate.error}
        </p>
      ) : null}
      <div className="evolution-candidate__actions">
        {canApproveEvolution(candidate.state) ? (
          <button
            className="button button--primary"
            disabled={Boolean(busyAction)}
            onClick={() => onApprove(candidate)}
          >
            <Icon name={busy ? "activity" : "check"} size={15} />
            <span>
              {busy
                ? "Evaluating…"
                : permission === "ask"
                  ? "Review, evaluate & promote"
                  : "Evaluate candidate"}
            </span>
          </button>
        ) : null}
        {canRollbackEvolution(candidate.state) ? (
          <button
            className="button button--danger"
            disabled={Boolean(busyAction)}
            onClick={() => onRollback(candidate)}
          >
            <Icon name={busy ? "activity" : "archive"} size={15} />
            <span>{busy ? "Rolling back…" : "Roll back exact promotion"}</span>
          </button>
        ) : null}
      </div>
    </article>
  );
}

export function EvolutionWorkspace({
  brain,
  onOpenPermissions,
  onToast
}: EvolutionWorkspaceProps) {
  const [objective, setObjective] = useState("");
  const [kind, setKind] = useState<EvolutionCandidateKind>("substrate");
  const [recursive, setRecursive] = useState(true);
  const [candidates, setCandidates] = useState<EvolutionCandidate[]>([]);
  const [knownRuns, setKnownRuns] = useState<EvolutionRun[]>([]);
  const [loading, setLoading] = useState(Boolean(window.omni));
  const [starting, setStarting] = useState(false);
  const [busyRun, setBusyRun] = useState("");
  const [busyCandidate, setBusyCandidate] = useState("");
  const [loadError, setLoadError] = useState("");
  const permission =
    brain.toolPermissions?.find(
      (entry) => entry.toolId === "source.self-modify"
    )?.level ?? "off";
  const configured = brain.config.recursiveImprovement !== false;
  const enabled = configured && permission !== "off";
  const dataAvailable = brain.trainingSources.length > 0;
  const runs = useMemo(
    () => groupEvolutionRuns(candidates, knownRuns),
    [candidates, knownRuns]
  );

  const reload = useCallback(async (quiet = false) => {
    if (!window.omni) {
      setLoading(false);
      return;
    }
    if (!quiet) setLoading(true);
    try {
      setCandidates(await window.omni.evolution.listCandidates(brain.id));
      setLoadError("");
    } catch (error) {
      setLoadError(
        error instanceof Error
          ? error.message
          : "Evolution history could not be loaded."
      );
    } finally {
      if (!quiet) setLoading(false);
    }
  }, [brain.id]);

  useEffect(() => {
    setCandidates([]);
    setKnownRuns([]);
    void reload();
    if (!window.omni) return;
    const timer = window.setInterval(() => void reload(true), 5_000);
    return () => window.clearInterval(timer);
  }, [reload]);

  const start = async () => {
    if (!objective.trim() || starting || !enabled) return;
    if (kind === "data" && !dataAvailable) {
      onToast("This brain has no retained training-source identifiers for a data candidate.");
      return;
    }
    if (!window.omni) {
      onToast("Evolution requires the packaged local desktop runtime.");
      return;
    }
    setStarting(true);
    try {
      const run = await window.omni.evolution.start(
        buildEvolutionStartRequest({
          brainId: brain.id,
          objective,
          recursive,
          candidateKind: kind,
          sourceIds: brain.trainingSources.map((source) => source.id)
        })
      );
      setKnownRuns((current) => [
        run,
        ...current.filter((entry) => entry.id !== run.id)
      ]);
      setObjective("");
      await reload(true);
      onToast(
        run.state === "failed"
          ? run.error ?? "The improvement experiment failed to start."
          : `Started ${kind} improvement run ${shortId(run.id)} in an isolated overlay.`
      );
    } catch (error) {
      onToast(
        error instanceof Error
          ? error.message
          : "The improvement experiment could not start."
      );
    } finally {
      setStarting(false);
    }
  };

  const stop = async (runId: string) => {
    if (!window.omni || busyRun) return;
    setBusyRun(runId);
    try {
      const run = await window.omni.evolution.stop(brain.id, runId);
      setKnownRuns((current) => [
        run,
        ...current.filter((entry) => entry.id !== run.id)
      ]);
      await reload(true);
      onToast(`Stopped evolution run ${shortId(run.id)}; its archive remains inspectable.`);
    } catch (error) {
      onToast(error instanceof Error ? error.message : "The evolution run could not be stopped.");
    } finally {
      setBusyRun("");
    }
  };

  const approve = async (candidate: EvolutionCandidate) => {
    if (!window.omni || busyCandidate) return;
    setBusyCandidate(candidate.id);
    try {
      const updated = await window.omni.evolution.approve({
        brainId: brain.id,
        candidateId: candidate.id
      });
      await reload(true);
      onToast(
        updated.state === "promoted"
          ? `Candidate ${shortId(updated.id)} passed evaluation and was promoted with rollback available.`
          : updated.state === "rejected"
            ? updated.error ?? "The immutable evaluator rejected this candidate."
            : `Candidate ${shortId(updated.id)} is ${stateLabel(updated.state)}.`
      );
    } catch (error) {
      onToast(error instanceof Error ? error.message : "Candidate evaluation failed.");
    } finally {
      setBusyCandidate("");
    }
  };

  const rollback = async (candidate: EvolutionCandidate) => {
    if (!window.omni || busyCandidate) return;
    setBusyCandidate(candidate.id);
    try {
      const updated = await window.omni.evolution.rollback({
        brainId: brain.id,
        candidateId: candidate.id
      });
      await reload(true);
      onToast(`Restored the exact parent state for candidate ${shortId(updated.id)}.`);
    } catch (error) {
      onToast(error instanceof Error ? error.message : "The candidate could not be rolled back.");
    } finally {
      setBusyCandidate("");
    }
  };

  return (
    <div className="content-page evolution-page">
      <div className="content-page__title">
        <div>
          <span className="eyebrow-text">RECURSIVE IMPROVEMENT</span>
          <h1>Evolution lab</h1>
          <p>
            Isolate, evaluate, promote, and roll back neural or source changes
            without exposing the immutable evaluator to candidates.
          </p>
        </div>
        <div className="evolution-policy">
          <span className={`evolution-policy__badge evolution-policy__badge--${permission}`}>
            <i /> {permission} permission
          </span>
          <button className="button button--secondary" onClick={onOpenPermissions}>
            <Icon name="settings" size={15} />
            <span>Permissions</span>
          </button>
        </div>
      </div>

      <div className="evolution-layout">
        <aside className="surface evolution-compose">
          <div className="surface-title">
            <div>
              <h2>New isolated candidate</h2>
              <p>No running state changes until a passing candidate is promoted.</p>
            </div>
            <Icon name="pulse" size={19} />
          </div>
          <label>
            <span>Improvement objective</span>
            <textarea
              rows={4}
              maxLength={20_000}
              value={objective}
              onChange={(event) => setObjective(event.target.value)}
              placeholder="Describe a measured limitation or capability to improve…"
            />
          </label>
          <fieldset className="evolution-routes">
            <legend>Candidate substrate</legend>
            {candidateKinds.map((candidate) => {
              const unavailable = candidate.id === "data" && !dataAvailable;
              return (
                <label
                  key={candidate.id}
                  className={kind === candidate.id ? "is-selected" : ""}
                >
                  <input
                    type="radio"
                    name="evolution-candidate-kind"
                    value={candidate.id}
                    checked={kind === candidate.id}
                    disabled={unavailable}
                    onChange={() => setKind(candidate.id)}
                  />
                  <span>
                    <strong>{candidate.label}</strong>
                    <small>
                      {unavailable
                        ? "Requires at least one retained training source"
                        : candidate.description}
                    </small>
                  </span>
                </label>
              );
            })}
          </fieldset>
          <p className="evolution-route-note">
            Source candidates enter through exact hash-bound typed edit actions
            in chat; an objective alone is never treated as source code.
          </p>
          <label className="evolution-recursive">
            <input
              type="checkbox"
              checked={recursive}
              onChange={(event) => setRecursive(event.target.checked)}
            />
            <span>
              <strong>Reassess recursively</strong>
              <small>
                After verified promotion, test whether the improved brain can
                improve its own improvement process.
              </small>
            </span>
          </label>
          <div className={`evolution-permission-note evolution-permission-note--${permission}`}>
            <Icon name={enabled ? "info" : "warning"} size={15} />
            <span>
              <strong>
                {!configured
                  ? "Recursive improvement is disabled for this brain"
                  : `${permission.toUpperCase()} promotion policy`}
              </strong>
              <small>
                {configured
                  ? policyCopy(permission)
                  : "The brain configuration must allow recursive experiments before a candidate can start."}
              </small>
            </span>
          </div>
          <button
            className="button button--primary evolution-start"
            disabled={!enabled || !objective.trim() || starting}
            onClick={() => void start()}
          >
            <Icon name={starting ? "activity" : "play"} size={16} />
            <span>{starting ? "Creating isolated overlay…" : "Start experiment"}</span>
          </button>
        </aside>

        <section className="surface evolution-archive">
          <div className="surface-title">
            <div>
              <h2>Runs & candidate lineage</h2>
              <p>
                {runs.length} run{runs.length === 1 ? "" : "s"} ·{" "}
                {candidates.length} candidate{candidates.length === 1 ? "" : "s"}
              </p>
            </div>
            <button
              className="icon-button"
              aria-label="Refresh evolution archive"
              title="Refresh evolution archive"
              disabled={loading}
              onClick={() => void reload()}
            >
              <Icon name="activity" size={15} />
            </button>
          </div>
          {loadError ? (
            <div className="evolution-empty evolution-empty--error">
              <Icon name="warning" size={18} />
              <span><strong>Archive unavailable</strong><small>{loadError}</small></span>
            </div>
          ) : loading ? (
            <div className="evolution-empty">
              <Icon name="activity" size={18} />
              <span><strong>Reading evolution archive</strong><small>Synchronizing isolated worker candidates…</small></span>
            </div>
          ) : runs.length === 0 ? (
            <div className="evolution-empty">
              <Icon name="fork" size={18} />
              <span>
                <strong>No improvement runs yet</strong>
                <small>Start with a measured limitation; the immutable origin stays untouched.</small>
              </span>
            </div>
          ) : (
            <div className="evolution-run-list">
              {runs.map((run) => (
                <section className="evolution-run" key={run.id}>
                  <header>
                    <span>
                      <small>RUN {shortId(run.id)} · GENERATION {run.generation}</small>
                      <strong>{run.objective}</strong>
                      <em>
                        {run.recursive ? "recursive lineage" : "single experiment"} ·{" "}
                        updated {relativeTime(run.updatedAt)}
                      </em>
                    </span>
                    <span>
                      <em className={`evolution-state evolution-state--${run.state}`}>
                        <i /> {stateLabel(run.state)}
                      </em>
                      {canStopEvolution(run.state) ? (
                        <button
                          className="button button--danger"
                          disabled={Boolean(busyRun)}
                          onClick={() => void stop(run.id)}
                        >
                          <Icon name={busyRun === run.id ? "activity" : "close"} size={14} />
                          <span>{busyRun === run.id ? "Stopping…" : "Stop"}</span>
                        </button>
                      ) : null}
                    </span>
                  </header>
                  <div className="evolution-run__candidates">
                    {run.candidates.map((candidate) => (
                      <EvolutionCandidateCard
                        key={candidate.id}
                        candidate={candidate}
                        permission={permission}
                        busyAction={busyCandidate}
                        onApprove={(value) => void approve(value)}
                        onRollback={(value) => void rollback(value)}
                      />
                    ))}
                    {!run.candidates.length ? (
                      <div className="evolution-empty evolution-empty--compact">
                        Candidate overlay is being prepared.
                      </div>
                    ) : null}
                  </div>
                </section>
              ))}
            </div>
          )}
        </section>
      </div>
    </div>
  );
}
