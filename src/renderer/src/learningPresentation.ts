import type { LearningTask, WebLearningConstraints, WebDomainDateRule, WebDomainScope } from "../../shared/learning";

export function mergeLearningTasks(current: LearningTask[], incoming: LearningTask[]): LearningTask[] {
  const tasks = new Map(current.map(task => [task.id, task]));
  for (const task of incoming) {
    const prior = tasks.get(task.id);
    if (!prior || task.revision > prior.revision) tasks.set(task.id, task);
  }
  return [...tasks.values()].sort((a, b) => b.createdAt.localeCompare(a.createdAt) || a.id.localeCompare(b.id));
}
/** Snapshot before clearing: React may run a functional updater later. */
export function drainLearningTaskEvents(pending: Map<string, LearningTask>): LearningTask[] {
  const received = [...pending.values()]; pending.clear(); return received;
}

export function learningDomainList(text: string): WebDomainScope[] | undefined {
  const lines = text.split(/[\n,]/).map(value => value.trim()).filter(Boolean);
  if (!lines.length) return undefined;
  return [...new Set(lines.map(value => {
    const url = new URL(value.includes("://") ? value : `https://${value}`);
    if (!["http:", "https:"].includes(url.protocol) || url.username || url.password || url.port || url.hash || url.search || url.pathname !== "/") throw new Error("Site filters need domains, not paths, ports or credentials.");
    return url.hostname.toLowerCase();
  }))].map(domain => ({ domain, includeSubdomains: true }));
}

export function learningConstraints(input: { allowed: string; excluded: string; from: string; through: string; domainRules: string }): WebLearningConstraints {
  if (input.from && input.through && input.from > input.through) throw new Error("The earliest date must not be after the latest date.");
  let domainDates: WebDomainDateRule[] | undefined;
  if (input.domainRules.trim()) {
    try { domainDates = JSON.parse(input.domainRules) as WebDomainDateRule[]; }
    catch { throw new Error("Per-site dates must be a valid JSON rule list."); }
    if (!Array.isArray(domainDates)) throw new Error("Per-site dates must be a rule list.");
  }
  const allowedSites = learningDomainList(input.allowed), excludedSites = learningDomainList(input.excluded);
  return { ...(allowedSites ? { allowedSites } : {}), ...(excludedSites ? { excludedSites } : {}),
    ...(input.from || input.through ? { dates: { ...(input.from ? { from: input.from } : {}), ...(input.through ? { through: input.through } : {}) } } : {}),
    ...(domainDates ? { domainDates } : {}) };
}

export function learningTaskStatus(task: LearningTask): string {
  if (task.state === "complete" && !task.traversalComplete) return "Awaiting final coverage receipt";
  return { queued: "Queued", running: "Running", paused: "Paused", stopped: "Stopped", waiting: "Waiting for a task choice", complete: "Traversal complete", failed: "Needs attention" }[task.state];
}

export function learningSourcePercent(task: LearningTask): string | undefined {
  if (task.progress === undefined || !Number.isFinite(task.progress)) return undefined;
  if (task.state === "complete" && task.traversalComplete) return "100%";
  if (!task.currentSource) return undefined;
  return `${Math.min(99.9, Math.max(0, task.progress * 100)).toFixed(1)}%`;
}
