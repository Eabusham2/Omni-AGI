import { createHash } from "node:crypto";
import { domainToASCII } from "node:url";
import type { WebDateEvidence, WebDateRange, WebDomainScope, WebLearningConstraints, WebLearningPolicy, WebSourceProvenance } from "../shared/learning";

function record(value: unknown): Record<string, unknown> {
  if (!value || typeof value !== "object" || Array.isArray(value)) throw new Error("Invalid web learning scope.");
  return value as Record<string, unknown>;
}
function keys(value: Record<string, unknown>, allowed: string[]): void {
  if (Object.keys(value).some(key => !allowed.includes(key))) throw new Error("Unknown web learning scope field.");
}
export function normalizeLearningUrl(value: unknown): string {
  if (typeof value !== "string" || !value.trim() || /[\u0000-\u0020]/.test(value.trim())) throw new Error("Enter a valid web source URL.");
  let raw = value.trim();
  if (!/^https?:\/\//i.test(raw)) {
    if (/^[a-z][a-z\d+.-]*:/i.test(raw) && !/^[^/?#]+:\d+(?:[/?#]|$)/.test(raw)) throw new Error("Web learning accepts HTTP(S) sources only.");
    raw = `https://${raw}`;
  }
  const url = new URL(raw);
  if (!["https:", "http:"].includes(url.protocol) || url.username || url.password) throw new Error("Use a credential-free HTTP(S) source.");
  url.hash = "";
  return url.toString();
}
function domain(value: unknown): string {
  if (typeof value !== "string" || !value.trim()) throw new Error("Enter a domain for the web scope.");
  const raw = value.trim().toLowerCase().replace(/\.$/, "");
  if (/[\s/@?#:*]/.test(raw)) throw new Error("A domain scope contains a hostname, not a URL or wildcard.");
  const ascii = domainToASCII(raw);
  if (!ascii || !/^[a-z\d](?:[a-z\d.-]*[a-z\d])?$/.test(ascii) || ascii.includes("..")) throw new Error("Invalid domain scope.");
  return ascii;
}
function scope(value: unknown, dated = false): WebDomainScope {
  const raw = record(value); keys(raw, dated ? ["domain", "includeSubdomains", "dates"] : ["domain", "includeSubdomains"]);
  if (raw.includeSubdomains !== undefined && typeof raw.includeSubdomains !== "boolean") throw new Error("Invalid subdomain scope.");
  return { domain: domain(raw.domain), includeSubdomains: raw.includeSubdomains === true };
}
function instant(value: unknown, end = false): string {
  if (typeof value !== "string" || !/^\d{4}-\d{2}-\d{2}(?:T.*)?$/.test(value)) throw new Error("Use an ISO date or time in the web scope.");
  const day = /^\d{4}-\d{2}-\d{2}$/.test(value);
  const normalized = day ? `${value}T${end ? "23:59:59.999" : "00:00:00.000"}Z` : value;
  if (!day && !/(?:Z|[+-]\d{2}:\d{2})$/.test(value)) throw new Error("Web date/time scope needs an explicit timezone.");
  const milliseconds = Date.parse(normalized);
  if (!Number.isFinite(milliseconds) || (day && new Date(milliseconds).toISOString().slice(0, 10) !== value)) throw new Error("Invalid web date.");
  return new Date(milliseconds).toISOString();
}
function dates(value: unknown): WebDateRange {
  const raw = record(value); keys(raw, ["from", "through"]);
  const result: WebDateRange = {};
  if (raw.from !== undefined) result.from = instant(raw.from);
  if (raw.through !== undefined) result.through = instant(raw.through, true);
  if (result.from && result.through && result.from > result.through) throw new Error("The web date range ends before it starts.");
  return result;
}
export function normalizeWebLearningConstraints(value: unknown = {}): WebLearningConstraints {
  const raw = record(value); keys(raw, ["allowedSites", "excludedSites", "dates", "domainDates"]);
  const result: WebLearningConstraints = {};
  for (const field of ["allowedSites", "excludedSites"] as const) if (raw[field] !== undefined) {
    if (!Array.isArray(raw[field])) throw new Error("Web domain scopes must be lists.");
    result[field] = raw[field].map(value => scope(value)).sort((a, b) => a.domain.localeCompare(b.domain));
    if (new Set(result[field].map(entry => `${entry.domain}:${entry.includeSubdomains}`)).size !== result[field].length) throw new Error("A web domain scope is duplicated.");
  }
  if (raw.dates !== undefined) result.dates = dates(raw.dates);
  if (raw.domainDates !== undefined) {
    if (!Array.isArray(raw.domainDates)) throw new Error("Per-domain dates must be a list.");
    result.domainDates = raw.domainDates.map(value => {
      const rawRule = record(value);
      return { ...scope(rawRule, true), dates: rawRule.dates === null ? null : dates(rawRule.dates) };
    }).sort((a, b) => b.domain.length - a.domain.length || a.domain.localeCompare(b.domain));
    if (new Set(result.domainDates.map(row => row.domain)).size !== result.domainDates.length) throw new Error("Per-domain date rules must be unique.");
  }
  return result;
}
function canonical(value: unknown): unknown {
  if (Array.isArray(value)) return value.map(canonical);
  if (value && typeof value === "object") return Object.fromEntries(Object.entries(value).sort(([a], [b]) => a.localeCompare(b)).map(([key, value]) => [key, canonical(value)]));
  return value;
}
export function makeWebLearningPolicy(constraints: unknown, instructions: string, revision: number, changedAt = new Date().toISOString()): WebLearningPolicy {
  if (!Number.isSafeInteger(revision) || revision < 1 || typeof instructions !== "string" || instructions.includes("\0")) throw new Error("Invalid learning policy revision.");
  const body = { revision, constraints: normalizeWebLearningConstraints(constraints), instructions, changedAt, changedBy: "user" as const };
  return { ...body, sha256: createHash("sha256").update(JSON.stringify(canonical(body))).digest("hex") };
}
export function validateWebLearningPolicy(value: unknown): WebLearningPolicy {
  const raw = record(value);
  const policy = makeWebLearningPolicy(raw.constraints, raw.instructions as string, raw.revision as number, raw.changedAt as string);
  if (policy.sha256 !== raw.sha256 || raw.changedBy !== "user" || !Number.isFinite(Date.parse(policy.changedAt))) throw new Error("Web scope revision failed integrity validation.");
  return policy;
}
function matches(host: string, site: WebDomainScope): boolean { return host === site.domain || (site.includeSubdomains === true && host.endsWith(`.${site.domain}`)); }
export function webUrlWithinScope(url: URL, constraints: WebLearningConstraints): boolean {
  const host = url.hostname.toLowerCase().replace(/\.$/, "");
  return !constraints.excludedSites?.some(site => matches(host, site)) &&
    (!constraints.allowedSites?.length || constraints.allowedSites.some(site => matches(host, site)));
}
function attribute(raw: string, name: string): string | undefined {
  const match = new RegExp(`\\b${name}\\s*=\\s*(?:"([^"]*)"|'([^']*)'|([^\\s>]+))`, "i").exec(raw);
  return match?.[1] ?? match?.[2] ?? match?.[3];
}
export function publicationFromHtml(html: string): WebDateEvidence {
  const declarations: WebDateEvidence["declarations"] = [];
  const add = (value: unknown, source: "html-meta" | "json-ld"): void => {
    if (typeof value !== "string") return;
    try { declarations.push({ value: instant(value), source }); } catch { /* Unknown remains unknown. */ }
  };
  for (const tag of html.matchAll(/<meta\b[^>]*>/gi)) {
    const name = (attribute(tag[0], "property") ?? attribute(tag[0], "name") ?? "").toLowerCase();
    if (["article:published_time", "datepublished", "dc.date.issued", "pubdate"].includes(name)) add(attribute(tag[0], "content"), "html-meta");
  }
  const visit = (value: unknown): void => {
    if (Array.isArray(value)) { for (const entry of value) visit(entry); return; }
    if (!value || typeof value !== "object") return;
    const data = value as Record<string, unknown>;
    add(data.datePublished, "json-ld");
    if (data["@graph"]) visit(data["@graph"]);
  };
  for (const script of html.matchAll(/<script\b([^>]*)>([\s\S]*?)<\/script\s*>/gi)) {
    if (attribute(script[1]!, "type")?.toLowerCase() !== "application/ld+json") continue;
    try { visit(JSON.parse(script[2]!)); } catch { /* A malformed declaration is not evidence. */ }
  }
  const unique = [...new Set(declarations.map(row => row.value))];
  return { status: unique.length === 1 ? "declared" : unique.length ? "conflicting" : "unknown",
    ...(unique.length === 1 ? { publishedAt: unique[0] } : {}), declarations, independentlyVerified: false };
}

export interface WebCrawlControl {
  policy(): Promise<WebLearningPolicy>;
  onSource?(provenance: WebSourceProvenance, result?: import("../shared/types").IngestResult): Promise<void>;
  onCoverage?(coverage: import("../shared/types").TrainingCoverage): void;
}
export function decideWebSource(requestedUrl: string, finalUrl: URL, contentType: string, publication: WebDateEvidence,
  policy: WebLearningPolicy, fetchedAt = new Date().toISOString()): WebSourceProvenance {
  const rule = policy.constraints.domainDates?.find(rule => matches(finalUrl.hostname.toLowerCase().replace(/\.$/, ""), rule));
  const range = rule ? rule.dates : policy.constraints.dates;
  let decision: WebSourceProvenance["decision"] = "admitted";
  if (!webUrlWithinScope(finalUrl, policy.constraints)) decision = "excluded-domain";
  else if (range && (range.from || range.through)) {
    if (publication.status !== "declared" || !publication.publishedAt) decision = "unknown-date";
    else if ((range.from && publication.publishedAt < range.from) || (range.through && publication.publishedAt > range.through)) decision = "excluded-date";
  }
  return { requestedUrl, finalUrl: finalUrl.toString(), fetchedAt, checkedAt: new Date().toISOString(), contentType, publication,
    policyRevision: policy.revision, policySha256: policy.sha256, decision };
}

/** Scope rejection is resumable under later explicit steering, not a network failure. */
export class WebScopeExclusion extends Error {
  constructor(readonly provenance: WebSourceProvenance) {
    super(`Web source excluded by current scope: ${provenance.decision}`);
    this.name = "WebScopeExclusion";
  }
}
export interface WebIngestContext {
  /** Host-owned frontier namespace, never a generated filesystem grant. */
  cacheNamespace?: string;
  requestedUrl: string;
  publication: WebDateEvidence;
  fetchedAt: string;
  policy(): Promise<WebLearningPolicy>;
  accepted?(provenance: WebSourceProvenance): void;
}
export async function guardLeasedWebSource(url: URL, contentType: string,
  lease: { publication?: WebDateEvidence; fetchedAt?: string }, context?: WebIngestContext): Promise<void> {
  if (!context) return;
  const provenance = decideWebSource(context.requestedUrl, url, contentType,
    lease.publication ?? { status: "unknown", declarations: [], independentlyVerified: false }, await context.policy(), lease.fetchedAt);
  // Old leases have no acquisition date. Never relabel their bytes with a
  // newer network response's publication date or capture timestamp.
  if (!lease.fetchedAt) delete provenance.fetchedAt;
  if (provenance.decision !== "admitted") throw new WebScopeExclusion(provenance);
  context.accepted?.(provenance);
}
