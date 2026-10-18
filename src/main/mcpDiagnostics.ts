import { createHash } from "node:crypto";
import { MCP_LOG_LEVELS, type McpDiagnosticPage, type McpDiagnosticRecord, type McpLogLevel } from "../shared/mcpHostServices";
import type { McpRequestOrigin } from "../shared/mcpInteractions";

const MAX_LOG_BYTES = 64 * 1024;
const MAX_RETAINED_BYTES = 4 * 1024 * 1024;
const MAX_RETAINED_RECORDS = 256;
const sourceKey = (origin: Pick<McpRequestOrigin, "brainId" | "serverId">) => JSON.stringify([origin.brainId, origin.serverId]);
const sensitiveKey = /^(?:authorization|proxy-authorization|cookie|set-cookie|password|passwd|secret|client[_-]?secret|api[_-]?key|access[_-]?token|refresh[_-]?token|bearer[_-]?token|session[_-]?token|view[_-]?token|private[_-]?key)$/i;
export interface McpDiagnosticOptions {
  requireAccess(origin: McpRequestOrigin): void;
  /** Reads configured secret snapshots only. Never refresh/connect or return them to UI. */
  knownSecrets(origin: McpRequestOrigin): Promise<readonly string[]>;
  admit(bytes: number, brainId: string): Promise<void>;
  now?(): number;
}
interface Counters { accepted: number; rejected: number; rateLimited: number; evicted: number; windowAt: number; windowCount: number; }

/** Bounded operational logs, not task content, durable brain history or learning.
 * Key/value redaction is defense in depth, not a universal PII detector. */
export class McpDiagnosticService {
  private readonly rows: Array<{ record: McpDiagnosticRecord; bytes: number; key: string }> = [];
  private readonly counters = new Map<string, Counters>();
  private readonly listeners = new Set<(record: McpDiagnosticRecord) => void>();
  private retainedBytes = 0;
  private sequence = 0;
  private closed = false;
  constructor(private readonly options: McpDiagnosticOptions) {}
  private now() { return this.options.now?.() ?? Date.now(); }
  private counter(key: string): Counters {
    let value = this.counters.get(key);
    if (!value) { value = { accepted: 0, rejected: 0, rateLimited: 0, evicted: 0, windowAt: this.now(), windowCount: 0 }; this.counters.set(key, value); }
    return value;
  }
  subscribe(listener: (record: McpDiagnosticRecord) => void): () => void { this.listeners.add(listener); return () => { this.listeners.delete(listener); }; }
  private checked(origin: McpRequestOrigin, signal: AbortSignal): void { signal.throwIfAborted(); if (this.closed) throw new Error("MCP diagnostics are closed."); this.options.requireAccess(origin); signal.throwIfAborted(); }
  private json(value: unknown): string {
    const stack: Array<{ value: unknown; depth: number }> = [{ value, depth: 0 }], seen = new Set<object>(); let estimated = 0;
    while (stack.length) {
      const item = stack.pop()!, current = item.value; if (item.depth > 128) throw new Error("MCP diagnostic nesting exceeds its transport bound.");
      if (typeof current === "string") estimated += Buffer.byteLength(current) + current.length * 6 + 2;
      else if (current === null || typeof current === "boolean") estimated += 5;
      else if (typeof current === "number" && Number.isFinite(current)) estimated += 32;
      else if (current && typeof current === "object") {
        if (seen.has(current) || !Array.isArray(current) && Object.getPrototypeOf(current) !== Object.prototype && Object.getPrototypeOf(current) !== null) throw new Error("MCP diagnostic data must be plain JSON without cycles.");
        seen.add(current); estimated += 2;
        for (const name in current) {
          if (!Object.hasOwn(current, name)) continue;
          const descriptor = Object.getOwnPropertyDescriptor(current, name);
          if (!descriptor || !Object.hasOwn(descriptor, "value")) throw new Error("MCP diagnostic JSON cannot evaluate accessors.");
          estimated += 4 + Buffer.byteLength(name) + name.length * 6;
          if (estimated > MAX_LOG_BYTES * 8 || stack.length >= MAX_LOG_BYTES) throw new Error("MCP diagnostic exceeds the bounded log envelope.");
          stack.push({ value: descriptor.value, depth: item.depth + 1 });
        }
      } else throw new Error("MCP diagnostic data must be finite JSON.");
      if (estimated > MAX_LOG_BYTES * 8 || stack.length > MAX_LOG_BYTES) throw new Error("MCP diagnostic exceeds the bounded log envelope.");
    }
    const raw = JSON.stringify(value); if (Buffer.byteLength(raw) > MAX_LOG_BYTES) throw new Error("MCP diagnostic exceeds 64 KiB; it was not silently truncated."); return raw;
  }
  async receive(origin: McpRequestOrigin, params: unknown, signal: AbortSignal, source: "server" | "app" = "server", viewId?: string, privateSecrets: readonly string[] = []): Promise<"recorded" | "rate-limited"> {
    this.checked(origin, signal); const id = sourceKey(origin), counters = this.counter(id), now = this.now();
    if (now - counters.windowAt >= 10_000 || now < counters.windowAt) { counters.windowAt = now; counters.windowCount = 0; }
    if (counters.windowCount >= 60) { counters.rateLimited++; return "rate-limited"; } counters.windowCount++;
    try {
      await this.options.admit(MAX_LOG_BYTES * 10, origin.brainId); this.checked(origin, signal);
      if (!params || typeof params !== "object" || Array.isArray(params)) throw new Error("Invalid MCP logging notification.");
      const input = params as Record<string, unknown>;
      if (!MCP_LOG_LEVELS.includes(input.level as McpLogLevel) || !Object.hasOwn(input, "data") ||
          input.logger !== undefined && (typeof input.logger !== "string" || input.logger.length > 256 || /[\0\r\n]/.test(input.logger)) ||
          Object.keys(input).some(key => !["level", "logger", "data", "_meta"].includes(key))) throw new Error("Invalid MCP logging level/logger/data.");
      const raw = this.json(input), secrets = [...privateSecrets, ...await this.options.knownSecrets(origin)].filter(value => typeof value === "string" && value.length).sort((a, b) => b.length - a.length);
      this.checked(origin, signal); let redactionApplied = false;
      const text = (value: string) => { let result = value; for (const secret of secrets) if (result.includes(secret)) { result = result.split(secret).join("[redacted]"); redactionApplied = true; }
        const safe = result.replace(/\bBearer\s+[A-Za-z0-9._~+\/-]+=*/gi, "Bearer [redacted]"); if (safe !== result) redactionApplied = true; return safe; };
      const redact = (value: unknown): unknown => typeof value === "string" ? text(value) : Array.isArray(value) ? value.map(redact) : value && typeof value === "object"
        ? Object.fromEntries(Object.entries(value).map(([key, child]) => { if (sensitiveKey.test(key)) { redactionApplied = true; return [text(key), "[redacted]"]; } return [text(key), redact(child)]; })) : value;
      let data = redact(input.data), logger = input.logger === undefined ? undefined : text(input.logger as string);
      try { this.json(data); } catch { data = { redacted: true, reason: "Sanitized log exceeded its bounded envelope; raw data was not retained." }; redactionApplied = true; }
      const record: McpDiagnosticRecord = { format: "omni-mcp-diagnostic", version: 1, sequence: ++this.sequence, observedAt: new Date(now).toISOString(),
        origin: { brainId: origin.brainId, serverId: origin.serverId, serverLabel: text(origin.serverLabel), endpoint: text(origin.endpoint), connectionEpoch: origin.connectionEpoch },
        source, ...(source === "app" && viewId ? { viewId } : {}), level: input.level as McpLogLevel, ...(logger === undefined ? {} : { logger }), data,
        sourceSha256: createHash("sha256").update(raw).digest("hex"), redactionApplied, untrustedData: true, nativeLearning: false };
      const bytes = Buffer.byteLength(JSON.stringify(record)); this.checked(origin, signal);
      while (this.rows.length && (this.rows.length >= MAX_RETAINED_RECORDS || this.retainedBytes + bytes > MAX_RETAINED_BYTES)) {
        const previous = this.rows.shift()!; this.retainedBytes -= previous.bytes; this.counter(previous.key).evicted++;
      }
      this.rows.push({ record, bytes, key: id }); this.retainedBytes += bytes; counters.accepted++;
      for (const listener of this.listeners) { try { listener(structuredClone(record)); } catch { /* readonly observer cannot replay logs or dispatch actions */ } }
      return "recorded";
    } catch (error) { counters.rejected++; throw error; }
  }
  page(brainId: string, serverId: string, afterSequence = 0, limit = 50): McpDiagnosticPage {
    if (typeof brainId !== "string" || !brainId || typeof serverId !== "string" || !serverId || !Number.isSafeInteger(afterSequence) || afterSequence < 0 || !Number.isSafeInteger(limit) || limit < 1 || limit > 100) throw new Error("Invalid diagnostic viewport.");
    const id = sourceKey({ brainId, serverId }), values = this.rows.filter(row => row.key === id && row.record.sequence > afterSequence), page = values.slice(0, limit), counts = this.counter(id);
    return { records: page.map(row => structuredClone(row.record)), hasMore: values.length > page.length,
      ...(values.length > page.length ? { nextSequence: page.at(-1)!.record.sequence } : {}), retainedBytes: this.rows.filter(row => row.key === id).reduce((sum, row) => sum + row.bytes, 0),
      accepted: counts.accepted, rejected: counts.rejected, rateLimited: counts.rateLimited, evicted: counts.evicted, retention: "bounded-current-process-only" };
  }
  close(): void { this.closed = true; this.rows.splice(0); this.retainedBytes = 0; this.counters.clear(); this.listeners.clear(); }
}
