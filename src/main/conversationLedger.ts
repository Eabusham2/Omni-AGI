import { createHash } from "node:crypto";
import { copyFile, mkdir } from "node:fs/promises";
import { join } from "node:path";
import { DatabaseSync } from "node:sqlite";
import type {
  ActionEvent,
  ChatMessage,
  ConversationLedgerEntry,
  ConversationLedgerPage,
  ConversationLedgerSummary,
  ThoughtTrace
} from "../shared/types";

const FORMAT = "omni-conversation-ledger";
const ZERO_HASH = "0".repeat(64);
const SAFE_ID = /^[a-zA-Z0-9][a-zA-Z0-9._-]{0,127}$/;
const ACTION_PAYLOAD_BRAIN_ID_META = "actionPayloadBrainId";

type Appendable =
  | { kind: "message"; value: ChatMessage }
  | { kind: "action"; value: ActionEvent }
  | { kind: "trace"; value: ThoughtTrace };

interface LedgerRow {
  sequence: number;
  entry_key: string;
  kind: "message" | "action" | "trace";
  created_at: string;
  attention_epoch: number;
  payload_json: string;
  payload_sha256: string;
  previous_sha256: string;
  row_sha256: string;
}

function sha256(value: string): string {
  return createHash("sha256").update(value).digest("hex");
}

function cleanOperationalPayload(value: unknown): unknown {
  if (Array.isArray(value)) return value.map(cleanOperationalPayload);
  if (typeof value !== "object" || value === null) return value;
  const output: Record<string, unknown> = {};
  for (const [key, child] of Object.entries(value as Record<string, unknown>)) {
    if (["dataUrl", "mediaUrl", "path", "artifactPath"].includes(key)) continue;
    output[key] = cleanOperationalPayload(child);
  }
  return output;
}

function attentionEpoch(value: Record<string, unknown>): number {
  const candidate = value.attentionEpoch ?? value.attention_epoch;
  return typeof candidate === "number" &&
    Number.isSafeInteger(candidate) &&
    candidate >= 0
      ? candidate
      : 0;
}

function entryKey(entry: Appendable): string {
  const value = entry.value as unknown as Record<string, unknown>;
  const id = typeof value.id === "string" ? value.id : sha256(JSON.stringify(value));
  if (entry.kind === "action") {
    return `action:${id}:${String(value.state ?? "unknown")}:${String(value.updatedAt ?? value.createdAt ?? "")}`;
  }
  return `${entry.kind}:${id}`;
}

function rowHash(row: Omit<LedgerRow, "row_sha256">): string {
  return sha256(JSON.stringify({
    sequence: row.sequence,
    entryKey: row.entry_key,
    kind: row.kind,
    createdAt: row.created_at,
    attentionEpoch: row.attention_epoch,
    payloadSha256: row.payload_sha256,
    previousSha256: row.previous_sha256
  }));
}

function semanticValue(value: Record<string, unknown>, ...keys: string[]): unknown {
  for (const key of keys) {
    if (value[key] !== undefined) return value[key];
  }
  return undefined;
}

function stableTraceBinding(value: Record<string, unknown>): Record<string, unknown> {
  return Object.fromEntries(
    Object.entries(value)
      .filter(([key]) =>
        /(?:receipt|checksum|turn_?id|input_?sha256)/i.test(key)
      )
      .sort(([left], [right]) => left.localeCompare(right))
  );
}

function idempotentProjectionDrift(
  kind: Appendable["kind"],
  storedJson: string,
  incomingJson: string
): boolean {
  let stored: Record<string, unknown> | null = null;
  let incoming: Record<string, unknown> | null = null;
  try {
    const storedValue = JSON.parse(storedJson) as unknown;
    const incomingValue = JSON.parse(incomingJson) as unknown;
    stored = typeof storedValue === "object" && storedValue !== null && !Array.isArray(storedValue)
      ? storedValue as Record<string, unknown>
      : null;
    incoming = typeof incomingValue === "object" && incomingValue !== null && !Array.isArray(incomingValue)
      ? incomingValue as Record<string, unknown>
      : null;
  } catch {
    return false;
  }
  if (!stored || !incoming || stored.id !== incoming.id) return false;
  if (kind === "message") {
    return JSON.stringify({
      role: stored.role,
      content: stored.content,
      turnId: semanticValue(stored, "turnId", "turn_id"),
      traceId: semanticValue(stored, "traceId", "trace_id")
    }) === JSON.stringify({
      role: incoming.role,
      content: incoming.content,
      turnId: semanticValue(incoming, "turnId", "turn_id"),
      traceId: semanticValue(incoming, "traceId", "trace_id")
    });
  }
  if (kind === "trace") {
    return JSON.stringify({
      input: stored.input,
      seed: stored.seed,
      turnId: semanticValue(stored, "turnId", "turn_id"),
      binding: stableTraceBinding(stored)
    }) === JSON.stringify({
      input: incoming.input,
      seed: incoming.seed,
      turnId: semanticValue(incoming, "turnId", "turn_id"),
      binding: stableTraceBinding(incoming)
    });
  }
  const storedAction = stored.action;
  const incomingAction = incoming.action;
  if (
    typeof storedAction !== "object" || storedAction === null || Array.isArray(storedAction) ||
    typeof incomingAction !== "object" || incomingAction === null || Array.isArray(incomingAction)
  ) return false;
  const immutableAction = (value: Record<string, unknown>): Record<string, unknown> => ({
    id: value.id,
    brainId: value.brainId,
    state: value.state,
    createdAt: value.createdAt,
    updatedAt: value.updatedAt,
    action: {
      kind: (value.action as Record<string, unknown>).kind,
      source: (value.action as Record<string, unknown>).source,
      toolId: (value.action as Record<string, unknown>).toolId,
      action: (value.action as Record<string, unknown>).action,
      arguments: (value.action as Record<string, unknown>).arguments
    }
  });
  return JSON.stringify(immutableAction(stored)) === JSON.stringify(immutableAction(incoming));
}

export class ConversationLedger {
  private constructor(
    readonly path: string,
    readonly brainId: string,
    private readonly database: DatabaseSync
  ) {}

  static async open(brainDirectory: string, brainId: string): Promise<ConversationLedger> {
    if (!SAFE_ID.test(brainId)) throw new Error("Invalid conversation ledger brain id.");
    const directory = join(brainDirectory, "conversation");
    await mkdir(directory, { recursive: true });
    const path = join(directory, "ledger.sqlite3");
    const database = new DatabaseSync(path);
    database.exec(`
      PRAGMA journal_mode=DELETE;
      PRAGMA synchronous=FULL;
      CREATE TABLE IF NOT EXISTS meta (
        key TEXT PRIMARY KEY,
        value TEXT NOT NULL
      );
      CREATE TABLE IF NOT EXISTS entries (
        sequence INTEGER PRIMARY KEY,
        entry_key TEXT NOT NULL UNIQUE,
        kind TEXT NOT NULL CHECK(kind IN ('message','action','trace')),
        created_at TEXT NOT NULL,
        attention_epoch INTEGER NOT NULL,
        payload_json TEXT NOT NULL,
        payload_sha256 TEXT NOT NULL,
        previous_sha256 TEXT NOT NULL,
        row_sha256 TEXT NOT NULL UNIQUE
      );
      CREATE INDEX IF NOT EXISTS entries_kind_sequence
        ON entries(kind, sequence DESC);
      CREATE INDEX IF NOT EXISTS entries_created_sequence
        ON entries(created_at, sequence);
    `);
    const stored = database.prepare("SELECT value FROM meta WHERE key='brainId'").get() as
      { value?: string } | undefined;
    if (stored?.value && stored.value !== brainId) {
      database.close();
      throw new Error("Conversation ledger belongs to another brain.");
    }
    database.prepare("INSERT OR REPLACE INTO meta(key,value) VALUES('brainId',?)").run(brainId);
    database.prepare("INSERT OR IGNORE INTO meta(key,value) VALUES('format',?)").run(FORMAT);
    const ledger = new ConversationLedger(path, brainId, database);
    try {
      ledger.ensureActionPayloadBrainIdentity();
      return ledger;
    } catch (error) {
      ledger.close();
      throw error;
    }
  }

  static async clone(
    sourceBrainDirectory: string,
    targetBrainDirectory: string,
    sourceBrainId: string,
    targetBrainId: string
  ): Promise<void> {
    const source = await ConversationLedger.open(sourceBrainDirectory, sourceBrainId);
    source.integrity();
    source.close();
    const targetDirectory = join(targetBrainDirectory, "conversation");
    await mkdir(targetDirectory, { recursive: true });
    const targetPath = join(targetDirectory, "ledger.sqlite3");
    await copyFile(join(sourceBrainDirectory, "conversation", "ledger.sqlite3"), targetPath);
    const database = new DatabaseSync(targetPath);
    try {
      const stored = database.prepare("SELECT value FROM meta WHERE key='brainId'").get() as
        { value?: string } | undefined;
      if (stored?.value !== sourceBrainId) {
        throw new Error("Cloned conversation ledger source identity changed.");
      }
      database.prepare("UPDATE meta SET value=? WHERE key='brainId'").run(targetBrainId);
    } finally {
      database.close();
    }
    // A copied conversation belongs to the new persistent identity. Re-open
    // through the normal migration path so action payloads and the complete
    // append-only hash chain are atomically rebound before the clone appears.
    const target = await ConversationLedger.open(targetBrainDirectory, targetBrainId);
    try {
      target.integrity();
    } finally {
      target.close();
    }
  }

  close(): void {
    this.database.close();
  }

  private ensureActionPayloadBrainIdentity(): void {
    const marker = this.database.prepare(
      "SELECT value FROM meta WHERE key=?"
    ).get(ACTION_PAYLOAD_BRAIN_ID_META) as { value?: string } | undefined;
    if (marker?.value === this.brainId) return;

    // Historical clones changed only meta.brainId after byte-copying the
    // source ledger. Their action payloads therefore still named the parent,
    // and replaying the copied tool journal collided with those rows. Verify
    // the old chain first: migration must never legitimize tampered content.
    this.integrity();
    const rows = this.database.prepare(
      "SELECT * FROM entries ORDER BY sequence"
    ).all() as unknown as LedgerRow[];
    const migrated: LedgerRow[] = [];
    let previous = ZERO_HASH;
    for (const row of rows) {
      let payloadJson = row.payload_json;
      if (row.kind === "action") {
        let parsed: unknown;
        try {
          parsed = JSON.parse(payloadJson) as unknown;
        } catch {
          throw new Error("Conversation ledger action payload is invalid.");
        }
        if (typeof parsed !== "object" || parsed === null || Array.isArray(parsed)) {
          throw new Error("Conversation ledger action payload is invalid.");
        }
        const action = parsed as Record<string, unknown>;
        if (
          typeof action.id !== "string" ||
          typeof action.brainId !== "string" ||
          !SAFE_ID.test(action.brainId) ||
          entryKey({ kind: "action", value: action as unknown as ActionEvent }) !==
            row.entry_key
        ) {
          throw new Error("Conversation ledger action identity is invalid.");
        }
        if (action.brainId !== this.brainId) {
          payloadJson = JSON.stringify({ ...action, brainId: this.brainId });
        }
      }
      const payloadSha256 = sha256(payloadJson);
      const base: Omit<LedgerRow, "row_sha256"> = {
        sequence: row.sequence,
        entry_key: row.entry_key,
        kind: row.kind,
        created_at: row.created_at,
        attention_epoch: row.attention_epoch,
        payload_json: payloadJson,
        payload_sha256: payloadSha256,
        previous_sha256: previous
      };
      const migratedRow: LedgerRow = {
        ...base,
        row_sha256: rowHash(base)
      };
      migrated.push(migratedRow);
      previous = migratedRow.row_sha256;
    }

    const update = this.database.prepare(`
      UPDATE entries SET
        payload_json=?, payload_sha256=?, previous_sha256=?, row_sha256=?
      WHERE sequence=?
    `);
    this.database.exec("BEGIN IMMEDIATE");
    try {
      for (const row of migrated) {
        update.run(
          row.payload_json,
          row.payload_sha256,
          row.previous_sha256,
          row.row_sha256,
          row.sequence
        );
      }
      this.database.prepare(
        "INSERT OR REPLACE INTO meta(key,value) VALUES(?,?)"
      ).run(ACTION_PAYLOAD_BRAIN_ID_META, this.brainId);
      this.database.exec("COMMIT");
    } catch (error) {
      this.database.exec("ROLLBACK");
      throw error;
    }
    this.integrity();
  }

  append(entries: Appendable[]): ConversationLedgerSummary {
    if (!entries.length) return this.summary();
    const last = this.database.prepare(
      "SELECT sequence,row_sha256 FROM entries ORDER BY sequence DESC LIMIT 1"
    ).get() as { sequence?: number; row_sha256?: string } | undefined;
    let sequence = Number(last?.sequence ?? 0);
    let previous = last?.row_sha256 ?? ZERO_HASH;
    const existing = this.database.prepare(
      "SELECT kind,payload_json,payload_sha256 FROM entries WHERE entry_key=?"
    );
    const insert = this.database.prepare(`
      INSERT INTO entries(
        sequence,entry_key,kind,created_at,attention_epoch,payload_json,
        payload_sha256,previous_sha256,row_sha256
      ) VALUES(?,?,?,?,?,?,?,?,?)
    `);
    this.database.exec("BEGIN IMMEDIATE");
    try {
      for (const entry of entries) {
        const key = entryKey(entry);
        const payload = cleanOperationalPayload(entry.value);
        const payloadJson = JSON.stringify(payload);
        const payloadSha256 = sha256(payloadJson);
        const prior = existing.get(key) as {
          kind?: Appendable["kind"];
          payload_json?: string;
          payload_sha256?: string;
        } | undefined;
        if (prior) {
          if (prior.payload_sha256 !== payloadSha256) {
            if (
              prior.kind !== entry.kind ||
              typeof prior.payload_json !== "string" ||
              !idempotentProjectionDrift(entry.kind, prior.payload_json, payloadJson)
            ) {
              throw new Error("Conversation ledger idempotency conflict.");
            }
          }
          continue;
        }
        sequence += 1;
        const source = entry.value as unknown as Record<string, unknown>;
        const createdAt = typeof source.createdAt === "string" &&
          Number.isFinite(Date.parse(source.createdAt))
            ? source.createdAt
            : new Date().toISOString();
        const base: Omit<LedgerRow, "row_sha256"> = {
          sequence,
          entry_key: key,
          kind: entry.kind,
          created_at: createdAt,
          attention_epoch: attentionEpoch(source),
          payload_json: payloadJson,
          payload_sha256: payloadSha256,
          previous_sha256: previous
        };
        const digest = rowHash(base);
        insert.run(
          base.sequence,
          base.entry_key,
          base.kind,
          base.created_at,
          base.attention_epoch,
          base.payload_json,
          base.payload_sha256,
          base.previous_sha256,
          digest
        );
        previous = digest;
      }
      this.database.exec("COMMIT");
    } catch (error) {
      this.database.exec("ROLLBACK");
      throw error;
    }
    return this.summary();
  }

  backfill(messages: ChatMessage[], traces: ThoughtTrace[]): ConversationLedgerSummary {
    const combined = [
      ...messages.map((value, index) => ({
        entry: { kind: "message" as const, value },
        sourceIndex: index,
        kindOrder: 0
      })),
      ...traces.map((value, index) => ({
        entry: { kind: "trace" as const, value },
        sourceIndex: index,
        kindOrder: 1
      }))
    ].sort((left, right) => {
      const leftTime = Date.parse(left.entry.value.createdAt);
      const rightTime = Date.parse(right.entry.value.createdAt);
      const time = (Number.isFinite(leftTime) ? leftTime : 0) -
        (Number.isFinite(rightTime) ? rightTime : 0);
      return time ||
        left.kindOrder - right.kindOrder ||
        left.sourceIndex - right.sourceIndex ||
        left.entry.value.id.localeCompare(right.entry.value.id);
    });
    return this.append(combined.map((item) => item.entry));
  }

  summary(): ConversationLedgerSummary {
    const counts = this.database.prepare(`
      SELECT COUNT(*) total,
        SUM(CASE WHEN kind='message' THEN 1 ELSE 0 END) messages,
        SUM(CASE WHEN kind='action' THEN 1 ELSE 0 END) actions,
        SUM(CASE WHEN kind='trace' THEN 1 ELSE 0 END) traces,
        MAX(attention_epoch) attention_epoch
      FROM entries
    `).get() as Record<string, number | null>;
    const head = this.database.prepare(
      "SELECT sequence,row_sha256 FROM entries ORDER BY sequence DESC LIMIT 1"
    ).get() as { sequence?: number; row_sha256?: string } | undefined;
    return {
      format: FORMAT,
      formatVersion: 1,
      totalEntries: Number(counts.total ?? 0),
      messageCount: Number(counts.messages ?? 0),
      actionCount: Number(counts.actions ?? 0),
      traceCount: Number(counts.traces ?? 0),
      headSequence: Number(head?.sequence ?? 0),
      headSha256: head?.row_sha256 ?? ZERO_HASH,
      attentionEpoch: Number(counts.attention_epoch ?? 0)
    };
  }

  private presentRow(row: LedgerRow): ConversationLedgerEntry {
    if (sha256(row.payload_json) !== row.payload_sha256 || rowHash(row) !== row.row_sha256) {
      throw new Error("Conversation ledger row checksum failed.");
    }
    const payload = JSON.parse(row.payload_json) as unknown;
    return {
      sequence: row.sequence,
      id: row.entry_key,
      kind: row.kind,
      createdAt: row.created_at,
      attentionEpoch: row.attention_epoch,
      payloadSha256: row.payload_sha256,
      rowSha256: row.row_sha256,
      ...(row.kind === "message" ? { message: payload as ChatMessage } : {}),
      ...(row.kind === "action" ? { action: payload as ActionEvent } : {}),
      ...(row.kind === "trace" ? { trace: payload as ThoughtTrace } : {})
    };
  }

  page(beforeSequence?: number, limitValue = 80): ConversationLedgerPage {
    const limit = Math.max(1, Math.min(200, Math.floor(limitValue)));
    const before = beforeSequence === undefined
      ? Number.MAX_SAFE_INTEGER
      : beforeSequence;
    if (!Number.isSafeInteger(before) || before < 1) {
      throw new Error("Conversation ledger cursor is invalid.");
    }
    const rows = this.database.prepare(`
      SELECT * FROM entries
      WHERE sequence < ? AND kind IN ('message','action')
      ORDER BY sequence DESC LIMIT ?
    `).all(before, limit) as unknown as LedgerRow[];
    rows.reverse();
    const entries = rows.map((row) => this.presentRow(row));
    const first = entries[0]?.sequence;
    const older = first === undefined
      ? 0
      : Number((this.database.prepare(`
          SELECT COUNT(*) count FROM entries
          WHERE sequence < ? AND kind IN ('message','action')
        `).get(first) as { count: number }).count);
    const summary = this.summary();
    return {
      brainId: this.brainId,
      entries,
      totalEntries: summary.messageCount + summary.actionCount,
      hasOlder: older > 0,
      ...(beforeSequence !== undefined ? { beforeSequence } : {}),
      ...(older > 0 && first !== undefined ? { nextBeforeSequence: first } : {}),
      headSequence: summary.headSequence,
      headSha256: summary.headSha256
    };
  }

  search(
    query: string,
    beforeSequence?: number,
    limitValue = 40,
    includePresentationReceipts = true
  ): ConversationLedgerPage {
    const limit = Math.max(1, Math.min(100, Math.floor(limitValue)));
    const before = beforeSequence ?? Number.MAX_SAFE_INTEGER;
    if (!Number.isSafeInteger(before) || before < 1) {
      throw new Error("Conversation history cursor is invalid.");
    }
    const needle = query.toLocaleLowerCase();
    const rows = this.database.prepare(`
      SELECT * FROM entries
      WHERE sequence < ?
        AND kind IN ('message','action','trace')
        AND (? = 1 OR instr(payload_json, '"presentationOnly":true') = 0)
        AND (? = '' OR instr(lower(payload_json), ?) > 0)
      ORDER BY sequence DESC LIMIT ?
    `).all(before, includePresentationReceipts ? 1 : 0, needle, needle, limit) as unknown as LedgerRow[];
    rows.reverse();
    const entries = rows.map((row) => this.presentRow(row));
    const matched = Number((this.database.prepare(`
      SELECT COUNT(*) count FROM entries
      WHERE kind IN ('message','action','trace')
        AND (? = 1 OR instr(payload_json, '"presentationOnly":true') = 0)
        AND (? = '' OR instr(lower(payload_json), ?) > 0)
    `).get(includePresentationReceipts ? 1 : 0, needle, needle) as { count: number }).count);
    return {
      brainId: this.brainId,
      entries,
      totalEntries: matched,
      hasOlder: entries.length === limit,
      ...(beforeSequence !== undefined ? { beforeSequence } : {}),
      ...(entries[0] ? { nextBeforeSequence: entries[0].sequence } : {}),
      headSequence: this.summary().headSequence,
      headSha256: this.summary().headSha256
    };
  }

  recentEvidence(limitValue = 500): ConversationLedgerEntry[] {
    const limit = Math.max(1, Math.min(2_000, Math.floor(limitValue)));
    const rows = this.database.prepare(
      "SELECT * FROM entries ORDER BY sequence DESC LIMIT ?"
    ).all(limit) as unknown as LedgerRow[];
    return rows.reverse().map((row) => this.presentRow(row));
  }

  integrity(): ConversationLedgerSummary {
    const rows = this.database.prepare("SELECT * FROM entries ORDER BY sequence").all() as
      unknown as LedgerRow[];
    let previous = ZERO_HASH;
    let expectedSequence = 1;
    for (const row of rows) {
      if (
        row.sequence !== expectedSequence ||
        row.previous_sha256 !== previous ||
        sha256(row.payload_json) !== row.payload_sha256 ||
        rowHash(row) !== row.row_sha256
      ) {
        throw new Error("Conversation ledger integrity verification failed.");
      }
      previous = row.row_sha256;
      expectedSequence += 1;
    }
    return this.summary();
  }
}
