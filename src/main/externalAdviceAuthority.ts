import { randomUUID } from "node:crypto";
import { lstat, mkdir, open, readFile, rename, unlink } from "node:fs/promises";
import { dirname } from "node:path";

export type AdviceAuthorityChannel = "assistant" | "reviewer";
export interface AdviceAuthority {
  brainId: string;
  channel: AdviceAuthorityChannel;
  ownerEpoch: string;
  revision: number;
  enabled: boolean;
}
interface Grant { ownerEpoch: string; revision: number; assistant: boolean; reviewer: boolean }
interface Document { format: "omni-advice-device-authority"; version: 1; grants: Record<string, Grant> }
const SAFE_ID = /^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$/;
const locks = new Map<string, Promise<unknown>>();
async function withPathLock<T>(path: string, operation: () => Promise<T>): Promise<T> {
  const previous = locks.get(path) ?? Promise.resolve();
  const current = previous.catch(() => undefined).then(operation);
  locks.set(path, current);
  try { return await current; }
  finally { if (locks.get(path) === current) locks.delete(path); }
}
function identity(brainId: string, channel: AdviceAuthorityChannel): void {
  if (!SAFE_ID.test(brainId) || !["assistant", "reviewer"].includes(channel)) throw new Error("Invalid advice authority owner.");
}

/** Installation-only consent. Never put this file in a brain or an export. */
export class ExternalAdviceAuthorityStore {
  constructor(private readonly path: string) {}

  private async read(): Promise<Document> {
    const info = await lstat(this.path).catch((error: NodeJS.ErrnoException) => {
      if (error.code === "ENOENT") return undefined; throw error;
    });
    if (!info) return { format: "omni-advice-device-authority", version: 1, grants: {} };
    if (!info.isFile() || info.isSymbolicLink() || info.size > 16 * 1024 * 1024) throw new Error("Advice authority storage is unsafe.");
    const value = JSON.parse(await readFile(this.path, "utf8")) as Document;
    if (value.format !== "omni-advice-device-authority" || value.version !== 1 ||
        !value.grants || typeof value.grants !== "object" || Array.isArray(value.grants)) throw new Error("Advice authority storage is invalid.");
    for (const [id, grant] of Object.entries(value.grants)) {
      if (!SAFE_ID.test(id) || !grant || typeof grant !== "object" ||
          typeof grant.ownerEpoch !== "string" || !/^[a-f0-9-]{36}$/.test(grant.ownerEpoch) ||
          !Number.isSafeInteger(grant.revision) || grant.revision < 1 ||
          typeof grant.assistant !== "boolean" || typeof grant.reviewer !== "boolean") throw new Error("Advice authority grant is invalid.");
    }
    return value;
  }

  async get(brainId: string, channel: AdviceAuthorityChannel): Promise<AdviceAuthority> {
    identity(brainId, channel);
    const document = await this.read();
    const grant = Object.hasOwn(document.grants, brainId) ? document.grants[brainId] : undefined;
    return { brainId, channel, enabled: grant?.[channel] ?? false,
      ownerEpoch: grant?.ownerEpoch ?? "unactivated", revision: grant?.revision ?? 0 };
  }

  /** Only trusted, explicit user settings IPC may call this; a model cannot. */
  async set(brainId: string, channel: AdviceAuthorityChannel, enabled: boolean): Promise<AdviceAuthority> {
    identity(brainId, channel);
    if (typeof enabled !== "boolean") throw new Error("Advice authority requires an explicit boolean.");
    return withPathLock(this.path, async () => {
      const document = await this.read(), previous = Object.hasOwn(document.grants, brainId) ? document.grants[brainId] : undefined;
      if (previous?.revision === Number.MAX_SAFE_INTEGER) throw new Error("Advice authority revision exhausted.");
      document.grants[brainId] = { ownerEpoch: previous?.ownerEpoch ?? randomUUID(),
        revision: (previous?.revision ?? 0) + 1, assistant: previous?.assistant ?? false,
        reviewer: previous?.reviewer ?? false, [channel]: enabled };
      await mkdir(dirname(this.path), { recursive: true });
      const temporary = `${this.path}.${randomUUID()}.tmp`;
      try {
        const file = await open(temporary, "wx", 0o600);
        try { await file.writeFile(JSON.stringify(document)); await file.sync(); }
        finally { await file.close(); }
        await rename(temporary, this.path);
      } finally { await unlink(temporary).catch((error: NodeJS.ErrnoException) => { if (error.code !== "ENOENT") throw error; }); }
      const grant = document.grants[brainId];
      return { brainId, channel, ownerEpoch: grant.ownerEpoch, revision: grant.revision, enabled: grant[channel] };
    });
  }
}
