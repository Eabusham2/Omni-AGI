import { randomUUID } from "node:crypto";
import { chmod, mkdir, readFile, rename, rm, writeFile } from "node:fs/promises";
import { dirname } from "node:path";

export interface SecretProtector {
  available(): boolean;
  encrypt(value: string): Buffer;
  decrypt(value: Buffer): string;
}

interface SecretDocument {
  format: "omni-encrypted-secrets";
  version: 1;
  records: Record<string, string>;
}

const SAFE_SECRET_ID = /^[a-zA-Z0-9][a-zA-Z0-9._:-]{0,255}$/;

function requireSecretId(id: string): string {
  if (!SAFE_SECRET_ID.test(id)) throw new Error("Invalid credential identifier.");
  return id;
}

function emptyDocument(): SecretDocument {
  return { format: "omni-encrypted-secrets", version: 1, records: {} };
}

/**
 * Application-level encrypted credentials. This file lives outside every
 * brain, origin snapshot, model checkpoint, journal, and .omni export.
 * Hosts without an OS credential backend keep secrets only for this process.
 */
export class SecureSecretStore {
  private readonly session = new Map<string, string>();
  private mutationQueue: Promise<void> = Promise.resolve();

  constructor(
    private readonly path: string,
    private readonly protector: SecretProtector
  ) {}

  persistence(): "encrypted" | "session" {
    return this.protector.available() ? "encrypted" : "session";
  }

  private serializeMutation<T>(operation: () => Promise<T>): Promise<T> {
    const result = this.mutationQueue.then(operation, operation);
    this.mutationQueue = result.then(() => undefined, () => undefined);
    return result;
  }

  private async readDocument(): Promise<SecretDocument> {
    try {
      const parsed = JSON.parse(await readFile(this.path, "utf8")) as Partial<SecretDocument>;
      if (
        parsed.format !== "omni-encrypted-secrets" ||
        parsed.version !== 1 ||
        typeof parsed.records !== "object" ||
        parsed.records === null ||
        Array.isArray(parsed.records)
      ) {
        throw new Error("The encrypted credential store is invalid.");
      }
      const records: Record<string, string> = {};
      for (const [id, value] of Object.entries(parsed.records)) {
        requireSecretId(id);
        if (typeof value !== "string" || value.length > 1_000_000) {
          throw new Error("The encrypted credential store is invalid.");
        }
        records[id] = value;
      }
      return { format: "omni-encrypted-secrets", version: 1, records };
    } catch (error) {
      if ((error as NodeJS.ErrnoException).code === "ENOENT") return emptyDocument();
      throw error;
    }
  }

  private async writeDocument(document: SecretDocument): Promise<void> {
    await mkdir(dirname(this.path), { recursive: true });
    const temporary = `${this.path}.${randomUUID()}.tmp`;
    try {
      await writeFile(temporary, JSON.stringify(document), { encoding: "utf8", mode: 0o600 });
      await chmod(temporary, 0o600).catch(() => undefined);
      await rename(temporary, this.path);
      await chmod(this.path, 0o600).catch(() => undefined);
    } finally {
      await rm(temporary, { force: true }).catch(() => undefined);
    }
  }

  async set(id: string, secret: string): Promise<"encrypted" | "session"> {
    const safeId = requireSecretId(id);
    const clean = secret.trim();
    if (!clean || clean.length > 64_000 || /[\r\n\0]/.test(clean)) {
      throw new Error("Credential value is invalid.");
    }
    if (!this.protector.available()) {
      this.session.set(safeId, clean);
      return "session";
    }
    return this.serializeMutation<"encrypted">(async () => {
      const document = await this.readDocument();
      document.records[safeId] = this.protector.encrypt(clean).toString("base64");
      await this.writeDocument(document);
      this.session.delete(safeId);
      return "encrypted" as const;
    });
  }

  async get(id: string): Promise<string | undefined> {
    const safeId = requireSecretId(id);
    const transient = this.session.get(safeId);
    if (transient !== undefined) return transient;
    if (!this.protector.available()) return undefined;
    await this.mutationQueue;
    const encoded = (await this.readDocument()).records[safeId];
    if (!encoded) return undefined;
    try {
      return this.protector.decrypt(Buffer.from(encoded, "base64"));
    } catch {
      throw new Error("This credential can no longer be decrypted by the current OS account.");
    }
  }

  async has(id: string): Promise<boolean> {
    return (await this.get(id)) !== undefined;
  }

  async remove(id: string): Promise<boolean> {
    const safeId = requireSecretId(id);
    let removed = this.session.delete(safeId);
    if (!this.protector.available()) return removed;
    return this.serializeMutation(async () => {
      const document = await this.readDocument();
      if (safeId in document.records) {
        delete document.records[safeId];
        await this.writeDocument(document);
        removed = true;
      }
      return removed;
    });
  }
}
