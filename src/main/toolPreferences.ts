import { randomUUID } from "node:crypto";
import { chmod, mkdir, readFile, rename, rm, writeFile } from "node:fs/promises";
import { dirname } from "node:path";
import type { ToolRuntimePreferences } from "../shared/types";

const DEFAULT_PREFERENCES: ToolRuntimePreferences = { approvalTimeoutSeconds: 30 };

function normalize(value: unknown): ToolRuntimePreferences {
  const record = typeof value === "object" && value !== null && !Array.isArray(value)
    ? value as Record<string, unknown>
    : {};
  const seconds = Number(record.approvalTimeoutSeconds);
  return {
    approvalTimeoutSeconds:
      Number.isSafeInteger(seconds) && seconds >= 1 && seconds <= 3_600
        ? seconds
        : DEFAULT_PREFERENCES.approvalTimeoutSeconds
  };
}

export class ToolPreferencesStore {
  private value: ToolRuntimePreferences = { ...DEFAULT_PREFERENCES };

  constructor(private readonly path: string) {}

  async initialize(): Promise<ToolRuntimePreferences> {
    try {
      this.value = normalize(JSON.parse(await readFile(this.path, "utf8")));
    } catch (error) {
      if ((error as NodeJS.ErrnoException).code !== "ENOENT") throw error;
    }
    return this.get();
  }

  get(): ToolRuntimePreferences {
    return { ...this.value };
  }

  async set(value: ToolRuntimePreferences): Promise<ToolRuntimePreferences> {
    const normalized = normalize(value);
    if (normalized.approvalTimeoutSeconds !== Number(value?.approvalTimeoutSeconds)) {
      throw new Error("Approval timeout must be a whole number from 1 to 3,600 seconds.");
    }
    this.value = normalized;
    await mkdir(dirname(this.path), { recursive: true });
    const temporary = `${this.path}.${randomUUID()}.tmp`;
    try {
      await writeFile(temporary, JSON.stringify(this.value), { encoding: "utf8", mode: 0o600 });
      await chmod(temporary, 0o600).catch(() => undefined);
      await rename(temporary, this.path);
    } finally {
      await rm(temporary, { force: true }).catch(() => undefined);
    }
    return this.get();
  }
}
