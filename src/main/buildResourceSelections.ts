import { randomUUID } from "node:crypto";
import { mkdir, readFile, rename, rm, writeFile } from "node:fs/promises";
import { dirname, isAbsolute } from "node:path";
import type { BuildResourceSelection } from "../shared/types";

const SAFE_ID = /^[a-zA-Z0-9][a-zA-Z0-9._-]{0,127}$/;

export interface StoredBuildResourceSelection extends BuildResourceSelection {
  paths: string[];
  createdAt: string;
  updatedAt: string;
  state: "selected" | "preparing" | "training" | "retryable" | "complete";
  brainId?: string;
  runtimeJobId?: string;
  manifestId?: string;
}

interface StoredSelectionDocument {
  schemaVersion: 1;
  selections: StoredBuildResourceSelection[];
}

function cloneSelection(
  selection: StoredBuildResourceSelection
): StoredBuildResourceSelection {
  return { ...selection, paths: [...selection.paths] };
}

function validateSelection(value: unknown): StoredBuildResourceSelection {
  if (typeof value !== "object" || value === null || Array.isArray(value)) {
    throw new Error("A pending Build resource is invalid.");
  }
  const selection = value as Record<string, unknown>;
  if (
    typeof selection.id !== "string" ||
    !SAFE_ID.test(selection.id) ||
    !["files", "folder"].includes(String(selection.kind)) ||
    typeof selection.label !== "string" ||
    !selection.label.trim() ||
    selection.label.length > 512 ||
    !Number.isSafeInteger(selection.itemCount) ||
    Number(selection.itemCount) < 1 ||
    (selection.bytes !== undefined &&
      (!Number.isSafeInteger(selection.bytes) || Number(selection.bytes) < 0)) ||
    (selection.fileCount !== undefined &&
      (!Number.isSafeInteger(selection.fileCount) || Number(selection.fileCount) < 0)) ||
    !Array.isArray(selection.paths) ||
    selection.paths.length !== selection.itemCount ||
    selection.paths.some(
      (path) =>
        typeof path !== "string" ||
        !isAbsolute(path) ||
        path.includes("\0")
    ) ||
    typeof selection.createdAt !== "string" ||
    !Number.isFinite(Date.parse(selection.createdAt)) ||
    typeof selection.updatedAt !== "string" ||
    !Number.isFinite(Date.parse(selection.updatedAt)) ||
    !["selected", "preparing", "training", "retryable", "complete"].includes(
      String(selection.state)
    ) ||
    (selection.brainId !== undefined &&
      (typeof selection.brainId !== "string" || !SAFE_ID.test(selection.brainId))) ||
    (selection.runtimeJobId !== undefined &&
      (typeof selection.runtimeJobId !== "string" || !SAFE_ID.test(selection.runtimeJobId))) ||
    (selection.manifestId !== undefined &&
      (typeof selection.manifestId !== "string" || !SAFE_ID.test(selection.manifestId)))
  ) {
    throw new Error("A pending Build resource is invalid.");
  }
  const state = selection.state as StoredBuildResourceSelection["state"];
  const brainId = selection.brainId as string | undefined;
  const runtimeJobId = selection.runtimeJobId as string | undefined;
  const manifestId = selection.manifestId as string | undefined;
  if (
    (state === "selected" && (brainId || runtimeJobId || manifestId)) ||
    (state !== "selected" && !brainId) ||
    (state === "retryable" && runtimeJobId) ||
    ((state === "training" || state === "complete") && !manifestId)
  ) {
    throw new Error("A pending Build resource has an invalid lifecycle.");
  }
  return {
    id: selection.id,
    kind: selection.kind as BuildResourceSelection["kind"],
    label: selection.label,
    itemCount: Number(selection.itemCount),
    ...(selection.bytes !== undefined ? { bytes: Number(selection.bytes) } : {}),
    ...(selection.fileCount !== undefined
      ? { fileCount: Number(selection.fileCount) }
      : {}),
    paths: [...selection.paths] as string[],
    createdAt: new Date(selection.createdAt).toISOString(),
    updatedAt: new Date(selection.updatedAt).toISOString(),
    state,
    ...(brainId ? { brainId } : {}),
    ...(runtimeJobId
      ? { runtimeJobId }
      : {}),
    ...(manifestId ? { manifestId } : {})
  };
}

/**
 * App-owned, non-neural staging metadata. Paths remain in the main process;
 * the renderer sees only labels/counts and stable selection IDs. Persisting
 * this small index prevents a renderer reload from silently dropping a large
 * folder before its deterministic manifest has been committed.
 */
export class BuildResourceSelectionStore {
  private readonly selections = new Map<string, StoredBuildResourceSelection>();
  private initialization?: Promise<void>;
  private writeQueue: Promise<void> = Promise.resolve();

  constructor(private readonly path: string) {}

  async list(): Promise<BuildResourceSelection[]> {
    await this.initialize();
    return [...this.selections.values()]
      .filter((selection) => selection.state === "selected")
      .sort((left, right) => left.createdAt.localeCompare(right.createdAt))
      .map(({
        paths: _paths,
        createdAt: _createdAt,
        updatedAt: _updatedAt,
        state: _state,
        brainId: _brainId,
        runtimeJobId: _runtimeJobId,
        manifestId: _manifestId,
        ...selection
      }) => ({
        ...selection
      }));
  }

  async listTasks(brainId?: string): Promise<StoredBuildResourceSelection[]> {
    await this.initialize();
    return [...this.selections.values()]
      .filter((selection) => !brainId || selection.brainId === brainId)
      .sort((left, right) => left.createdAt.localeCompare(right.createdAt))
      .map(cloneSelection);
  }

  async get(id: string): Promise<StoredBuildResourceSelection | undefined> {
    await this.initialize();
    const selection = this.selections.get(id);
    return selection ? cloneSelection(selection) : undefined;
  }

  async put(selection: StoredBuildResourceSelection): Promise<void> {
    await this.initialize();
    const validated = validateSelection(selection);
    this.selections.set(validated.id, validated);
    await this.persist();
  }

  async delete(id: string): Promise<boolean> {
    await this.initialize();
    const deleted = this.selections.delete(id);
    if (deleted) await this.persist();
    return deleted;
  }

  async claim(id: string, brainId: string): Promise<StoredBuildResourceSelection> {
    if (!SAFE_ID.test(brainId)) throw new Error("Invalid Build brain id.");
    return this.update(id, (selection) => {
      if (
        selection.state !== "selected" &&
        (selection.state !== "preparing" || selection.brainId !== brainId)
      ) {
        throw new Error("The Build resource is already claimed by another brain.");
      }
      return {
        ...selection,
        state: "preparing",
        brainId,
        updatedAt: new Date().toISOString()
      };
    });
  }

  async attachRuntimeJob(id: string, runtimeJobId: string): Promise<void> {
    if (!SAFE_ID.test(runtimeJobId)) throw new Error("Invalid Build runtime job id.");
    await this.update(id, (selection) => {
      if (selection.state === "complete") {
        throw new Error("The Build resource has already completed.");
      }
      return {
        ...selection,
        state: selection.manifestId ? "training" : "preparing",
        runtimeJobId,
        updatedAt: new Date().toISOString()
      };
    });
  }

  async commitManifest(id: string, manifestId: string): Promise<void> {
    if (!SAFE_ID.test(manifestId)) throw new Error("Invalid Build manifest id.");
    await this.update(id, (selection) => {
      if (!selection.brainId) {
        throw new Error("The Build resource must be claimed before its manifest is committed.");
      }
      return {
        ...selection,
        state: "training",
        manifestId,
        updatedAt: new Date().toISOString()
      };
    });
  }

  async complete(id: string): Promise<void> {
    await this.update(id, (selection) => {
      if (!selection.manifestId) {
        throw new Error("The Build resource has no committed manifest.");
      }
      return {
        ...selection,
        state: "complete",
        updatedAt: new Date().toISOString()
      };
    });
  }

  /**
   * Releases transient RuntimeJob ownership after a recoverable initialization
   * stop without discarding the source paths or committed manifest. The brain
   * keeps exclusive ownership, so a later retry can resume the same cursor but
   * no stale process-local job id is presented as active after restart/failure.
   */
  async releaseForRetry(id: string, brainId: string): Promise<void> {
    if (!SAFE_ID.test(brainId)) throw new Error("Invalid Build brain id.");
    await this.update(id, (selection) => {
      if (selection.state === "complete") return selection;
      if (selection.brainId && selection.brainId !== brainId) {
        throw new Error("The Build resource is claimed by another brain.");
      }
      const {
        runtimeJobId: _runtimeJobId,
        ...retained
      } = selection;
      return {
        ...retained,
        state: "retryable",
        brainId,
        updatedAt: new Date().toISOString()
      };
    });
  }

  private async initialize(): Promise<void> {
    if (!this.initialization) {
      this.initialization = this.load();
    }
    await this.initialization;
  }

  private async update(
    id: string,
    mutate: (selection: StoredBuildResourceSelection) => StoredBuildResourceSelection
  ): Promise<StoredBuildResourceSelection> {
    await this.initialize();
    const current = this.selections.get(id);
    if (!current) throw new Error("The selected Build resource is no longer available.");
    const next = validateSelection(mutate(cloneSelection(current)));
    this.selections.set(id, next);
    await this.persist();
    return cloneSelection(next);
  }

  private async load(): Promise<void> {
    let serialized: string;
    try {
      serialized = await readFile(this.path, "utf8");
    } catch (error) {
      if ((error as NodeJS.ErrnoException).code === "ENOENT") {
        try {
          serialized = await readFile(`${this.path}.bak`, "utf8");
        } catch (backupError) {
          if ((backupError as NodeJS.ErrnoException).code === "ENOENT") return;
          throw backupError;
        }
      } else {
        throw error;
      }
    }
    const document = JSON.parse(serialized) as unknown;
    if (
      typeof document !== "object" ||
      document === null ||
      Array.isArray(document) ||
      (document as { schemaVersion?: unknown }).schemaVersion !== 1 ||
      !Array.isArray((document as { selections?: unknown }).selections)
    ) {
      throw new Error("The pending Build resource index is invalid.");
    }
    for (const raw of (document as StoredSelectionDocument).selections) {
      const selection = validateSelection(raw);
      if (this.selections.has(selection.id)) {
        throw new Error("The pending Build resource index contains a duplicate ID.");
      }
      this.selections.set(selection.id, selection);
    }
  }

  private async persist(): Promise<void> {
    const snapshot: StoredSelectionDocument = {
      schemaVersion: 1,
      selections: [...this.selections.values()]
        .sort((left, right) => left.createdAt.localeCompare(right.createdAt))
        .map(cloneSelection)
    };
    this.writeQueue = this.writeQueue.then(async () => {
      await mkdir(dirname(this.path), { recursive: true });
      const temporary = `${this.path}.${randomUUID()}.next`;
      await writeFile(temporary, JSON.stringify(snapshot, null, 2), {
        encoding: "utf8",
        flag: "wx",
        mode: 0o600
      });
      try {
        await rename(temporary, this.path);
      } catch (error) {
        const code = (error as NodeJS.ErrnoException).code;
        if (code !== "EEXIST" && code !== "EPERM") {
          await rm(temporary, { force: true });
          throw error;
        }
        const backup = `${this.path}.bak`;
        await rm(backup, { force: true });
        await rename(this.path, backup);
        try {
          await rename(temporary, this.path);
          await rm(backup, { force: true });
        } catch (replacementError) {
          await rename(backup, this.path).catch(() => undefined);
          await rm(temporary, { force: true });
          throw replacementError;
        }
      }
    });
    await this.writeQueue;
  }
}
