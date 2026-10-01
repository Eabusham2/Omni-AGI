import { createReadStream } from "node:fs";
import { createHash } from "node:crypto";
import { lstat, readFile, readdir } from "node:fs/promises";
import { join } from "node:path";
import { freemem } from "node:os";
import { getHeapStatistics } from "node:v8";

const id = "[a-f0-9]{32}";
const baseline = new RegExp(`^evolution-baselines/${id}(?:\\.json|\\.safetensors|\\.trained\\.json|\\.evaluation\\.json|\\.paired-(?:baseline|candidate)\\.jsonl)$`);
const candidate = new RegExp(`^candidates/${id}/(?:candidate\\.json|geometry-migration\\.json|(?:stable|geometry-insertion|model/engine)/[A-Za-z0-9._/-]+)$`);

/** Saved lineage only. No imported record authorizes execution or promotion. */
export function safeEvolutionContinuationPath(relative: string): boolean {
  return !relative.split("/").some(part => !part || part === "." || part === "..") &&
    (baseline.test(relative) || candidate.test(relative)) &&
    !/\.(?:exe|dll|com|bat|cmd|ps1|msi|scr|js|py|pyc|so|dylib)$/i.test(relative);
}

async function metadata(path: string): Promise<Record<string, unknown>> {
  const info = await lstat(path);
  if (!info.isFile() || info.isSymbolicLink()) throw new Error("Evolution metadata is not a regular owned file.");
  const need = info.size * 12, reserve = 64 * 1024 * 1024;
  // Main-process admission is cooperative, not an invented second RAM lease.
  if (getHeapStatistics().heap_size_limit - process.memoryUsage().heapUsed - reserve < need || freemem() - reserve < need) {
    throw new Error("Saved evolution metadata awaits physical transfer RAM.");
  }
  const value = JSON.parse(await readFile(path, "utf8")) as Record<string, unknown>;
  if (!value || typeof value !== "object" || Array.isArray(value)) throw new Error("Saved evolution metadata must be an explicit object.");
  return value;
}
async function fileHash(path: string): Promise<string> {
  const hash = createHash("sha256");
  for await (const bytes of createReadStream(path, { highWaterMark: 64 * 1024 })) hash.update(bytes);
  return hash.digest("hex");
}

export async function validateEvolutionContinuationFiles(files: ReadonlyMap<string, string>): Promise<void> {
  for (const [relative, path] of files) {
    if (!safeEvolutionContinuationPath(relative)) throw new Error("Unsafe saved evolution continuation path.");
    const info = await lstat(path);
    if (!info.isFile() || info.isSymbolicLink()) throw new Error("Evolution continuation contains a filesystem link or nonfile.");
    if (relative.endsWith("/candidate.json")) {
      const record = await metadata(path);
      if (record.kind !== "neural-evolution" || record.id !== relative.split("/")[1]) throw new Error("Saved native candidate identity is invalid.");
    } else if (relative.startsWith("evolution-baselines/") && relative.endsWith(".json")) {
      const record = await metadata(path);
      const candidateId = relative.split("/")[1]!.slice(0, 32);
      if (record.candidateId !== candidateId) throw new Error("Saved evolution baseline/measurement identity is invalid.");
      if (relative === `evolution-baselines/${candidateId}.json`) {
        const tensors = files.get(`evolution-baselines/${candidateId}.safetensors`);
        if (!tensors || await fileHash(tensors) !== record.anchorTensorSha256) throw new Error("Saved evolution baseline tensor binding is invalid.");
      }
      const holdout = record.geometryHoldouts as { pairedScores?: { path?: unknown; sha256?: unknown } } | undefined;
      if (holdout?.pairedScores) {
        const scores = holdout.pairedScores;
        const target = typeof scores.path === "string" ? `evolution-baselines/${scores.path}` : "";
        const source = files.get(target);
        if (!/^evolution-baselines\/[a-f0-9]{32}\.paired-(?:baseline|candidate)\.jsonl$/.test(target) ||
            !target.startsWith(`evolution-baselines/${candidateId}.`) || !source || await fileHash(source) !== scores.sha256) {
          throw new Error("Saved geometry paired observations are unavailable or changed.");
        }
      }
    }
  }
}

export async function savedEvolutionContinuationFiles(engineDirectory: string): Promise<Map<string, string>> {
  const files = new Map<string, string>();
  async function visit(path: string, prefix: string): Promise<void> {
    const detail = await lstat(path).catch((error: NodeJS.ErrnoException) => { if (error.code === "ENOENT") return undefined; throw error; });
    if (!detail) return;
    if (!detail.isDirectory() || detail.isSymbolicLink()) throw new Error("Evolution continuation root is unsafe.");
    for (const entry of (await readdir(path, { withFileTypes: true })).sort((a, b) => a.name.localeCompare(b.name))) {
      const relative = `${prefix}/${entry.name}`, source = join(path, entry.name);
      if (entry.isSymbolicLink()) throw new Error("Evolution continuation contains a filesystem link.");
      if (entry.isDirectory()) {
        if (prefix === "evolution-baselines" || prefix === "candidates" && !/^[a-f0-9]{32}$/.test(entry.name)) throw new Error("Evolution continuation directory is invalid.");
        await visit(source, relative);
      } else if (entry.isFile()) {
        if (/(?:-wal|-shm|-journal)$/.test(entry.name)) {
          if ((await lstat(source)).size) throw new Error("Saved evolution state is not quiescent; live SQLite sidecar cannot be omitted.");
          continue;
        }
        if (entry.name.endsWith(".tmp") || entry.name.endsWith(".omni-next")) continue;
        if (!safeEvolutionContinuationPath(relative)) throw new Error("Evolution continuation file is invalid.");
        files.set(relative, source);
      } else throw new Error("Evolution continuation is not a regular saved file.");
    }
  }
  for (const root of ["evolution-baselines", "candidates"]) await visit(join(engineDirectory, root), root);
  await validateEvolutionContinuationFiles(files);
  return files;
}

export async function savedEvolutionArchive(brainDirectory: string): Promise<string | undefined> {
  const directory = join(brainDirectory, "evolution");
  const info = await lstat(directory).catch((error: NodeJS.ErrnoException) => { if (error.code === "ENOENT") return undefined; throw error; });
  if (!info) return undefined;
  if (!info.isDirectory() || info.isSymbolicLink()) throw new Error("Saved evolution archive directory is unsafe.");
  const path = join(directory, "archive.json");
  if (!await lstat(path).catch((error: NodeJS.ErrnoException) => { if (error.code === "ENOENT") return undefined; throw error; })) return undefined;
  await validateSavedEvolutionArchive(path);
  return path;
}
export async function validateSavedEvolutionArchive(path: string): Promise<void> {
  const archive = await metadata(path);
  if (archive.schemaVersion !== 1 || !Array.isArray(archive.runs) || !Array.isArray(archive.candidates)) throw new Error("Saved main evolution archive is invalid.");
}
