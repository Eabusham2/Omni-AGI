import { createHash } from "node:crypto";
import { createReadStream } from "node:fs";
import {
  lstat,
  readFile,
  readdir,
  readlink,
  realpath
} from "node:fs/promises";
import {
  dirname,
  isAbsolute,
  relative,
  resolve
} from "node:path";

export type SourceRuntimePlatform = "win32" | "darwin" | "linux";
export type SourceRuntimeArchitecture = "x64" | "arm64";

export interface SourceRuntimeArtifact {
  path: string;
  kind: "file" | "symlink";
  sha256: string;
  bytes: number;
  target?: string;
}

export interface SourceRuntimeLineage {
  parentCommit: string;
  diffSha256: string;
  evaluatorSha256: string;
  brainSnapshotId: string;
}

export interface SourceRuntimeManifest {
  schemaVersion: 1;
  slotId: string;
  createdAt: string;
  state: "staged" | "scheduled" | "deferred" | "abandoned";
  platform: SourceRuntimePlatform;
  architecture: SourceRuntimeArchitecture;
  lineage: SourceRuntimeLineage;
  executableRelativePath: string;
  executableSha256: string;
  currentExecutableSha256: string;
  artifactSha256: string;
  artifacts: SourceRuntimeArtifact[];
  engineStrategy: "reused-current-worker" | "rebuilt-protected-worker";
  promotionCommit?: string;
  candidateCommit?: string;
  activation?: {
    state: "scheduled" | "deferred";
    scheduledAt: string;
    delayMs: number;
    reason?: string;
  };
  abandonedReason?: string;
}

export interface SourceRuntimeStageRequest {
  brainId: string;
  authorizedRepository: string;
  worktree: string;
  parentCommit: string;
  diffSha256: string;
  evaluatorSha256: string;
  changedPaths: string[];
  brainSnapshotId: string;
  currentExecutablePath: string;
  platform: SourceRuntimePlatform;
  architecture: SourceRuntimeArchitecture;
}

export interface SourceRuntimeStageResult {
  slotId: string;
  rootPath: string;
  manifestPath: string;
  manifestSha256: string;
  executablePath: string;
  manifest: SourceRuntimeManifest;
}

export interface SourceRuntimeScheduleRequest {
  stage: SourceRuntimeStageResult;
  promotionCommit: string;
  candidateCommit: string;
  delayMs: number;
}

export interface SourceRuntimeActivationResult {
  state: "scheduled" | "deferred";
  slotId: string;
  executablePath: string;
  manifestPath: string;
  manifestSha256: string;
  promotionCommit: string;
  delayMs: number;
  reason?: string;
}

export interface SourceRuntimeLifecycle {
  stage(
    request: SourceRuntimeStageRequest,
    signal: AbortSignal
  ): Promise<SourceRuntimeStageResult>;
  scheduleActivation(
    request: SourceRuntimeScheduleRequest
  ): Promise<SourceRuntimeActivationResult>;
  abandonStage?(stage: SourceRuntimeStageResult, reason: string): Promise<void>;
}

export function normalizeRuntimeRelativePath(value: string): string {
  if (
    !value ||
    value !== value.trim() ||
    value.length > 4_096 ||
    value.includes("\\") ||
    value.startsWith("/") ||
    /^[a-z]:/i.test(value) ||
    /[\0-\x1f\x7f]/.test(value)
  ) {
    throw new Error("Runtime artifact paths must use safe portable relative syntax.");
  }
  const segments = value.split("/");
  if (
    segments.some(
      (segment) =>
        !segment ||
        segment === "." ||
        segment === ".."
    )
  ) {
    throw new Error("Runtime artifact path traversal is not allowed.");
  }
  return segments.join("/");
}

export async function sha256File(path: string): Promise<{
  sha256: string;
  bytes: number;
}> {
  const info = await lstat(path);
  if (!info.isFile() || info.isSymbolicLink()) {
    throw new Error("Runtime artifact hashing requires a regular file.");
  }
  const hash = createHash("sha256");
  let bytes = 0;
  await new Promise<void>((resolveStream, rejectStream) => {
    const stream = createReadStream(path);
    stream.on("data", (chunk: string | Buffer) => {
      const contents = typeof chunk === "string" ? Buffer.from(chunk) : chunk;
      hash.update(contents);
      bytes += contents.byteLength;
    });
    stream.once("error", rejectStream);
    stream.once("end", resolveStream);
  });
  if (bytes !== info.size) {
    throw new Error("Runtime artifact changed while it was being hashed.");
  }
  return { sha256: hash.digest("hex"), bytes };
}

function inside(root: string, target: string): boolean {
  const fromRoot = relative(root, target);
  return fromRoot === "" || (!fromRoot.startsWith("..") && !isAbsolute(fromRoot));
}

export async function inspectRuntimeArtifacts(
  rootPath: string,
  excludedPaths: readonly string[] = ["runtime-manifest.json"]
): Promise<{
  rootPath: string;
  artifacts: SourceRuntimeArtifact[];
  artifactSha256: string;
}> {
  const root = await realpath(rootPath);
  const excluded = new Set(excludedPaths.map(normalizeRuntimeRelativePath));
  const artifacts: SourceRuntimeArtifact[] = [];
  let totalBytes = 0;
  const walk = async (directory: string, prefix: string): Promise<void> => {
    const entries = (await readdir(directory, { withFileTypes: true })).sort(
      (left, right) => left.name.localeCompare(right.name)
    );
    for (const entry of entries) {
      const path = resolve(directory, entry.name);
      const relativePath = normalizeRuntimeRelativePath(
        prefix ? `${prefix}/${entry.name}` : entry.name
      );
      if (excluded.has(relativePath)) continue;
      const info = await lstat(path);
      if (info.isSymbolicLink()) {
        const target = await readlink(path);
        if (isAbsolute(target)) {
          throw new Error(`Runtime artifact contains an absolute symlink: ${relativePath}`);
        }
        const resolvedTarget = await realpath(resolve(dirname(path), target));
        if (!inside(root, resolvedTarget)) {
          throw new Error(`Runtime artifact symlink escapes its slot: ${relativePath}`);
        }
        const targetBytes = Buffer.byteLength(target, "utf8");
        artifacts.push({
          path: relativePath,
          kind: "symlink",
          sha256: createHash("sha256").update(target).digest("hex"),
          bytes: targetBytes,
          target
        });
        totalBytes += targetBytes;
      } else if (info.isDirectory()) {
        await walk(path, relativePath);
      } else if (info.isFile()) {
        const hashed = await sha256File(path);
        artifacts.push({
          path: relativePath,
          kind: "file",
          ...hashed
        });
        totalBytes += hashed.bytes;
      } else {
        throw new Error(`Runtime slot contains a special filesystem entry: ${relativePath}`);
      }
      if (artifacts.length > 100_000 || totalBytes > 32 * 1024 * 1024 * 1024) {
        throw new Error("Runtime slot exceeds artifact verification resource limits.");
      }
    }
  };
  await walk(root, "");
  artifacts.sort((left, right) => left.path.localeCompare(right.path));
  return {
    rootPath: root,
    artifacts,
    artifactSha256: createHash("sha256")
      .update(JSON.stringify(artifacts))
      .digest("hex")
  };
}

export async function readRuntimeManifest(
  path: string
): Promise<SourceRuntimeManifest> {
  return JSON.parse(await readFile(path, "utf8")) as SourceRuntimeManifest;
}
