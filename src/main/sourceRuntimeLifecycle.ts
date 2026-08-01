import { randomUUID } from "node:crypto";
import { spawn } from "node:child_process";
import {
  lstat,
  mkdir,
  readFile,
  readdir,
  realpath,
  rename,
  rm,
  symlink,
  unlink,
  writeFile
} from "node:fs/promises";
import {
  basename,
  dirname,
  isAbsolute,
  join,
  relative,
  resolve,
  sep
} from "node:path";
import {
  inspectRuntimeArtifacts,
  readRuntimeManifest,
  sha256File,
  type SourceRuntimeActivationResult,
  type SourceRuntimeLifecycle,
  type SourceRuntimeManifest,
  type SourceRuntimeScheduleRequest,
  type SourceRuntimeStageRequest,
  type SourceRuntimeStageResult
} from "./sourceRuntimeContract";

export interface ElectronRuntimeHost {
  isPackaged: boolean;
  relaunch(options: { execPath: string; args: string[] }): void;
  exit(exitCode?: number): void;
}

export interface ElectronSourceRuntimeOptions {
  app: ElectronRuntimeHost;
  userDataPath: string;
  resourcesPath: string;
}

const PROCESS_OUTPUT_LIMIT = 4 * 1024 * 1024;
const BUILD_TIMEOUT_MS = 20 * 60_000;

function inside(root: string, target: string): boolean {
  const fromRoot = relative(root, target);
  return fromRoot === "" || (!fromRoot.startsWith("..") && !isAbsolute(fromRoot));
}

async function atomicJson(path: string, value: unknown): Promise<void> {
  await mkdir(dirname(path), { recursive: true });
  const temporary = `${path}.${randomUUID()}.omni-next`;
  await writeFile(temporary, JSON.stringify(value, null, 2), {
    encoding: "utf8",
    flag: "wx",
    mode: 0o600
  });
  await rename(temporary, path);
}

function runProtectedProcess(
  executable: string,
  args: string[],
  cwd: string,
  environment: Record<string, string>,
  signal: AbortSignal
): Promise<void> {
  return new Promise((resolveProcess, rejectProcess) => {
    if (signal.aborted) {
      rejectProcess(new Error("Runtime staging was cancelled."));
      return;
    }
    const child = spawn(executable, args, {
      cwd,
      env: {
        ...process.env,
        ...environment,
        npm_config_ignore_scripts: "true",
        NPM_CONFIG_IGNORE_SCRIPTS: "true",
        CSC_IDENTITY_AUTO_DISCOVERY: "false"
      },
      shell: false,
      windowsHide: true,
      detached: process.platform !== "win32",
      stdio: ["ignore", "pipe", "pipe"]
    });
    let output = "";
    const append = (chunk: Buffer): void => {
      if (output.length >= PROCESS_OUTPUT_LIMIT) return;
      output += chunk.toString("utf8", 0, PROCESS_OUTPUT_LIMIT - output.length);
    };
    child.stdout.on("data", append);
    child.stderr.on("data", append);
    let settled = false;
    const stop = (): void => {
      if (child.exitCode !== null) return;
      if (process.platform === "win32" && child.pid) {
        spawn(
          "taskkill.exe",
          ["/PID", String(child.pid), "/T", "/F"],
          { shell: false, windowsHide: true, stdio: "ignore" }
        );
      } else if (child.pid) {
        try {
          process.kill(-child.pid, "SIGTERM");
        } catch {
          child.kill("SIGTERM");
        }
      }
    };
    const timer = setTimeout(stop, BUILD_TIMEOUT_MS);
    const abort = (): void => stop();
    signal.addEventListener("abort", abort, { once: true });
    const finish = (error?: Error): void => {
      if (settled) return;
      settled = true;
      clearTimeout(timer);
      signal.removeEventListener("abort", abort);
      if (error) rejectProcess(error);
      else resolveProcess();
    };
    child.once("error", (error) => finish(error));
    child.once("close", (code) => {
      if (signal.aborted) {
        finish(new Error("Runtime staging was cancelled."));
      } else if (code !== 0) {
        finish(
          new Error(
            `Protected native runtime build failed with exit code ${code ?? -1}: ${output.slice(-8_000)}`
          )
        );
      } else {
        finish();
      }
    });
  });
}

async function findExecutable(
  root: string,
  platform: SourceRuntimeStageRequest["platform"]
): Promise<string> {
  const candidates: string[] = [];
  const visit = async (directory: string): Promise<void> => {
    for (const entry of await readdir(directory, { withFileTypes: true })) {
      const path = join(directory, entry.name);
      if (entry.isDirectory()) {
        await visit(path);
      } else if (entry.isFile()) {
        const normalized = path.split(sep).join("/");
        const matches =
          platform === "win32"
            ? entry.name === "Omni AGI Studio.exe"
            : platform === "linux"
              ? entry.name === "omni-agi-studio"
              : normalized.endsWith(
                  "/Omni AGI Studio.app/Contents/MacOS/Omni AGI Studio"
                );
        if (matches) candidates.push(path);
      }
      if (candidates.length > 2) break;
    }
  };
  await visit(root);
  if (candidates.length !== 1) {
    throw new Error(
      `Native runtime staging expected one desktop executable and found ${candidates.length}.`
    );
  }
  return candidates[0]!;
}

function unpackedRoot(
  executablePath: string,
  platform: SourceRuntimeStageRequest["platform"]
): string {
  if (platform !== "darwin") return dirname(executablePath);
  const marker = `${sep}Omni AGI Studio.app${sep}`;
  const index = executablePath.indexOf(marker);
  if (index < 0) throw new Error("The staged macOS executable is outside its app bundle.");
  return executablePath.slice(0, index + `${sep}Omni AGI Studio.app`.length);
}

export class ElectronSourceRuntimeLifecycle implements SourceRuntimeLifecycle {
  private readonly timers = new Map<string, NodeJS.Timeout>();

  constructor(private readonly options: ElectronSourceRuntimeOptions) {}

  private runtimeRoot(): string {
    return join(this.options.userDataPath, "source-runtimes");
  }

  async stage(
    request: SourceRuntimeStageRequest,
    signal: AbortSignal
  ): Promise<SourceRuntimeStageResult> {
    if (request.platform !== process.platform || request.architecture !== process.arch) {
      throw new Error("Source runtime staging must target the native host architecture.");
    }
    const [repository, worktree, currentExecutable] = await Promise.all([
      realpath(request.authorizedRepository),
      realpath(request.worktree),
      realpath(request.currentExecutablePath)
    ]);
    if (inside(repository, worktree) || repository === worktree) {
      throw new Error("Source runtime staging requires an isolated sibling worktree.");
    }
    if (!request.changedPaths.length) {
      throw new Error("A runtime cannot be staged for an empty source candidate.");
    }
    const slotId = `source-${Date.now()}-${randomUUID().slice(0, 12)}`;
    const runtimeRoot = this.runtimeRoot();
    await mkdir(runtimeRoot, { recursive: true });
    const managedRoot = await realpath(runtimeRoot);
    const temporaryRoot = join(managedRoot, `.${slotId}.omni-next`);
    const finalRoot = join(managedRoot, slotId);
    const buildOutput = join(temporaryRoot, "builder-output");
    const desktopRoot = join(temporaryRoot, "desktop");
    const engineChanged = request.changedPaths.some(
      (path) =>
        path === "engine" ||
        path.startsWith("engine/")
    );
    const engineStrategy = engineChanged
      ? "rebuilt-protected-worker"
      : "reused-current-worker";
    let linkedWorkerPath: string | undefined;
    try {
      await mkdir(buildOutput, { recursive: true });
      if (engineChanged) {
        const builder =
          request.platform === "win32"
            ? join(worktree, "scripts", "build-engine.ps1")
            : join(worktree, "scripts", "build-engine-posix.sh");
        const builderInfo = await lstat(builder);
        if (!builderInfo.isFile() || builderInfo.isSymbolicLink()) {
          throw new Error("The protected local worker builder is unavailable.");
        }
        if (request.platform === "win32") {
          await runProtectedProcess(
            process.env.ComSpec?.trim() || "powershell.exe",
            process.env.ComSpec?.trim()
              ? [
                  "/d",
                  "/c",
                  "powershell.exe",
                  "-NoLogo",
                  "-NoProfile",
                  "-ExecutionPolicy",
                  "Bypass",
                  "-File",
                  builder
                ]
              : [
                  "-NoLogo",
                  "-NoProfile",
                  "-ExecutionPolicy",
                  "Bypass",
                  "-File",
                  builder
                ],
            worktree,
            { OMNI_SKIP_BUILD_DEPENDENCY_INSTALL: "1" },
            signal
          );
        } else {
          await runProtectedProcess(
            "bash",
            [
              builder,
              request.platform === "darwin" ? "mac" : "linux"
            ],
            worktree,
            { OMNI_SKIP_BUILD_DEPENDENCY_INSTALL: "1" },
            signal
          );
        }
      } else {
        const candidateWorkerRoots = [
          join(this.options.resourcesPath, "engine-runtime"),
          join(repository, "engine-dist", "omni-engine")
        ];
        let currentWorkerRoot: string | undefined;
        for (const candidate of candidateWorkerRoots) {
          try {
            const info = await lstat(candidate);
            if (!info.isDirectory() || info.isSymbolicLink()) continue;
            currentWorkerRoot = await realpath(candidate);
            break;
          } catch {
            // Try the next app-owned/authorized worker location.
          }
        }
        if (!currentWorkerRoot) {
          throw new Error(
            "The current protected neural worker is unavailable for runtime reuse."
          );
        }
        linkedWorkerPath = join(worktree, "engine-dist", "omni-engine");
        await mkdir(dirname(linkedWorkerPath), { recursive: true });
        try {
          await lstat(linkedWorkerPath);
          throw new Error("The candidate already contains an engine runtime output.");
        } catch (error) {
          if ((error as NodeJS.ErrnoException).code !== "ENOENT") throw error;
        }
        await symlink(
          currentWorkerRoot,
          linkedWorkerPath,
          request.platform === "win32" ? "junction" : "dir"
        );
        if (
          resolve(await realpath(linkedWorkerPath)) !== resolve(currentWorkerRoot)
        ) {
          throw new Error("The reused worker link escaped its protected source.");
        }
      }

      const builderCli = join(
        repository,
        "node_modules",
        "electron-builder",
        "out",
        "cli",
        "cli.js"
      );
      const cliInfo = await lstat(builderCli);
      if (!cliInfo.isFile() || cliInfo.isSymbolicLink()) {
        throw new Error("The authorized electron-builder executable is unavailable.");
      }
      const targetFlag =
        request.platform === "win32"
          ? "--win"
          : request.platform === "darwin"
            ? "--mac"
            : "--linux";
      await runProtectedProcess(
        process.execPath,
        [
          builderCli,
          "--dir",
          targetFlag,
          `--${request.architecture}`,
          "--publish",
          "never",
          `--config.directories.output=${buildOutput}`,
          "--config.forceCodeSigning=false",
          ...(request.platform === "darwin"
            ? ["--config.mac.notarize=false"]
            : [])
        ],
        worktree,
        { ELECTRON_RUN_AS_NODE: "1" },
        signal
      );
      const builtExecutable = await findExecutable(buildOutput, request.platform);
      const builtRoot = unpackedRoot(builtExecutable, request.platform);
      const executableSuffix = relative(builtRoot, builtExecutable);
      if (
        !executableSuffix ||
        executableSuffix.startsWith("..") ||
        isAbsolute(executableSuffix)
      ) {
        throw new Error("The packaged executable escaped its unpacked runtime.");
      }
      await rename(builtRoot, desktopRoot);
      await rm(buildOutput, { recursive: true, force: true });
      const executablePath = join(desktopRoot, executableSuffix);
      const currentExecutableHash = await sha256File(currentExecutable);
      const executableHash = await sha256File(executablePath);
      const inspected = await inspectRuntimeArtifacts(temporaryRoot);
      const executableRelativePath = relative(temporaryRoot, executablePath)
        .split(sep)
        .join("/");
      const manifest: SourceRuntimeManifest = {
        schemaVersion: 1,
        slotId,
        createdAt: new Date().toISOString(),
        state: "staged",
        platform: request.platform,
        architecture: request.architecture,
        lineage: {
          parentCommit: request.parentCommit,
          diffSha256: request.diffSha256,
          evaluatorSha256: request.evaluatorSha256,
          brainSnapshotId: request.brainSnapshotId
        },
        executableRelativePath,
        executableSha256: executableHash.sha256,
        currentExecutableSha256: currentExecutableHash.sha256,
        artifactSha256: inspected.artifactSha256,
        artifacts: inspected.artifacts,
        engineStrategy
      };
      await atomicJson(join(temporaryRoot, "runtime-manifest.json"), manifest);
      await rename(temporaryRoot, finalRoot);
      const finalManifestPath = join(finalRoot, "runtime-manifest.json");
      const finalExecutablePath = join(finalRoot, executableRelativePath);
      return {
        slotId,
        rootPath: finalRoot,
        manifestPath: finalManifestPath,
        manifestSha256: (await sha256File(finalManifestPath)).sha256,
        executablePath: finalExecutablePath,
        manifest
      };
    } catch (error) {
      await rm(temporaryRoot, { recursive: true, force: true }).catch(() => undefined);
      throw error;
    } finally {
      if (linkedWorkerPath) {
        const info = await lstat(linkedWorkerPath).catch(() => undefined);
        if (info?.isSymbolicLink()) {
          await unlink(linkedWorkerPath).catch(() => undefined);
        }
      }
      if (engineChanged) {
        await Promise.all([
          rm(join(worktree, ".engine-build"), { recursive: true, force: true }),
          rm(join(worktree, "engine-dist"), { recursive: true, force: true })
        ]).catch(() => undefined);
      }
    }
  }

  async scheduleActivation(
    request: SourceRuntimeScheduleRequest
  ): Promise<SourceRuntimeActivationResult> {
    const runtimeRoot = await realpath(this.runtimeRoot());
    const [slotRoot, executablePath, manifestPath] = await Promise.all([
      realpath(request.stage.rootPath),
      realpath(request.stage.executablePath),
      realpath(request.stage.manifestPath)
    ]);
    if (
      !inside(runtimeRoot, slotRoot) ||
      !inside(slotRoot, executablePath) ||
      !inside(slotRoot, manifestPath) ||
      request.stage.slotId !== basename(slotRoot) ||
      request.delayMs < 3_000
    ) {
      throw new Error("The runtime activation slot is outside app-managed storage.");
    }
    const manifest = await readRuntimeManifest(manifestPath);
    if (
      manifest.state !== "staged" ||
      manifest.slotId !== request.stage.slotId ||
      manifest.lineage.diffSha256 !== request.stage.manifest.lineage.diffSha256
    ) {
      throw new Error("The staged runtime manifest changed before activation.");
    }
    const state = this.options.app.isPackaged ? "scheduled" : "deferred";
    const reason =
      state === "deferred"
        ? "Development/test hosts preserve the verified slot but do not relaunch Electron."
        : undefined;
    const scheduledAt = new Date().toISOString();
    const updated: SourceRuntimeManifest = {
      ...manifest,
      state,
      promotionCommit: request.promotionCommit,
      candidateCommit: request.candidateCommit,
      activation: {
        state,
        scheduledAt,
        delayMs: request.delayMs,
        reason
      }
    };
    await atomicJson(manifestPath, updated);
    if (state === "scheduled") {
      const timer = setTimeout(() => {
        this.timers.delete(request.stage.slotId);
        try {
          this.options.app.relaunch({
            execPath: executablePath,
            args: [`--omni-runtime-slot=${request.stage.slotId}`]
          });
          this.options.app.exit(0);
        } catch (error) {
          console.error("Failed to relaunch staged Omni runtime:", error);
        }
      }, request.delayMs);
      timer.unref();
      this.timers.set(request.stage.slotId, timer);
    }
    return {
      state,
      slotId: request.stage.slotId,
      executablePath,
      manifestPath,
      manifestSha256: (await sha256File(manifestPath)).sha256,
      promotionCommit: request.promotionCommit,
      delayMs: request.delayMs,
      reason
    };
  }

  async abandonStage(stage: SourceRuntimeStageResult, reason: string): Promise<void> {
    const timer = this.timers.get(stage.slotId);
    if (timer) {
      clearTimeout(timer);
      this.timers.delete(stage.slotId);
    }
    const runtimeRoot = await realpath(this.runtimeRoot());
    const slotRoot = await realpath(stage.rootPath);
    if (!inside(runtimeRoot, slotRoot) || basename(slotRoot) !== stage.slotId) {
      throw new Error("Cannot abandon a runtime outside app-managed storage.");
    }
    const manifest = await readRuntimeManifest(stage.manifestPath);
    await atomicJson(stage.manifestPath, {
      ...manifest,
      state: "abandoned",
      abandonedReason: reason.slice(0, 2_000)
    } satisfies SourceRuntimeManifest);
  }
}
