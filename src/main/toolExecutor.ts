import { createHash, randomUUID } from "node:crypto";
import { spawn, type ChildProcess } from "node:child_process";
import { isUtf8 } from "node:buffer";
import {
  lstat,
  mkdir,
  readFile,
  readdir,
  realpath,
  rename,
  stat,
  symlink,
  unlink,
  writeFile
} from "node:fs/promises";
import {
  basename,
  dirname,
  extname,
  isAbsolute,
  join,
  relative,
  resolve
} from "node:path";
import type {
  ToolExecutionResult,
  ToolInvocation,
  RuntimeJob,
  ToolPermissionLevel
} from "../shared/types";
import {
  assertSafeRemoteUrl,
  readResponseBounded,
  safeFetch,
  type BrainService,
  type RuntimeJobManager
} from "./brainService";
import {
  EVOLUTION_BENCHMARK_DOMAINS,
  EVOLUTION_EVALUATOR_VERSION,
  EVOLUTION_POLICY_SHA256,
  EVOLUTION_PROTECTED_PATHS,
  EVOLUTION_TEST_NAMES,
  isProtectedEvolutionPath
} from "./evolutionPolicy";
import {
  inspectRuntimeArtifacts,
  normalizeRuntimeRelativePath,
  readRuntimeManifest,
  sha256File,
  type SourceRuntimeActivationResult,
  type SourceRuntimeLifecycle,
  type SourceRuntimeManifest,
  type SourceRuntimeStageResult
} from "./sourceRuntimeContract";
import { withBrainWrite } from "./brainWriteCoordinator";

const MAX_TEXT_BYTES = 8 * 1024 * 1024;
const MAX_PROCESS_OUTPUT = 2 * 1024 * 1024;
const MAX_WEB_BYTES = 8 * 1024 * 1024;
const MAX_SOURCE_EDIT_TOTAL_BYTES = 8 * 1024 * 1024;
const MAX_SOURCE_EDIT_COUNT = 256;
// Subagents report a focused result into the parent workspace. Their response
// length is deliberately independent of the brain's much larger working-memory
// context so hardware-scaled context growth does not multiply CPU latency for
// every isolated fork.
const SUBAGENT_RESPONSE_TOKENS = 96;

const SOURCE_TEXT_EXTENSIONS = new Set([
  ".c",
  ".cc",
  ".cpp",
  ".css",
  ".go",
  ".h",
  ".hpp",
  ".html",
  ".java",
  ".js",
  ".json",
  ".jsx",
  ".kt",
  ".md",
  ".mjs",
  ".mts",
  ".py",
  ".rs",
  ".scss",
  ".toml",
  ".ts",
  ".tsx",
  ".txt",
  ".yaml",
  ".yml"
]);

const SOURCE_EDIT_PROTECTED_NAMES = new Set([
  "package.json",
  "package-lock.json",
  "pnpm-lock.yaml",
  "yarn.lock"
]);

const SOURCE_EDIT_PROTECTED_SEGMENTS = new Set([
  ".git",
  ".github",
  ".openai",
  "build",
  "dist",
  "node_modules",
  "out",
  "release",
  "scripts"
]);

interface Approval {
  brainId: string;
  toolId: string;
  action: string;
  argumentSha256: string;
  expiresAt: number;
}

interface EvolutionProposalRecord {
  schemaVersion: 1;
  worktree: string;
  branch: string;
  parentCommit: string;
  evaluatorVersion: number;
  evaluatorSha256: string;
  baselineTestPaths: string[];
  packageScripts: Record<string, string>;
  sourceEditLineage: SourceEditLineageRecord[];
  authoredChangedPaths: string[];
  authoredDiffSha256: string;
  authoredBytes: number;
  createdAt: string;
}

interface SourceEditRequest {
  path: string;
  content: string;
  expectedSha256: string | null;
}

interface SourceEditLineageRecord {
  path: string;
  expectedSha256: string | null;
  resultSha256: string;
  bytes: number;
}

function sha256(contents: Buffer | string): string {
  return createHash("sha256").update(contents).digest("hex");
}

function argumentString(
  argumentsValue: Record<string, unknown>,
  key: string,
  maximum = 100_000
): string {
  const value = argumentsValue[key];
  if (typeof value !== "string" || !value.trim()) throw new Error(`${key} must be a non-empty string.`);
  return value.replace(/\0/g, "").slice(0, maximum);
}

function absolutePath(value: string): string {
  if (!isAbsolute(value)) throw new Error("Tool paths must be absolute.");
  return resolve(value);
}

async function atomicWrite(path: string, contents: string): Promise<void> {
  await mkdir(dirname(path), { recursive: true });
  const temporary = `${path}.${randomUUID()}.omni-next`;
  await writeFile(temporary, contents, { encoding: "utf8", flag: "wx", mode: 0o600 });
  await rename(temporary, path);
}

function normalizeSourceEditPath(value: unknown): string {
  if (typeof value !== "string" || !value || value !== value.trim()) {
    throw new Error("Each source edit path must be a non-empty relative path.");
  }
  if (
    value.length > 1_024 ||
    value.includes("\\") ||
    value.startsWith("/") ||
    /^[a-z]:/i.test(value) ||
    /[\0-\x1f\x7f:]/.test(value)
  ) {
    throw new Error("Source edit paths must use safe portable relative syntax.");
  }
  const segments = value.split("/");
  if (
    segments.some(
      (segment) =>
        !segment ||
        segment === "." ||
        segment === ".." ||
        segment.endsWith(" ") ||
        segment.endsWith(".")
    )
  ) {
    throw new Error("Source edit paths may not traverse or contain ambiguous segments.");
  }
  const lowerSegments = segments.map((segment) => segment.toLocaleLowerCase());
  if (lowerSegments.some((segment) => SOURCE_EDIT_PROTECTED_SEGMENTS.has(segment))) {
    throw new Error("Source edits may not target build, release, script, or control directories.");
  }
  const name = lowerSegments.at(-1)!;
  if (SOURCE_EDIT_PROTECTED_NAMES.has(name)) {
    throw new Error("Source edits may not rewrite package installation or execution manifests.");
  }
  if (!SOURCE_TEXT_EXTENSIONS.has(extname(name))) {
    throw new Error("Source evolution accepts UTF-8 source and documentation text files only.");
  }
  return segments.join("/");
}

function sourceEditRequests(value: unknown): SourceEditRequest[] {
  if (value === undefined) return [];
  if (!Array.isArray(value)) throw new Error("sourceEdits must be an array.");
  if (value.length > MAX_SOURCE_EDIT_COUNT) {
    throw new Error(`A source proposal may author at most ${MAX_SOURCE_EDIT_COUNT} files.`);
  }
  const edits: SourceEditRequest[] = [];
  const paths = new Set<string>();
  let totalBytes = 0;
  for (const entry of value) {
    if (typeof entry !== "object" || entry === null || Array.isArray(entry)) {
      throw new Error("Every source edit must be a typed object.");
    }
    const record = entry as Record<string, unknown>;
    const path = normalizeSourceEditPath(record.path);
    const portableIdentity = path.toLocaleLowerCase();
    if (paths.has(portableIdentity)) {
      throw new Error(`Source edit path is duplicated: ${path}`);
    }
    paths.add(portableIdentity);
    if (typeof record.content !== "string" || record.content.includes("\0")) {
      throw new Error("Source edit content must be UTF-8 text without NUL bytes.");
    }
    const expectedSha256 =
      record.expectedSha256 === null
        ? null
        : typeof record.expectedSha256 === "string" &&
            /^[a-f0-9]{64}$/i.test(record.expectedSha256)
          ? record.expectedSha256.toLocaleLowerCase()
          : undefined;
    if (expectedSha256 === undefined) {
      throw new Error(
        "Every source edit requires expectedSha256, or null when the file must be absent."
      );
    }
    totalBytes += Buffer.byteLength(record.content, "utf8");
    if (totalBytes > MAX_SOURCE_EDIT_TOTAL_BYTES) {
      throw new Error(
        `Typed source edits exceed the ${MAX_SOURCE_EDIT_TOTAL_BYTES} byte proposal limit.`
      );
    }
    edits.push({ path, content: record.content, expectedSha256 });
  }
  return edits;
}

function boundedTimeout(value: unknown, fallback = 60_000): number {
  return typeof value === "number" && Number.isFinite(value)
    ? Math.round(Math.max(1_000, Math.min(600_000, value)))
    : fallback;
}

function riskyInvocation(toolId: string, action: string): boolean {
  return (
    (toolId === "windows.files" && action === "write") ||
    toolId === "windows.powershell" ||
    toolId === "code.execute" ||
    toolId === "browser.automation" ||
    toolId === "agent.fork" ||
    (toolId === "source.self-modify" && ["promote", "rollback"].includes(action))
  );
}

function toolCancellationError(): Error {
  return new Error("Tool execution was cancelled.");
}

function assertToolActive(signal: AbortSignal): void {
  if (signal.aborted) throw toolCancellationError();
}

function toolDelay(milliseconds: number, signal: AbortSignal): Promise<void> {
  const duration = Math.max(0, Math.min(30_000, Math.round(milliseconds)));
  assertToolActive(signal);
  return new Promise((resolveDelay, rejectDelay) => {
    const timer = setTimeout(finish, duration);
    const abort = (): void => {
      clearTimeout(timer);
      signal.removeEventListener("abort", abort);
      rejectDelay(toolCancellationError());
    };
    function finish(): void {
      signal.removeEventListener("abort", abort);
      resolveDelay();
    }
    signal.addEventListener("abort", abort, { once: true });
  });
}

function terminateProcessTree(
  child: ChildProcess,
  force: boolean
): void {
  if (child.exitCode !== null) return;
  if (process.platform === "win32" && child.pid) {
    const killer = spawn(
      "taskkill.exe",
      ["/PID", String(child.pid), "/T", "/F"],
      {
        shell: false,
        windowsHide: true,
        stdio: "ignore"
      }
    );
    const fallback = (): void => {
      if (child.exitCode === null) child.kill(force ? "SIGKILL" : "SIGTERM");
    };
    killer.once("error", fallback);
    killer.once("exit", (code) => {
      if (code !== 0) fallback();
    });
    return;
  }
  try {
    if (!child.pid) throw new Error("The child process has no process id.");
    process.kill(-child.pid, force ? "SIGKILL" : "SIGTERM");
  } catch {
    child.kill(force ? "SIGKILL" : "SIGTERM");
  }
}

async function runProcess(
  command: string,
  args: string[],
  cwd: string,
  timeoutMs: number,
  environment?: Record<string, string>,
  signal?: AbortSignal
): Promise<{ exitCode: number; stdout: string; stderr: string; truncated: boolean }> {
  return new Promise((resolveProcess, rejectProcess) => {
    if (signal?.aborted) {
      rejectProcess(toolCancellationError());
      return;
    }
    const child = spawn(command, args, {
      cwd,
      env: { ...process.env, ...environment },
      shell: false,
      windowsHide: true,
      detached: process.platform !== "win32",
      stdio: ["ignore", "pipe", "pipe"]
    });
    let stdout: Buffer<ArrayBufferLike> = Buffer.alloc(0);
    let stderr: Buffer<ArrayBufferLike> = Buffer.alloc(0);
    let truncated = false;
    let termination: "cancelled" | "timed-out" | undefined;
    let settled = false;
    let forceTimer: NodeJS.Timeout | undefined;
    const append = (
      current: Buffer<ArrayBufferLike>,
      chunk: Buffer<ArrayBufferLike>
    ): Buffer<ArrayBufferLike> => {
      if (current.byteLength >= MAX_PROCESS_OUTPUT) {
        truncated = true;
        return current;
      }
      const remaining = MAX_PROCESS_OUTPUT - current.byteLength;
      if (chunk.byteLength > remaining) truncated = true;
      return Buffer.concat([current, chunk.subarray(0, remaining)]);
    };
    child.stdout.on("data", (chunk: Buffer<ArrayBufferLike>) => {
      stdout = append(stdout, chunk);
    });
    child.stderr.on("data", (chunk: Buffer<ArrayBufferLike>) => {
      stderr = append(stderr, chunk);
    });
    const terminate = (reason: "cancelled" | "timed-out"): void => {
      if (termination) return;
      termination = reason;
      terminateProcessTree(child, false);
      forceTimer = setTimeout(() => terminateProcessTree(child, true), 1_000);
      forceTimer.unref();
    };
    const timeout = setTimeout(() => terminate("timed-out"), timeoutMs);
    const abort = (): void => terminate("cancelled");
    signal?.addEventListener("abort", abort, { once: true });
    if (signal?.aborted) abort();
    const cleanup = (): void => {
      clearTimeout(timeout);
      if (forceTimer) clearTimeout(forceTimer);
      signal?.removeEventListener("abort", abort);
    };
    const rejectOnce = (error: Error): void => {
      if (settled) return;
      settled = true;
      cleanup();
      rejectProcess(error);
    };
    child.once("error", (error) => {
      if (termination === "cancelled") rejectOnce(toolCancellationError());
      else if (termination === "timed-out") {
        rejectOnce(new Error(`Tool execution timed out after ${timeoutMs} ms.`));
      } else rejectOnce(error);
    });
    child.once("close", (code) => {
      if (settled) return;
      settled = true;
      cleanup();
      if (termination === "cancelled") {
        rejectProcess(toolCancellationError());
        return;
      }
      if (termination === "timed-out") {
        rejectProcess(new Error(`Tool execution timed out after ${timeoutMs} ms.`));
        return;
      }
      resolveProcess({
        exitCode: code ?? -1,
        stdout: stdout.toString("utf8"),
        stderr: stderr.toString("utf8"),
        truncated
      });
    });
  });
}

interface StaticBrowserPage {
  title: string;
  text: string;
  links: Array<{ label: string; href: string }>;
  document: string;
}

function decodeHtmlEntities(value: string): string {
  const named: Record<string, string> = {
    amp: "&",
    apos: "'",
    gt: ">",
    lt: "<",
    nbsp: " ",
    quot: "\""
  };
  return value.replace(
    /&(?:#(\d{1,7})|#x([a-f0-9]{1,6})|([a-z]{2,12}));/gi,
    (match, decimal: string | undefined, hexadecimal: string | undefined, name: string | undefined) => {
      if (name) return named[name.toLocaleLowerCase()] ?? match;
      const codePoint = Number.parseInt(decimal ?? hexadecimal ?? "", decimal ? 10 : 16);
      return Number.isSafeInteger(codePoint) && codePoint >= 0 && codePoint <= 0x10ffff
        ? String.fromCodePoint(codePoint)
        : "\uFFFD";
    }
  );
}

function passiveHtml(value: string): string {
  return value
    .replace(/<!--[\s\S]*?-->/g, " ")
    .replace(
      /<(?:script|style|noscript|template|iframe|object|embed|svg|math)\b[^>]*>[\s\S]*?<\/(?:script|style|noscript|template|iframe|object|embed|svg|math)\s*>/gi,
      " "
    );
}

function htmlFragmentText(value: string, maximum: number): string {
  return decodeHtmlEntities(
    passiveHtml(value)
      .replace(/<(?:br|hr)\b[^>]*\/?>/gi, "\n")
      .replace(/<\/(?:p|div|section|article|header|footer|main|nav|aside|li|h[1-6]|tr|pre)\s*>/gi, "\n")
      .replace(/<[^>]+>/g, " ")
  )
    .replace(/\r/g, "")
    .replace(/[^\S\n]+/g, " ")
    .replace(/\n[ \t]+/g, "\n")
    .replace(/\n{3,}/g, "\n\n")
    .trim()
    .slice(0, maximum);
}

function escapeHtml(value: string): string {
  return value
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll("\"", "&quot;")
    .replaceAll("'", "&#39;");
}

export function buildInertBrowserDocument(
  source: string,
  baseUrl: string
): StaticBrowserPage {
  const passive = passiveHtml(source);
  const titleMatch = passive.match(/<title\b[^>]*>([\s\S]*?)<\/title\s*>/i);
  const title = htmlFragmentText(titleMatch?.[1] ?? "", 1_000);
  const bodyMatch = passive.match(/<body\b[^>]*>([\s\S]*?)<\/body\s*>/i);
  const text = htmlFragmentText(bodyMatch?.[1] ?? passive, 200_000);
  const links: StaticBrowserPage["links"] = [];
  const pattern = /<a\b[^>]*\bhref\s*=\s*(?:"([^"]*)"|'([^']*)'|([^\s>]+))[^>]*>([\s\S]*?)<\/a\s*>/gi;
  let match: RegExpExecArray | null;
  while ((match = pattern.exec(passive)) && links.length < 500) {
    const rawHref = decodeHtmlEntities(match[1] ?? match[2] ?? match[3] ?? "").trim();
    if (!rawHref) continue;
    try {
      const href = new URL(rawHref, baseUrl);
      if (href.protocol !== "https:" || href.username || href.password) continue;
      href.hash = "";
      links.push({
        label: htmlFragmentText(match[4] ?? "", 500),
        href: href.href.slice(0, 16_000)
      });
    } catch {
      // Ignore malformed or non-URL link targets.
    }
  }
  const document =
    "<!doctype html><html><head><meta charset=\"utf-8\">" +
    `<title>${escapeHtml(title)}</title>` +
    "<style>html{color-scheme:light}body{margin:32px;background:#fff;color:#171717;" +
    "font:16px/1.55 system-ui,sans-serif}pre{white-space:pre-wrap;overflow-wrap:anywhere;" +
    "font:inherit}nav{display:none}</style></head><body>" +
    `<main><pre>${escapeHtml(text)}</pre></main><nav aria-hidden=\"true\">` +
    links
      .map(
        (link) =>
          `<a href="${escapeHtml(link.href)}" rel="noreferrer">${escapeHtml(link.label)}</a>`
      )
      .join("") +
    "</nav></body></html>";
  return { title, text, links, document };
}

export class ToolExecutor {
  private readonly approvals = new Map<string, Approval>();
  private readonly activeExecutions = new Map<
    string,
    { brainId: string; controller: AbortController }
  >();

  constructor(
    private readonly service: BrainService,
    private readonly jobs: RuntimeJobManager,
    private readonly sourceRuntime?: SourceRuntimeLifecycle
  ) {}

  async execute(
    invocation: ToolInvocation,
    onProgress?: (job: RuntimeJob) => void
  ): Promise<ToolExecutionResult> {
    const id = randomUUID();
    const startedAt = new Date().toISOString();
    const base: ToolExecutionResult = {
      id,
      toolId: invocation.toolId,
      action: invocation.action,
      state: "failed",
      startedAt
    };
    try {
      const permission = await this.permission(invocation.brainId, invocation.toolId);
      if (permission === "off") throw new Error("This tool is disabled for the current brain.");
      const outsideAutomaticFileScope =
        permission === "auto" &&
        invocation.toolId === "windows.files" &&
        !(await this.insideAutomaticFileScope(invocation.brainId, invocation.arguments));
      const autonomousEvolutionExperiment =
        invocation.toolId === "source.self-modify" &&
        ["propose", "diff", "test"].includes(invocation.action);
      const needsApproval =
        (permission === "ask" && !autonomousEvolutionExperiment) ||
        (permission === "auto" &&
          (riskyInvocation(invocation.toolId, invocation.action) || outsideAutomaticFileScope));
      if (needsApproval && !this.consumeApproval(invocation)) {
        const approvalToken = randomUUID();
        this.approvals.set(approvalToken, {
          brainId: invocation.brainId,
          toolId: invocation.toolId,
          action: invocation.action,
          argumentSha256: sha256(JSON.stringify(invocation.arguments)),
          expiresAt: Date.now() + 5 * 60_000
        });
        return { ...base, state: "approval-required", approvalToken };
      }
      const controller = new AbortController();
      this.activeExecutions.set(id, { brainId: invocation.brainId, controller });
      let output: unknown;
      try {
        output = await this.dispatch(
          invocation,
          controller.signal,
          onProgress,
          permission
        );
      } catch (error) {
        if (controller.signal.aborted) throw toolCancellationError();
        throw error;
      } finally {
        this.activeExecutions.delete(id);
      }
      const result: ToolExecutionResult = {
        ...base,
        state: "complete",
        finishedAt: new Date().toISOString(),
        output
      };
      await this.audit(invocation, result);
      return result;
    } catch (error) {
      const result: ToolExecutionResult = {
        ...base,
        state: "failed",
        finishedAt: new Date().toISOString(),
        error: error instanceof Error ? error.message : String(error)
      };
      await this.audit(invocation, result).catch(() => undefined);
      return result;
    }
  }

  cancel(brainId: string): number {
    let cancelled = 0;
    for (const execution of this.activeExecutions.values()) {
      if (execution.brainId !== brainId || execution.controller.signal.aborted) continue;
      execution.controller.abort();
      cancelled += 1;
    }
    return cancelled;
  }

  hasPendingOrActive(brainId: string): boolean {
    const now = Date.now();
    for (const [token, approval] of this.approvals) {
      if (approval.expiresAt < now) this.approvals.delete(token);
    }
    return (
      [...this.approvals.values()].some(
        (approval) => approval.brainId === brainId
      ) ||
      [...this.activeExecutions.values()].some(
        (execution) =>
          execution.brainId === brainId && !execution.controller.signal.aborted
      )
    );
  }

  private async permission(brainId: string, toolId: string): Promise<ToolPermissionLevel> {
    const permissions = await this.service.listToolPermissions(brainId);
    return permissions.find((permission) => permission.toolId === toolId)?.level ?? "off";
  }

  private consumeApproval(invocation: ToolInvocation): boolean {
    if (!invocation.approvalToken) return false;
    const approval = this.approvals.get(invocation.approvalToken);
    this.approvals.delete(invocation.approvalToken);
    return Boolean(
      approval &&
        approval.expiresAt >= Date.now() &&
        approval.brainId === invocation.brainId &&
        approval.toolId === invocation.toolId &&
        approval.action === invocation.action &&
        approval.argumentSha256 === sha256(JSON.stringify(invocation.arguments))
    );
  }

  private async insideAutomaticFileScope(
    brainId: string,
    args: Record<string, unknown>
  ): Promise<boolean> {
    try {
      const requested = absolutePath(argumentString(args, "path", 32_000));
      const [root, target] = await Promise.all([
        realpath(this.service.repository.brainDirectory(brainId)),
        realpath(requested)
      ]);
      const fromRoot = relative(root, target);
      return fromRoot === "" || (!fromRoot.startsWith("..") && !isAbsolute(fromRoot));
    } catch {
      return false;
    }
  }

  private async dispatch(
    invocation: ToolInvocation,
    signal: AbortSignal,
    onProgress?: (job: RuntimeJob) => void,
    permission?: ToolPermissionLevel
  ): Promise<unknown> {
    switch (invocation.toolId) {
      case "windows.files":
        return this.files(invocation.action, invocation.arguments);
      case "windows.powershell":
        return this.powershell(invocation.action, invocation.arguments, signal);
      case "code.execute":
        return this.code(invocation.action, invocation.arguments, signal);
      case "web.fetch":
        return this.fetch(invocation.action, invocation.arguments, signal);
      case "web.search":
        return this.search(invocation.action, invocation.arguments, signal);
      case "modality.imagine":
        return this.imagine(
          invocation.brainId,
          invocation.action,
          invocation.arguments,
          signal,
          onProgress
        );
      case "agent.fork":
        return this.agent(
          invocation.brainId,
          invocation.action,
          invocation.arguments,
          signal
        );
      case "browser.automation":
        return this.browser(invocation.brainId, invocation.action, invocation.arguments, signal);
      case "source.self-modify":
        return this.sourceEvolution(
          invocation.brainId,
          invocation.action,
          invocation.arguments,
          signal,
          permission ?? "off"
        );
      default:
        throw new Error("Unknown tool protocol.");
    }
  }

  private async files(action: string, args: Record<string, unknown>): Promise<unknown> {
    const path = absolutePath(argumentString(args, "path", 32_000));
    if (action === "list") {
      const entries = await readdir(path, { withFileTypes: true });
      return {
        entries: await Promise.all(
          entries.slice(0, 5_000).map(async (entry) => {
            const childPath = resolve(path, entry.name);
            const info = await lstat(childPath);
            return {
              name: entry.name,
              path: childPath,
              kind: info.isSymbolicLink()
                ? "link"
                : info.isDirectory()
                  ? "directory"
                  : info.isFile()
                    ? "file"
                    : "other",
              bytes: info.size,
              modifiedAt: info.mtime.toISOString()
            };
          })
        )
      };
    }
    if (action === "read") {
      const info = await stat(path);
      const requested =
        typeof args.maxBytes === "number" ? Math.round(args.maxBytes) : MAX_TEXT_BYTES;
      const maximum = Math.max(1, Math.min(MAX_TEXT_BYTES, requested));
      if (!info.isFile() || info.size > maximum) throw new Error("File exceeds the allowed read size.");
      const contents = await readFile(path);
      return { content: contents.toString("utf8"), sha256: sha256(contents), bytes: contents.byteLength };
    }
    if (action === "write") {
      const content = argumentString(args, "content", MAX_TEXT_BYTES);
      const expected = typeof args.expectedSha256 === "string" ? args.expectedSha256 : undefined;
      if (expected) {
        const current = await readFile(path);
        if (sha256(current) !== expected.toLocaleLowerCase()) {
          throw new Error("File changed since the tool read it; expected checksum does not match.");
        }
      }
      await atomicWrite(path, content);
      return { sha256: sha256(content), bytes: Buffer.byteLength(content) };
    }
    throw new Error("Unknown windows.files action.");
  }

  private async powershell(
    action: string,
    args: Record<string, unknown>,
    signal: AbortSignal
  ): Promise<unknown> {
    if (action !== "run") throw new Error("Unknown PowerShell action.");
    const command = argumentString(args, "command");
    const cwd = absolutePath(argumentString(args, "cwd", 32_000));
    const executable = process.platform === "win32" ? "powershell.exe" : "pwsh";
    return runProcess(
      executable,
      ["-NoLogo", "-NoProfile", "-NonInteractive", "-Command", command],
      cwd,
      boundedTimeout(args.timeoutMs),
      undefined,
      signal
    );
  }

  private async code(
    action: string,
    args: Record<string, unknown>,
    signal: AbortSignal
  ): Promise<unknown> {
    if (action !== "run") throw new Error("Unknown code runner action.");
    const language = argumentString(args, "language", 32).toLocaleLowerCase();
    const entryPath = absolutePath(argumentString(args, "entryPath", 32_000));
    const userArgs = Array.isArray(args.arguments)
      ? args.arguments
          .filter((value): value is string => typeof value === "string")
          .slice(0, 64)
          .map((value) => value.slice(0, 8_000))
      : [];
    const mapping: Record<
      string,
      { executable: string; args: string[]; environment?: Record<string, string> }
    > = {
      python: {
        executable:
          process.env.OMNI_PYTHON || process.env.OMNI_AGI_PYTHON || (process.platform === "win32" ? "python.exe" : "python3"),
        args: [entryPath]
      },
      javascript: {
        executable: process.execPath,
        args: [entryPath],
        environment: { ELECTRON_RUN_AS_NODE: "1" }
      },
      node: {
        executable: process.execPath,
        args: [entryPath],
        environment: { ELECTRON_RUN_AS_NODE: "1" }
      },
      powershell: {
        executable: process.platform === "win32" ? "powershell.exe" : "pwsh",
        args: ["-NoLogo", "-NoProfile", "-NonInteractive", "-File", entryPath]
      }
    };
    const selected = mapping[language];
    if (!selected) throw new Error("Language runner is not configured.");
    return runProcess(
      selected.executable,
      [...selected.args, ...userArgs],
      dirname(entryPath),
      boundedTimeout(args.timeoutMs),
      selected.environment,
      signal
    );
  }

  private async fetch(
    action: string,
    args: Record<string, unknown>,
    signal: AbortSignal
  ): Promise<unknown> {
    if (action !== "fetch") throw new Error("Unknown web.fetch action.");
    const url = new URL(argumentString(args, "url", 16_000));
    const response = await safeFetch(url, {
      signal: AbortSignal.any([
        signal,
        AbortSignal.timeout(boundedTimeout(args.timeoutMs, 120_000))
      ]),
      headers: { Accept: "text/*, application/json;q=0.9" }
    });
    const maximum =
      typeof args.maxBytes === "number"
        ? Math.max(1, Math.min(MAX_WEB_BYTES, Math.round(args.maxBytes)))
        : MAX_WEB_BYTES;
    const contents = await readResponseBounded(response, maximum);
    return {
      status: response.status,
      finalUrl: response.url,
      contentType: response.headers.get("content-type") ?? "application/octet-stream",
      content: contents.toString("utf8"),
      sha256: sha256(contents)
    };
  }

  private async search(
    action: string,
    args: Record<string, unknown>,
    signal: AbortSignal
  ): Promise<unknown> {
    if (action !== "search") throw new Error("Unknown web.search action.");
    const endpoint = process.env.OMNI_SEARXNG_URL;
    if (!endpoint) throw new Error("OMNI_SEARXNG_URL is not configured.");
    const url = new URL(endpoint);
    url.searchParams.set("q", argumentString(args, "query", 4_000));
    url.searchParams.set("format", "json");
    const response = await safeFetch(url, {
      signal: AbortSignal.any([signal, AbortSignal.timeout(120_000)]),
      headers: { Accept: "application/json" }
    });
    if (!response.ok) throw new Error(`Search provider returned HTTP ${response.status}.`);
    const data = JSON.parse(
      (await readResponseBounded(response, 4 * 1024 * 1024)).toString("utf8")
    ) as { results?: unknown[] };
    const limit =
      typeof args.limit === "number" ? Math.max(1, Math.min(50, Math.round(args.limit))) : 10;
    return { results: Array.isArray(data.results) ? data.results.slice(0, limit) : [] };
  }

  private async browser(
    brainId: string,
    action: string,
    args: Record<string, unknown>,
    signal: AbortSignal
  ): Promise<unknown> {
    if (!["task", "open"].includes(action)) {
      throw new Error("Browser automation supports task or open.");
    }
    const requested = new URL(argumentString(args, "url", 16_000));
    await assertSafeRemoteUrl(requested);
    const steps = Array.isArray(args.steps)
      ? args.steps
          .filter(
            (value): value is Record<string, unknown> =>
              typeof value === "object" && value !== null && !Array.isArray(value)
          )
          .slice(0, 200)
      : [];
    const { BrowserWindow } = await import("electron");
    const browser = new BrowserWindow({
      show: args.visible === true,
      width: 1280,
      height: 900,
      backgroundColor: "#ffffff",
      webPreferences: {
        // Persistent per-brain cookies/storage allow a user-approved browser
        // task to continue inside an existing signed-in session.
        partition: `persist:omni-browser-${sha256(brainId).slice(0, 20)}`,
        sandbox: true,
        contextIsolation: true,
        nodeIntegration: false,
        javascript: true,
        webSecurity: true
      }
    });
    const requestSession = browser.webContents.session;
    requestSession.webRequest.onBeforeRequest((details, callback) => {
      let url: URL;
      try {
        url = new URL(details.url);
      } catch {
        callback({ cancel: true });
        return;
      }
      if (["data:", "blob:", "about:"].includes(url.protocol)) {
        callback({ cancel: false });
        return;
      }
      const validation =
        url.protocol === "wss:"
          ? new URL(`https://${url.host}${url.pathname}${url.search}`)
          : url;
      void assertSafeRemoteUrl(validation).then(
        () => callback({ cancel: false }),
        () => callback({ cancel: true })
      );
    });
    requestSession.setPermissionRequestHandler((_contents, _permission, callback) => {
      callback(false);
    });
    const preventDownload = (
      _event: Electron.Event,
      item: Electron.DownloadItem
    ): void => item.cancel();
    requestSession.on("will-download", preventDownload);
    browser.webContents.setWindowOpenHandler(() => ({ action: "deny" }));
    browser.webContents.on("will-navigate", (event, destination) => {
      try {
        const target = new URL(destination);
        if (!["https:", "http:"].includes(target.protocol)) event.preventDefault();
      } catch {
        event.preventDefault();
      }
    });
    const abort = (): void => {
      if (!browser.isDestroyed()) browser.destroy();
    };
    signal.addEventListener("abort", abort, { once: true });
    const waitForLoading = async (maximum = 30_000): Promise<void> => {
      const deadline = Date.now() + Math.max(1_000, Math.min(30_000, maximum));
      while (!browser.isDestroyed() && browser.webContents.isLoading()) {
        assertToolActive(signal);
        if (Date.now() >= deadline) {
          throw new Error("Browser navigation did not settle before its timeout.");
        }
        await toolDelay(50, signal);
      }
    };
    const selector = (step: Record<string, unknown>): string =>
      argumentString(step, "selector", 2_000);
    const pageSnapshot = async (): Promise<{
      title: string;
      text: string;
      links: Array<{ label: string; href: string }>;
    }> =>
      browser.webContents.executeJavaScript(
        `(() => ({
          title: document.title.slice(0, 1000),
          text: (document.body?.innerText || "").slice(0, 200000),
          links: Array.from(document.querySelectorAll("a[href]")).slice(0, 500).map((item) => ({
            label: (item.textContent || "").trim().slice(0, 500),
            href: item.href.slice(0, 16000)
          }))
        }))()`,
        true
      ) as Promise<{
        title: string;
        text: string;
        links: Array<{ label: string; href: string }>;
      }>;
    try {
      assertToolActive(signal);
      await browser.loadURL(requested.href);
      assertToolActive(signal);
      const stepResults: unknown[] = [];
      for (const [index, step] of steps.entries()) {
        assertToolActive(signal);
        const kind = argumentString(step, "kind", 32).toLocaleLowerCase();
        if (kind === "navigate") {
          const destination = new URL(argumentString(step, "url", 16_000));
          await assertSafeRemoteUrl(destination);
          await browser.loadURL(destination.href);
          stepResults.push({ index, kind, finalUrl: browser.webContents.getURL() });
          continue;
        }
        if (kind === "click") {
          const query = selector(step);
          const clicked = await browser.webContents.executeJavaScript(
            `((query) => {
              const node = document.querySelector(query);
              if (!(node instanceof HTMLElement)) return false;
              node.scrollIntoView({block: "center", inline: "center"});
              node.click();
              return true;
            })(${JSON.stringify(query)})`,
            true
          );
          if (clicked !== true) throw new Error(`Browser step ${index} could not find its selector.`);
          await waitForLoading(
            typeof step.timeoutMs === "number" ? step.timeoutMs : 30_000
          );
          stepResults.push({ index, kind, selector: query });
          continue;
        }
        if (kind === "type") {
          const query = selector(step);
          const value = argumentString(step, "value", 100_000);
          const typed = await browser.webContents.executeJavaScript(
            `((input) => {
              const node = document.querySelector(input.selector);
              if (!(node instanceof HTMLInputElement) &&
                  !(node instanceof HTMLTextAreaElement) &&
                  !(node instanceof HTMLElement && node.isContentEditable)) return false;
              node.focus();
              if (node instanceof HTMLInputElement || node instanceof HTMLTextAreaElement) {
                if (input.clear) node.value = "";
                node.value += input.value;
              } else {
                if (input.clear) node.textContent = "";
                node.textContent = (node.textContent || "") + input.value;
              }
              node.dispatchEvent(new InputEvent("input", {bubbles: true, inputType: "insertText", data: input.value}));
              node.dispatchEvent(new Event("change", {bubbles: true}));
              return true;
            })(${JSON.stringify({
              selector: query,
              value,
              clear: step.clear !== false
            })})`,
            true
          );
          if (typed !== true) throw new Error(`Browser step ${index} could not type into its selector.`);
          stepResults.push({
            index,
            kind,
            selector: query,
            characters: value.length,
            sensitive: step.sensitive === true
          });
          continue;
        }
        if (kind === "press") {
          const key = argumentString(step, "key", 64);
          browser.webContents.sendInputEvent({ type: "keyDown", keyCode: key });
          browser.webContents.sendInputEvent({ type: "keyUp", keyCode: key });
          await waitForLoading(
            typeof step.timeoutMs === "number" ? step.timeoutMs : 30_000
          );
          stepResults.push({ index, kind, key });
          continue;
        }
        if (kind === "wait") {
          const waitSelector =
            typeof step.selector === "string" && step.selector.trim()
              ? selector(step)
              : "";
          const timeout = boundedTimeout(step.timeoutMs, 30_000);
          if (waitSelector) {
            const deadline = Date.now() + Math.min(30_000, timeout);
            let found = false;
            while (!found && Date.now() < deadline) {
              found =
                (await browser.webContents.executeJavaScript(
                  `document.querySelector(${JSON.stringify(waitSelector)}) !== null`,
                  true
                )) === true;
              if (!found) await toolDelay(100, signal);
            }
            if (!found) throw new Error(`Browser step ${index} timed out waiting for its selector.`);
          } else {
            const milliseconds =
              typeof step.milliseconds === "number"
                ? step.milliseconds
                : 500;
            await toolDelay(milliseconds, signal);
          }
          stepResults.push({ index, kind, selector: waitSelector || undefined });
          continue;
        }
        if (kind === "extract") {
          const query =
            typeof step.selector === "string" && step.selector.trim()
              ? selector(step)
              : "body";
          const extracted = await browser.webContents.executeJavaScript(
            `((query) => Array.from(document.querySelectorAll(query)).slice(0, 1000).map((node) => ({
              text: (node.textContent || "").trim().slice(0, 20000),
              href: node instanceof HTMLAnchorElement ? node.href.slice(0, 16000) : undefined,
              value: node instanceof HTMLInputElement || node instanceof HTMLTextAreaElement
                ? node.value.slice(0, 20000)
                : undefined
            })))(${JSON.stringify(query)})`,
            true
          );
          stepResults.push({ index, kind, selector: query, values: extracted });
          continue;
        }
        if (kind === "screenshot") {
          const image = await browser.webContents.capturePage();
          const artifactDirectory = join(
            this.service.repository.brainDirectory(brainId),
            "artifacts",
            "browser"
          );
          await mkdir(artifactDirectory, { recursive: true });
          const path = join(artifactDirectory, `${randomUUID()}.png`);
          await writeFile(path, image.toPNG(), { flag: "wx", mode: 0o600 });
          stepResults.push({ index, kind, artifactPath: path });
          continue;
        }
        throw new Error(`Unknown browser step kind at index ${index}.`);
      }
      const page = await pageSnapshot();
      assertToolActive(signal);
      const artifactDirectory = join(
        this.service.repository.brainDirectory(brainId),
        "artifacts",
        "browser"
      );
      await mkdir(artifactDirectory, { recursive: true });
      const artifactPath = join(artifactDirectory, `${randomUUID()}.png`);
      const screenshot = await browser.webContents.capturePage();
      await writeFile(artifactPath, screenshot.toPNG(), { flag: "wx", mode: 0o600 });
      return {
        finalUrl: browser.webContents.getURL(),
        title: page.title,
        text: page.text,
        links: page.links,
        steps: stepResults,
        artifactPath,
        mode: "interactive-persistent-session",
        sessionPersistent: true,
        note:
          "Public-network validation applies to every request. Actions, navigation, and the persistent per-brain signed-in session remain permission-gated and audited."
      };
    } finally {
      signal.removeEventListener("abort", abort);
      requestSession.webRequest.onBeforeRequest(null);
      requestSession.removeListener("will-download", preventDownload);
      if (!browser.isDestroyed()) browser.destroy();
    }
  }

  private async imagine(
    brainId: string,
    action: string,
    args: Record<string, unknown>,
    signal: AbortSignal,
    onProgress?: (job: RuntimeJob) => void
  ): Promise<unknown> {
    if (action !== "generate") throw new Error("Unknown imagination action.");
    const modality = argumentString(args, "modality", 16);
    if (!["image", "audio", "video"].includes(modality)) {
      throw new Error("Imagination modality must be image, audio, or video.");
    }
    const neuralActionId =
      typeof args.neuralActionId === "string" &&
      /^[a-f0-9]{32}$/.test(args.neuralActionId)
        ? args.neuralActionId
        : undefined;
    const seed =
      typeof args.seed === "number" &&
      Number.isSafeInteger(args.seed)
        ? args.seed
        : undefined;
    const job = this.jobs.generate({
      brainId,
      modality: modality as "image" | "audio" | "video",
      prompt: typeof args.prompt === "string" ? args.prompt.slice(0, 1_000_000) : undefined,
      conceptIds: Array.isArray(args.conceptIds)
        ? args.conceptIds.filter((value): value is string => typeof value === "string")
        : undefined,
      settings:
        typeof args.settings === "object" && args.settings !== null
          ? (args.settings as Record<string, string | number | boolean>)
          : undefined,
      ...(neuralActionId ? { neuralActionId } : {}),
      ...(seed !== undefined ? { seed } : {})
    });
    onProgress?.(job);
    const progressListener = ({ job: update }: { job: RuntimeJob }): void => {
      if (update.id === job.id) onProgress?.(update);
    };
    this.jobs.on("event", progressListener);
    let finished: RuntimeJob;
    try {
      finished = await this.jobs.wait(
        job.id,
        signal,
        boundedTimeout(args.timeoutMs, 600_000)
      );
    } finally {
      this.jobs.off("event", progressListener);
    }
    if (finished.state === "failed") {
      throw new Error(finished.error || `${modality} generation failed.`);
    }
    if (finished.state === "cancelled") throw toolCancellationError();
    if (finished.state !== "complete") {
      throw new Error(`${modality} generation ended in an unexpected state.`);
    }
    const output =
      typeof finished.output === "object" && finished.output !== null
        ? (finished.output as Record<string, unknown>)
        : { value: finished.output };
    return {
      ...output,
      jobId: finished.id,
      state: finished.state,
      artifactPath:
        typeof output.artifactPath === "string"
          ? output.artifactPath
          : typeof output.path === "string"
            ? output.path
            : undefined
    };
  }

  private async agent(
    brainId: string,
    action: string,
    args: Record<string, unknown>,
    signal: AbortSignal
  ): Promise<unknown> {
    assertToolActive(signal);
    if (action !== "start") throw new Error("Unknown subagent action.");
    const objective = argumentString(args, "objective", 20_000);
    const requestedWorkers =
      typeof args.workers === "number" && Number.isFinite(args.workers)
        ? Math.round(args.workers)
        : 1;
    const workers = Math.max(1, Math.min(4, requestedWorkers));
    const jobId = randomUUID();
    const forks = [];
    for (let index = 0; index < workers; index += 1) {
      assertToolActive(signal);
      const suffix = workers > 1 ? ` ${index + 1}` : "";
      forks.push(
        await this.service.repository.fork(
          brainId,
          `${basename(objective).slice(0, 52) || "Subagent"}${suffix} branch`
        )
      );
    }
    const results = [];
    for (const fork of forks) {
      assertToolActive(signal);
      const result = await this.service.chat(
        fork.id,
        objective,
        signal,
        undefined,
        undefined,
        SUBAGENT_RESPONSE_TOKENS
      );
      assertToolActive(signal);
      results.push({
        forkId: fork.id,
        response: result.brainMessage.content,
        traceId: result.trace.id,
        concepts: Object.keys(result.brain.concepts).length,
        synapses: Object.keys(result.brain.synapses).length
      });
    }
    return {
      jobId,
      forkIds: forks.map((fork) => fork.id),
      objective,
      state: "complete",
      results,
      mergePolicy: "ideas-evidence-replay-only"
    };
  }

  private async verifyRuntimeStage(
    stage: SourceRuntimeStageResult,
    expected: {
      repository: string;
      worktree: string;
      parentCommit: string;
      diffSha256: string;
      evaluatorSha256: string;
      brainSnapshotId: string;
      currentExecutablePath: string;
      currentExecutableSha256: string;
    }
  ): Promise<SourceRuntimeStageResult> {
    if (
      !stage ||
      typeof stage !== "object" ||
      !/^[a-z0-9][a-z0-9-]{5,127}$/i.test(stage.slotId ?? "")
    ) {
      throw new Error("The staged runtime returned an invalid slot identity.");
    }
    if (!["win32", "darwin", "linux"].includes(process.platform)) {
      throw new Error(`Native runtime activation is unsupported on ${process.platform}.`);
    }
    if (!["x64", "arm64"].includes(process.arch)) {
      throw new Error(`Native runtime activation is unsupported on ${process.arch}.`);
    }
    const [root, manifestPath, executablePath, repository, worktree, currentExecutable] =
      await Promise.all([
        realpath(stage.rootPath),
        realpath(stage.manifestPath),
        realpath(stage.executablePath),
        realpath(expected.repository),
        realpath(expected.worktree),
        realpath(expected.currentExecutablePath)
      ]);
    const within = (parent: string, child: string): boolean => {
      const fromParent = relative(parent, child);
      return (
        fromParent === "" ||
        (!fromParent.startsWith("..") && !isAbsolute(fromParent))
      );
    };
    if (
      !within(root, manifestPath) ||
      !within(root, executablePath) ||
      within(repository, root) ||
      within(worktree, root) ||
      resolve(executablePath) === resolve(currentExecutable)
    ) {
      throw new Error(
        "The staged runtime must be side-by-side, outside source worktrees, and distinct from the running executable."
      );
    }
    if (
      resolve(manifestPath) !== resolve(join(root, "runtime-manifest.json"))
    ) {
      throw new Error("The staged runtime manifest must be rooted in its exact slot.");
    }
    const manifestFile = await sha256File(manifestPath);
    if (
      !/^[a-f0-9]{64}$/i.test(stage.manifestSha256) ||
      manifestFile.sha256 !== stage.manifestSha256.toLocaleLowerCase()
    ) {
      throw new Error("The staged runtime manifest hash does not match its artifact.");
    }
    const manifest = await readRuntimeManifest(manifestPath);
    const lineage = manifest.lineage;
    if (
      manifest.schemaVersion !== 1 ||
      manifest.slotId !== stage.slotId ||
      manifest.state !== "staged" ||
      manifest.platform !== process.platform ||
      manifest.architecture !== process.arch ||
      lineage?.parentCommit !== expected.parentCommit ||
      lineage?.diffSha256 !== expected.diffSha256 ||
      lineage?.evaluatorSha256 !== expected.evaluatorSha256 ||
      lineage?.brainSnapshotId !== expected.brainSnapshotId ||
      manifest.currentExecutableSha256 !== expected.currentExecutableSha256 ||
      !["reused-current-worker", "rebuilt-protected-worker"].includes(
        manifest.engineStrategy
      )
    ) {
      throw new Error("The staged runtime manifest lineage is invalid or mismatched.");
    }
    if (JSON.stringify(stage.manifest) !== JSON.stringify(manifest)) {
      throw new Error("The runtime host result does not match its persisted manifest.");
    }
    const inspected = await inspectRuntimeArtifacts(root);
    if (
      inspected.artifactSha256 !== manifest.artifactSha256 ||
      JSON.stringify(inspected.artifacts) !== JSON.stringify(manifest.artifacts)
    ) {
      throw new Error("The staged runtime artifact tree failed hash verification.");
    }
    const executableRelativePath = normalizeRuntimeRelativePath(
      manifest.executableRelativePath
    );
    const recordedExecutable = await realpath(
      resolve(root, ...executableRelativePath.split("/"))
    );
    if (resolve(recordedExecutable) !== resolve(executablePath)) {
      throw new Error("The staged executable path does not match its manifest.");
    }
    const executableArtifact = inspected.artifacts.find(
      (artifact) => artifact.path === executableRelativePath
    );
    const executableHash = await sha256File(executablePath);
    if (
      executableArtifact?.kind !== "file" ||
      executableArtifact.sha256 !== executableHash.sha256 ||
      manifest.executableSha256 !== executableHash.sha256
    ) {
      throw new Error("The staged executable failed independent hash verification.");
    }
    const currentAfterStage = await sha256File(currentExecutable);
    if (currentAfterStage.sha256 !== expected.currentExecutableSha256) {
      throw new Error("The running executable changed while staging its side-by-side successor.");
    }
    return {
      ...stage,
      rootPath: root,
      manifestPath,
      executablePath,
      manifestSha256: manifestFile.sha256,
      manifest
    };
  }

  private async verifyRuntimeActivation(
    activation: SourceRuntimeActivationResult,
    stage: SourceRuntimeStageResult,
    promotionCommit: string,
    currentExecutablePath: string,
    currentExecutableSha256: string
  ): Promise<SourceRuntimeActivationResult> {
    if (
      !activation ||
      !["scheduled", "deferred"].includes(activation.state) ||
      activation.slotId !== stage.slotId ||
      resolve(activation.executablePath) !== resolve(stage.executablePath) ||
      resolve(activation.manifestPath) !== resolve(stage.manifestPath) ||
      activation.promotionCommit !== promotionCommit ||
      !Number.isSafeInteger(activation.delayMs) ||
      activation.delayMs < 3_000
    ) {
      throw new Error("The runtime host returned an invalid activation schedule.");
    }
    const [manifestFile, manifest, currentExecutable, inspected] = await Promise.all([
      sha256File(stage.manifestPath),
      readRuntimeManifest(stage.manifestPath),
      sha256File(currentExecutablePath),
      inspectRuntimeArtifacts(stage.rootPath)
    ]);
    if (
      manifestFile.sha256 !== activation.manifestSha256 ||
      manifest.slotId !== stage.slotId ||
      manifest.state !== activation.state ||
      manifest.promotionCommit !== promotionCommit ||
      manifest.activation?.state !== activation.state ||
      manifest.activation.delayMs !== activation.delayMs ||
      inspected.artifactSha256 !== stage.manifest.artifactSha256 ||
      JSON.stringify(inspected.artifacts) !==
        JSON.stringify(stage.manifest.artifacts) ||
      currentExecutable.sha256 !== currentExecutableSha256
    ) {
      throw new Error(
        "The scheduled runtime manifest is invalid or the running executable was replaced."
      );
    }
    return {
      ...activation,
      executablePath: stage.executablePath,
      manifestPath: stage.manifestPath,
      manifestSha256: manifestFile.sha256
    };
  }

  private async sourceEvolution(
    brainId: string,
    action: string,
    args: Record<string, unknown>,
    signal: AbortSignal,
    permission: ToolPermissionLevel
  ): Promise<unknown> {
    assertToolActive(signal);
    const configured = process.env.OMNI_SOURCE_REPOSITORY;
    if (!configured) {
      throw new Error(
        "Source evolution requires OMNI_SOURCE_REPOSITORY to point at an authorized Git clone."
      );
    }
    const repository = await realpath(absolutePath(configured));
    const evolutionRoot = process.env.OMNI_EVOLUTION_ROOT
      ? absolutePath(process.env.OMNI_EVOLUTION_ROOT)
      : join(dirname(repository), ".omni-evolution");
    await mkdir(evolutionRoot, { recursive: true });
    const runEvolutionProcess = (
      command: string,
      commandArgs: string[],
      cwd: string,
      timeoutMs: number
    ): ReturnType<typeof runProcess> =>
      runProcess(command, commandArgs, cwd, timeoutMs, undefined, signal);
    const probe = await runEvolutionProcess(
      "git",
      ["-C", repository, "rev-parse", "--show-toplevel"],
      repository,
      30_000
    );
    if (probe.exitCode !== 0) throw new Error(`Configured source is not a Git clone: ${probe.stderr}`);

    const proposalPath = (worktree: string): string =>
      join(evolutionRoot, `${basename(worktree)}.proposal.json`);
    const readPackageScripts = async (
      root: string
    ): Promise<Record<string, string>> => {
      const value = JSON.parse(await readFile(join(root, "package.json"), "utf8")) as {
        scripts?: unknown;
      };
      if (typeof value.scripts !== "object" || value.scripts === null) return {};
      return Object.fromEntries(
        EVOLUTION_TEST_NAMES.map((name) => {
          const script = (value.scripts as Record<string, unknown>)[name];
          return [name, typeof script === "string" ? script : ""];
        })
      );
    };
    const readProposal = async (worktree: string): Promise<EvolutionProposalRecord> => {
      const value = JSON.parse(await readFile(proposalPath(worktree), "utf8")) as
        Partial<EvolutionProposalRecord>;
      const recordedWorktree =
        typeof value.worktree === "string"
          ? await realpath(value.worktree).catch(() => "")
          : "";
      const invalid = [
        value.schemaVersion !== 1 ? "schema" : undefined,
        resolve(recordedWorktree) !== resolve(worktree) ? "worktree" : undefined,
        typeof value.branch !== "string" ? "branch" : undefined,
        !/^[a-f0-9]{40,64}$/i.test(value.parentCommit ?? "") ? "parent" : undefined,
        value.evaluatorVersion !== EVOLUTION_EVALUATOR_VERSION ? "version" : undefined,
        !/^[a-f0-9]{64}$/i.test(value.evaluatorSha256 ?? "") ? "evaluator" : undefined,
        !Array.isArray(value.baselineTestPaths) ? "baseline-tests" : undefined,
        typeof value.packageScripts !== "object" || value.packageScripts === null
          ? "scripts"
          : undefined,
        !Array.isArray(value.sourceEditLineage) ? "source-edit-lineage" : undefined,
        !Array.isArray(value.authoredChangedPaths) ? "authored-paths" : undefined,
        !/^[a-f0-9]{64}$/i.test(value.authoredDiffSha256 ?? "")
          ? "authored-diff"
          : undefined,
        !Number.isSafeInteger(value.authoredBytes) || (value.authoredBytes ?? -1) < 0
          ? "authored-bytes"
          : undefined
      ].filter((entry): entry is string => Boolean(entry));
      if (invalid.length) {
        throw new Error(`The external evolution proposal record is invalid (${invalid.join(", ")}).`);
      }
      const proposal = value as EvolutionProposalRecord;
      const invalidLineage = proposal.sourceEditLineage.some(
        (entry) =>
          typeof entry !== "object" ||
          entry === null ||
          normalizeSourceEditPath(entry.path) !== entry.path ||
          !(
            entry.expectedSha256 === null ||
            /^[a-f0-9]{64}$/i.test(entry.expectedSha256)
          ) ||
          !/^[a-f0-9]{64}$/i.test(entry.resultSha256) ||
          !Number.isSafeInteger(entry.bytes) ||
          entry.bytes < 0
      );
      const lineagePaths = proposal.sourceEditLineage
        .map((entry) => entry.path)
        .sort();
      const authoredPaths = [...proposal.authoredChangedPaths].sort();
      if (
        invalidLineage ||
        proposal.authoredChangedPaths.some(
          (path) =>
            typeof path !== "string" ||
            normalizeSourceEditPath(path) !== path
        ) ||
        JSON.stringify(lineagePaths) !== JSON.stringify(authoredPaths) ||
        proposal.sourceEditLineage.reduce((total, entry) => total + entry.bytes, 0) !==
          proposal.authoredBytes
      ) {
        throw new Error("The external evolution proposal source-edit lineage is invalid.");
      }
      return proposal;
    };
    const resolveCandidate = async (): Promise<string> => {
      const requested = absolutePath(argumentString(args, "worktree", 32_000));
      const [root, candidate] = await Promise.all([realpath(evolutionRoot), realpath(requested)]);
      assertToolActive(signal);
      const fromRoot = relative(root, candidate);
      if (fromRoot.startsWith("..") || isAbsolute(fromRoot)) {
        throw new Error("Evolution worktree must remain inside OMNI_EVOLUTION_ROOT.");
      }
      const top = await runEvolutionProcess(
        "git",
        ["-C", candidate, "rev-parse", "--show-toplevel"],
        candidate,
        30_000
      );
      if (top.exitCode !== 0 || resolve(top.stdout.trim()) !== resolve(candidate)) {
        throw new Error("The selected evolution candidate is not an isolated Git worktree.");
      }
      await readProposal(candidate);
      return candidate;
    };
    const withCandidateDependencies = async <T>(
      worktree: string,
      operation: () => Promise<T>
    ): Promise<T> => {
      const [authorizedRoot, candidateRoot] = await Promise.all([
        realpath(repository),
        realpath(worktree)
      ]);
      const candidateFromRepository = relative(authorizedRoot, candidateRoot);
      if (
        candidateFromRepository === "" ||
        (!candidateFromRepository.startsWith("..") &&
          !isAbsolute(candidateFromRepository))
      ) {
        throw new Error("Evolution dependencies require a sibling isolated Git worktree.");
      }
      const sourceModulesPath = join(authorizedRoot, "node_modules");
      const sourceModules = await lstat(sourceModulesPath).catch((error) => {
        if ((error as NodeJS.ErrnoException).code === "ENOENT") {
          throw new Error(
            "The authorized source repository has no existing node_modules to reuse; dependencies are never installed during evolution."
          );
        }
        throw error;
      });
      if (!sourceModules.isDirectory() || sourceModules.isSymbolicLink()) {
        throw new Error(
          "The authorized source repository node_modules must be a real directory."
        );
      }
      const sourceModulesRoot = await realpath(sourceModulesPath);
      if (resolve(sourceModulesRoot) !== resolve(sourceModulesPath)) {
        throw new Error("The authorized node_modules root resolved outside its exact path.");
      }
      const candidateModulesPath = join(candidateRoot, "node_modules");
      try {
        await lstat(candidateModulesPath);
        throw new Error("The isolated candidate already contains node_modules.");
      } catch (error) {
        if ((error as NodeJS.ErrnoException).code !== "ENOENT") throw error;
      }
      await symlink(
        sourceModulesRoot,
        candidateModulesPath,
        process.platform === "win32" ? "junction" : "dir"
      );
      try {
        const [linked, linkedRoot] = await Promise.all([
          lstat(candidateModulesPath),
          realpath(candidateModulesPath)
        ]);
        if (
          !linked.isSymbolicLink() ||
          resolve(linkedRoot) !== resolve(sourceModulesRoot)
        ) {
          throw new Error(
            "Candidate dependency linkage did not resolve to the authorized repository."
          );
        }
        return await operation();
      } finally {
        const linked = await lstat(candidateModulesPath).catch((error) => {
          if ((error as NodeJS.ErrnoException).code === "ENOENT") return undefined;
          throw error;
        });
        if (linked) {
          if (!linked.isSymbolicLink()) {
            throw new Error(
              "Candidate dependency link was replaced during execution and was not removed."
            );
          }
          const linkedRoot = await realpath(candidateModulesPath);
          if (resolve(linkedRoot) !== resolve(sourceModulesRoot)) {
            throw new Error(
              "Candidate dependency link target changed during execution and was not removed."
            );
          }
          await unlink(candidateModulesPath);
        }
      }
    };

    const diffSnapshot = async (
      worktree: string,
      parentCommit: string
    ): Promise<{
      status: string;
      diff: string;
      untracked: Array<{ path: string; sha256: string; bytes: number }>;
      changedPaths: string[];
      changedBytes: number;
      sha256: string;
    }> => {
      const [status, diff, changedList, untrackedList] = await Promise.all([
        runEvolutionProcess("git", ["-C", worktree, "status", "--short"], worktree, 30_000),
        runEvolutionProcess(
          "git",
          ["-C", worktree, "diff", "--no-ext-diff", "--no-renames", "--binary", parentCommit],
          worktree,
          60_000
        ),
        runEvolutionProcess(
          "git",
          [
            "-C",
            worktree,
            "diff",
            "--no-ext-diff",
            "--no-renames",
            "--name-only",
            "-z",
            parentCommit
          ],
          worktree,
          30_000
        ),
        runEvolutionProcess(
          "git",
          ["-C", worktree, "ls-files", "--others", "--exclude-standard", "-z"],
          worktree,
          30_000
        )
      ]);
      if (
        status.exitCode !== 0 ||
        diff.exitCode !== 0 ||
        changedList.exitCode !== 0 ||
        untrackedList.exitCode !== 0
      ) {
        throw new Error(
          `Could not inspect evolution candidate: ${
            status.stderr || diff.stderr || changedList.stderr || untrackedList.stderr
          }`
        );
      }
      const untracked = [];
      let untrackedBytes = 0;
      for (const rawPath of untrackedList.stdout.split("\0").filter(Boolean)) {
        assertToolActive(signal);
        const path = resolve(worktree, rawPath);
        const fromRoot = relative(worktree, path);
        if (fromRoot.startsWith("..") || isAbsolute(fromRoot)) {
          throw new Error("Git reported an unsafe untracked path.");
        }
        const info = await lstat(path);
        if (!info.isFile() || info.isSymbolicLink()) {
          throw new Error("Evolution candidates may not promote untracked links or special files.");
        }
        untrackedBytes += info.size;
        if (untrackedBytes > 64 * 1024 * 1024) {
          throw new Error("Untracked evolution files exceed the 64 MB validation limit.");
        }
        const contents = await readFile(path);
        untracked.push({ path: rawPath, sha256: sha256(contents), bytes: contents.byteLength });
      }
      const changedPaths = [
        ...new Set([
          ...changedList.stdout.split("\0").filter(Boolean),
          ...untracked.map((entry) => entry.path)
        ])
      ].sort();
      const changedBytes =
        Buffer.byteLength(diff.stdout) +
        untracked.reduce((total, entry) => total + entry.bytes, 0);
      const material =
        `${parentCommit}\n${status.stdout}\n${diff.stdout}\n` +
        `${JSON.stringify(untracked)}\n${JSON.stringify(changedPaths)}`;
      return {
        status: status.stdout,
        diff: diff.stdout,
        untracked,
        changedPaths,
        changedBytes,
        sha256: sha256(material)
      };
    };

    const applySourceEdits = async (
      worktree: string,
      baselineTestPaths: ReadonlySet<string>,
      edits: SourceEditRequest[]
    ): Promise<SourceEditLineageRecord[]> => {
      const root = await realpath(worktree);
      const inspectTarget = async (
        edit: SourceEditRequest
      ): Promise<{
        target: string;
        exists: boolean;
        mode: number;
        currentSha256: string | null;
      }> => {
        if (isProtectedEvolutionPath(edit.path, baselineTestPaths)) {
          throw new Error(`Source edit targets an immutable evaluator file: ${edit.path}`);
        }
        const target = resolve(root, ...edit.path.split("/"));
        const fromRoot = relative(root, target);
        if (!fromRoot || fromRoot.startsWith("..") || isAbsolute(fromRoot)) {
          throw new Error("Source edit path escaped the isolated worktree.");
        }
        let cursor = root;
        for (const segment of edit.path.split("/").slice(0, -1)) {
          cursor = join(cursor, segment);
          try {
            const info = await lstat(cursor);
            if (info.isSymbolicLink()) {
              throw new Error(`Source edit parent is a symbolic link: ${edit.path}`);
            }
            if (!info.isDirectory()) {
              throw new Error(`Source edit parent is not a directory: ${edit.path}`);
            }
          } catch (error) {
            if ((error as NodeJS.ErrnoException).code === "ENOENT") break;
            throw error;
          }
        }
        try {
          const info = await lstat(target);
          if (info.isSymbolicLink()) {
            throw new Error(`Source edits may not replace symbolic links: ${edit.path}`);
          }
          if (!info.isFile()) {
            throw new Error(`Source edit target is not a regular file: ${edit.path}`);
          }
          if (info.size > MAX_SOURCE_EDIT_TOTAL_BYTES) {
            throw new Error(
              `Existing source file exceeds the ${MAX_SOURCE_EDIT_TOTAL_BYTES} byte authoring limit.`
            );
          }
          const current = await readFile(target);
          if (!isUtf8(current) || current.includes(0)) {
            throw new Error(`Source edit target is not valid UTF-8 text: ${edit.path}`);
          }
          return {
            target,
            exists: true,
            mode: info.mode & 0o777,
            currentSha256: sha256(current)
          };
        } catch (error) {
          if ((error as NodeJS.ErrnoException).code !== "ENOENT") throw error;
          return {
            target,
            exists: false,
            mode: 0o600,
            currentSha256: null
          };
        }
      };

      const assertExpectedState = (
        edit: SourceEditRequest,
        inspected: Awaited<ReturnType<typeof inspectTarget>>
      ): void => {
        if (edit.expectedSha256 === null) {
          if (inspected.exists) {
            throw new Error(
              `Source edit expected a new file but the path already exists: ${edit.path}`
            );
          }
          return;
        }
        if (!inspected.exists) {
          throw new Error(
            `Source edit expected an existing file but the path is absent: ${edit.path}`
          );
        }
        if (inspected.currentSha256 !== edit.expectedSha256) {
          throw new Error(`Source edit checksum does not match current content: ${edit.path}`);
        }
      };

      const prepared: Array<{
        edit: SourceEditRequest;
        target: string;
        temporary: string;
      }> = [];
      try {
        for (const edit of edits) {
          assertToolActive(signal);
          const inspected = await inspectTarget(edit);
          assertExpectedState(edit, inspected);
          await mkdir(dirname(inspected.target), { recursive: true });
          const realParent = await realpath(dirname(inspected.target));
          const parentFromRoot = relative(root, realParent);
          if (parentFromRoot.startsWith("..") || isAbsolute(parentFromRoot)) {
            throw new Error("Source edit parent escaped the isolated worktree.");
          }
          const temporary = join(
            realParent,
            `.${basename(inspected.target)}.${randomUUID()}.omni-source-next`
          );
          await writeFile(temporary, edit.content, {
            encoding: "utf8",
            flag: "wx",
            mode: inspected.mode || 0o600
          });
          prepared.push({ edit, target: inspected.target, temporary });
        }

        // Recheck every compare-and-write precondition after all temporary
        // files exist. The worktree is discarded if any check or rename fails,
        // so no partially authored candidate is ever returned to a caller.
        for (const entry of prepared) {
          assertToolActive(signal);
          assertExpectedState(entry.edit, await inspectTarget(entry.edit));
        }
        for (const entry of prepared) {
          assertToolActive(signal);
          assertExpectedState(entry.edit, await inspectTarget(entry.edit));
          await rename(entry.temporary, entry.target);
        }

        const lineage: SourceEditLineageRecord[] = [];
        for (const edit of edits) {
          const target = resolve(root, ...edit.path.split("/"));
          const contents = await readFile(target);
          if (!isUtf8(contents) || contents.includes(0)) {
            throw new Error(`Authored source is not valid UTF-8 text: ${edit.path}`);
          }
          lineage.push({
            path: edit.path,
            expectedSha256: edit.expectedSha256,
            resultSha256: sha256(contents),
            bytes: contents.byteLength
          });
        }
        return lineage.sort((left, right) => left.path.localeCompare(right.path));
      } finally {
        await Promise.all(
          prepared.map((entry) => unlink(entry.temporary).catch(() => undefined))
        );
      }
    };

    const verifyEvaluatorBoundary = async (
      worktree: string,
      proposal: EvolutionProposalRecord,
      expectedEvaluatorSha256?: unknown
    ): Promise<void> => {
      if (
        typeof expectedEvaluatorSha256 === "string" &&
        expectedEvaluatorSha256.toLocaleLowerCase() !==
          proposal.evaluatorSha256.toLocaleLowerCase()
      ) {
        throw new Error("The candidate evaluator hash does not match its immutable proposal.");
      }
      const ancestry = await runEvolutionProcess(
        "git",
        ["-C", worktree, "merge-base", "--is-ancestor", proposal.parentCommit, "HEAD"],
        worktree,
        30_000
      );
      if (ancestry.exitCode !== 0) {
        throw new Error("The candidate no longer descends from its recorded parent commit.");
      }
      const currentScripts = await readPackageScripts(worktree);
      if (JSON.stringify(currentScripts) !== JSON.stringify(proposal.packageScripts)) {
        throw new Error("Evolution candidates may not rewrite benchmark package scripts.");
      }
      const snapshot = await diffSnapshot(worktree, proposal.parentCommit);
      const baselineTests = new Set(proposal.baselineTestPaths);
      const protectedChanges = snapshot.changedPaths.filter((path) =>
        isProtectedEvolutionPath(path, baselineTests)
      );
      if (protectedChanges.length) {
        throw new Error(
          `Evolution candidates may not modify immutable evaluator files: ${protectedChanges.join(", ")}`
        );
      }
    };

    if (action === "propose") {
      const objective = argumentString(args, "objective", 20_000);
      const sourceEdits = sourceEditRequests(args.sourceEdits);
      const [parentCommitResult, baselineTestsResult, packageScripts] = await Promise.all([
        runEvolutionProcess(
          "git",
          ["-C", repository, "rev-parse", "HEAD"],
          repository,
          30_000
        ),
        runEvolutionProcess(
          "git",
          ["-C", repository, "ls-tree", "-r", "--name-only", "HEAD", "--", "tests"],
          repository,
          30_000
        ),
        readPackageScripts(repository)
      ]);
      if (parentCommitResult.exitCode !== 0 || baselineTestsResult.exitCode !== 0) {
        throw new Error(
          `Could not anchor the evolution evaluator: ${
            parentCommitResult.stderr || baselineTestsResult.stderr
          }`
        );
      }
      const parentCommit = parentCommitResult.stdout.trim().toLocaleLowerCase();
      const baselineTestPaths = baselineTestsResult.stdout
        .split(/\r?\n/)
        .map((path) => path.trim())
        .filter(Boolean)
        .sort();
      const evaluatorSha256 = sha256(
        [
          EVOLUTION_POLICY_SHA256,
          parentCommit,
          JSON.stringify(baselineTestPaths),
          JSON.stringify(packageScripts)
        ].join("\n")
      );
      const identifier = randomUUID().slice(0, 12);
      const branch = `omni-evolution/${identifier}`;
      const worktree = join(evolutionRoot, identifier);
      const created = await runEvolutionProcess(
        "git",
        ["-C", repository, "worktree", "add", "-b", branch, worktree, "HEAD"],
        repository,
        120_000
      );
      if (created.exitCode !== 0) throw new Error(`Git worktree creation failed: ${created.stderr}`);
      const taskFile = join(evolutionRoot, `${identifier}.task.md`);
      try {
        const sourceEditLineage = await applySourceEdits(
          worktree,
          new Set(baselineTestPaths),
          sourceEdits
        );
        const authored = await diffSnapshot(worktree, parentCommit);
        const requestedPaths = sourceEdits.map((edit) => edit.path).sort();
        if (
          JSON.stringify(authored.changedPaths) !== JSON.stringify(requestedPaths)
        ) {
          throw new Error(
            "Typed source authoring changed paths outside its declared edit set."
          );
        }
        const authoredBytes = sourceEditLineage.reduce(
          (total, entry) => total + entry.bytes,
          0
        );
        await atomicWrite(
          taskFile,
          `# OmniCortex source-evolution candidate\n\n${objective}\n\n` +
            `${sourceEditLineage.length} typed source edit(s) were authored in this isolated worktree. ` +
            "Run diff and test before requesting promotion.\n"
        );
        const externalProposal: EvolutionProposalRecord = {
          schemaVersion: 1,
          worktree,
          branch,
          parentCommit,
          evaluatorVersion: EVOLUTION_EVALUATOR_VERSION,
          evaluatorSha256,
          baselineTestPaths,
          packageScripts,
          sourceEditLineage,
          authoredChangedPaths: authored.changedPaths,
          authoredDiffSha256: authored.sha256,
          authoredBytes,
          createdAt: new Date().toISOString()
        };
        await atomicWrite(proposalPath(worktree), JSON.stringify(externalProposal, null, 2));
        return {
          worktree,
          branch,
          taskFile,
          parentCommit,
          evaluatorVersion: EVOLUTION_EVALUATOR_VERSION,
          evaluatorSha256,
          benchmarkDomains: EVOLUTION_BENCHMARK_DOMAINS,
          protectedPaths: EVOLUTION_PROTECTED_PATHS,
          sourceEditLineage,
          authoredChangedPaths: authored.changedPaths,
          authoredDiffSha256: authored.sha256,
          authoredBytes,
          diff: authored.diff,
          checks: [
            {
              name: "isolated-worktree",
              passed: true,
              detail: "No running application files were overwritten."
            },
            {
              name: "typed-source-authoring",
              passed: sourceEditLineage.length > 0,
              detail: sourceEditLineage.length
                ? `${sourceEditLineage.length} compare-and-write text edit(s) were applied.`
                : "No typed source edits were provided; an empty candidate cannot be promoted."
            },
            {
              name: "promotion",
              passed: false,
              detail: "Promotion requires a non-empty diff, passing tests, and exact authorization."
            }
          ]
        };
      } catch (error) {
        await Promise.all([
          unlink(taskFile).catch(() => undefined),
          unlink(proposalPath(worktree)).catch(() => undefined)
        ]);
        await runProcess(
          "git",
          ["-C", repository, "worktree", "remove", "--force", worktree],
          repository,
          60_000
        ).catch(() => undefined);
        await runProcess(
          "git",
          ["-C", repository, "branch", "-D", branch],
          repository,
          30_000
        ).catch(() => undefined);
        throw error;
      }
    }

    if (action === "diff") {
      const worktree = await resolveCandidate();
      const proposal = await readProposal(worktree);
      return {
        worktree,
        parentCommit: proposal.parentCommit,
        evaluatorSha256: proposal.evaluatorSha256,
        ...(await diffSnapshot(worktree, proposal.parentCommit))
      };
    }

    if (action === "test") {
      const worktree = await resolveCandidate();
      const proposal = await readProposal(worktree);
      await verifyEvaluatorBoundary(worktree, proposal, args.expectedEvaluatorSha256);
      const requested = Array.isArray(args.tests)
        ? args.tests.filter((value): value is string => typeof value === "string").slice(0, 3)
        : [];
      const names = requested.length ? requested : ["typecheck", "unit", "build"];
      const allowed = new Set<string>(EVOLUTION_TEST_NAMES);
      if (names.some((name) => !allowed.has(name))) {
        throw new Error("Evolution tests may be typecheck, unit, or build.");
      }
      const executable =
        process.platform === "win32"
          ? (process.env.ComSpec?.trim() || "cmd.exe")
          : "npm";
      const commands: Record<string, string[]> = {
        typecheck: ["run", "typecheck"],
        unit: ["test"],
        build: ["run", "build"]
      };
      const checks: Array<{
        name: string;
        passed: boolean;
        exitCode: number;
        durationMs: number;
        stdout: string;
        stderr: string;
        truncated: boolean;
      }> = [];
      const baselineChecks: Array<{
        name: string;
        passed: boolean;
        exitCode: number;
        durationMs: number;
        stdout: string;
        stderr: string;
        truncated: boolean;
      }> = [];
      await withCandidateDependencies(worktree, async () => {
        for (const name of names) {
          const commandArgs =
            process.platform === "win32"
              ? ["/d", "/c", "npm.cmd", ...commands[name]!]
              : commands[name]!;
          const environment = {
            npm_config_ignore_scripts: "true",
            NPM_CONFIG_IGNORE_SCRIPTS: "true"
          };
          const baselineStartedAt = Date.now();
          const baseline = await runProcess(
            executable,
            commandArgs,
            repository,
            boundedTimeout(args.timeoutMs, 600_000),
            { ...environment, OMNI_EVOLUTION_CANDIDATE_WORKTREE: "0" },
            signal
          );
          baselineChecks.push({
            name,
            passed: baseline.exitCode === 0,
            exitCode: baseline.exitCode,
            durationMs: Date.now() - baselineStartedAt,
            stdout: baseline.stdout,
            stderr: baseline.stderr,
            truncated: baseline.truncated
          });
          const candidateStartedAt = Date.now();
          const result = await runProcess(
            executable,
            commandArgs,
            worktree,
            boundedTimeout(args.timeoutMs, 600_000),
            { ...environment, OMNI_EVOLUTION_CANDIDATE_WORKTREE: "1" },
            signal
          );
          checks.push({
            name,
            passed: result.exitCode === 0,
            exitCode: result.exitCode,
            durationMs: Date.now() - candidateStartedAt,
            stdout: result.stdout,
            stderr: result.stderr,
            truncated: result.truncated
          });
          if (result.exitCode !== 0) break;
        }
      });
      const diffCheck = await runEvolutionProcess(
        "git",
        ["-C", worktree, "diff", "--check"],
        worktree,
        60_000
      );
      checks.push({
        name: "diff-check",
        passed: diffCheck.exitCode === 0,
        exitCode: diffCheck.exitCode,
        durationMs: 0,
        stdout: diffCheck.stdout,
        stderr: diffCheck.stderr,
        truncated: diffCheck.truncated
      });
      const snapshot = await diffSnapshot(worktree, proposal.parentCommit);
      checks.push({
        name: "candidate-change",
        passed: snapshot.changedPaths.length > 0,
        exitCode: snapshot.changedPaths.length > 0 ? 0 : 1,
        durationMs: 0,
        stdout:
          snapshot.changedPaths.length > 0
            ? `${snapshot.changedPaths.length} source path(s) changed.`
            : "",
        stderr:
          snapshot.changedPaths.length > 0
            ? ""
            : "An empty source candidate cannot pass evaluation or be promoted.",
        truncated: false
      });
      const passed = checks.every((check) => check.passed);
      const baselineDurationMs = baselineChecks.reduce(
        (total, check) => total + check.durationMs,
        0
      );
      const candidateDurationMs = checks.reduce(
        (total, check) => total + check.durationMs,
        0
      );
      const regressions = checks
        .filter((check) => {
          const baseline = baselineChecks.find((entry) => entry.name === check.name);
          return Boolean(baseline?.passed && !check.passed);
        })
        .map((check) => check.name);
      const resources = {
        baselineDurationMs,
        candidateDurationMs,
        durationDeltaMs: candidateDurationMs - baselineDurationMs,
        changedBytes: snapshot.changedBytes,
        changedPaths: snapshot.changedPaths.length,
        untrackedBytes: snapshot.untracked.reduce((total, entry) => total + entry.bytes, 0)
      };
      const validationPath = join(evolutionRoot, `${basename(worktree)}.validation.json`);
      await writeFile(
        validationPath,
        JSON.stringify(
          {
            worktree,
            parentCommit: proposal.parentCommit,
            evaluatorVersion: proposal.evaluatorVersion,
            evaluatorSha256: proposal.evaluatorSha256,
            diffSha256: snapshot.sha256,
            passed,
            boundaryPassed: true,
            checks: checks.map(({ name, passed: checkPassed, exitCode }) => ({
              name,
              passed: checkPassed,
              exitCode
            })),
            baselineChecks: baselineChecks.map(
              ({ name, passed: checkPassed, exitCode, durationMs }) => ({
                name,
                passed: checkPassed,
                exitCode,
                durationMs
              })
            ),
            regressions,
            resources,
            createdAt: new Date().toISOString()
          },
          null,
          2
        ),
        { encoding: "utf8", mode: 0o600 }
      );
      return {
        worktree,
        parentCommit: proposal.parentCommit,
        evaluatorVersion: proposal.evaluatorVersion,
        evaluatorSha256: proposal.evaluatorSha256,
        benchmarkDomains: EVOLUTION_BENCHMARK_DOMAINS,
        boundaryPassed: true,
        passed,
        diffSha256: snapshot.sha256,
        validationPath,
        checks,
        baselineChecks,
        regressions,
        resources
      };
    }

    if (action === "promote") {
      const worktree = await resolveCandidate();
      const proposal = await readProposal(worktree);
      await verifyEvaluatorBoundary(worktree, proposal, args.expectedEvaluatorSha256);
      const expected = argumentString(args, "expectedDiffSha256", 64).toLocaleLowerCase();
      if (!/^[a-f0-9]{64}$/.test(expected)) {
        throw new Error("expectedDiffSha256 must be a SHA-256 digest from source.self-modify.test.");
      }
      const snapshot = await diffSnapshot(worktree, proposal.parentCommit);
      if (snapshot.changedPaths.length === 0) {
        throw new Error("An empty source candidate cannot be promoted.");
      }
      if (snapshot.sha256 !== expected) {
        throw new Error("Evolution candidate changed after validation; test it again.");
      }
      const validationPath = join(evolutionRoot, `${basename(worktree)}.validation.json`);
      const validation = JSON.parse(await readFile(validationPath, "utf8")) as {
        worktree?: string;
        parentCommit?: string;
        evaluatorSha256?: string;
        diffSha256?: string;
        passed?: boolean;
        boundaryPassed?: boolean;
      };
      if (
        validation.passed !== true ||
        validation.boundaryPassed !== true ||
        validation.diffSha256 !== expected ||
        validation.parentCommit !== proposal.parentCommit ||
        validation.evaluatorSha256 !== proposal.evaluatorSha256 ||
        resolve(validation.worktree ?? "") !== resolve(worktree)
      ) {
        throw new Error("Evolution candidate has no matching passing validation record.");
      }
      const repositoryStatus = await runEvolutionProcess(
        "git",
        ["-C", repository, "status", "--porcelain"],
        repository,
        30_000
      );
      if (repositoryStatus.exitCode !== 0 || repositoryStatus.stdout.trim()) {
        throw new Error("Authorized source repository must be clean before promotion.");
      }
      const parentCommit = await runEvolutionProcess(
        "git",
        ["-C", repository, "rev-parse", "HEAD"],
        repository,
        30_000
      );
      if (parentCommit.exitCode !== 0) {
        throw new Error(`Could not resolve the promotion parent: ${parentCommit.stderr}`);
      }
      if (
        parentCommit.stdout.trim().toLocaleLowerCase() !==
        proposal.parentCommit.toLocaleLowerCase()
      ) {
        throw new Error("Authorized source changed after the candidate was forked; rebase and retest it.");
      }
      let runtimeStage: SourceRuntimeStageResult | undefined;
      let activationSnapshotId: string | undefined;
      let currentExecutablePath: string | undefined;
      let currentExecutableSha256: string | undefined;
      if (permission === "full") {
        if (!this.sourceRuntime) {
          throw new Error(
            "Full-Authority source promotion requires an injected side-by-side runtime lifecycle."
          );
        }
        if (
          !["win32", "darwin", "linux"].includes(process.platform) ||
          !["x64", "arm64"].includes(process.arch)
        ) {
          throw new Error(
            `Full-Authority runtime activation is unsupported on ${process.platform}-${process.arch}.`
          );
        }
        const snapshotSummary = await this.service.repository.snapshot(
          brainId,
          `Before Full-Authority source activation ${basename(worktree)}`
        );
        activationSnapshotId = snapshotSummary.id;
        currentExecutablePath = await realpath(process.execPath);
        currentExecutableSha256 = (await sha256File(currentExecutablePath)).sha256;
        let staged: SourceRuntimeStageResult | undefined;
        try {
          staged = await withCandidateDependencies(worktree, () =>
            this.sourceRuntime!.stage(
              {
                brainId,
                authorizedRepository: repository,
                worktree,
                parentCommit: proposal.parentCommit,
                diffSha256: expected,
                evaluatorSha256: proposal.evaluatorSha256,
                changedPaths: snapshot.changedPaths,
                brainSnapshotId: activationSnapshotId!,
                currentExecutablePath: currentExecutablePath!,
                platform: process.platform as "win32" | "darwin" | "linux",
                architecture: process.arch as "x64" | "arm64"
              },
              signal
            )
          );
          runtimeStage = await this.verifyRuntimeStage(staged, {
            repository,
            worktree,
            parentCommit: proposal.parentCommit,
            diffSha256: expected,
            evaluatorSha256: proposal.evaluatorSha256,
            brainSnapshotId: activationSnapshotId,
            currentExecutablePath,
            currentExecutableSha256
          });
          await verifyEvaluatorBoundary(
            worktree,
            proposal,
            args.expectedEvaluatorSha256
          );
          const afterStage = await diffSnapshot(worktree, proposal.parentCommit);
          if (afterStage.sha256 !== expected) {
            throw new Error(
              "Runtime staging changed the validated source candidate; promotion was not merged."
            );
          }
        } catch (error) {
          if (staged) {
            await this.sourceRuntime
              .abandonStage?.(
                staged,
                error instanceof Error ? error.message : String(error)
              )
              .catch(() => undefined);
          }
          throw error;
        }
      }
      try {
        await unlink(join(worktree, "OMNI_EVOLUTION_TASK.md")).catch(() => undefined);
        const added = await runEvolutionProcess(
          "git",
          ["-C", worktree, "add", "-A"],
          worktree,
          60_000
        );
        if (added.exitCode !== 0) {
          throw new Error(`Could not stage candidate: ${added.stderr}`);
        }
        const committed = await runEvolutionProcess(
          "git",
          [
            "-C",
            worktree,
            "-c",
            "user.name=OmniCortex Evolution",
            "-c",
            "user.email=omni-evolution@local.invalid",
            "commit",
            "-m",
            `Promote OmniCortex evolution ${basename(worktree)}`
          ],
          worktree,
          120_000
        );
        if (committed.exitCode !== 0) {
          throw new Error(
            `Candidate commit failed (an empty candidate cannot be promoted): ${committed.stderr}`
          );
        }
        const commit = await runEvolutionProcess(
          "git",
          ["-C", worktree, "rev-parse", "HEAD"],
          worktree,
          30_000
        );
        if (commit.exitCode !== 0) {
          throw new Error(`Could not resolve candidate commit: ${commit.stderr}`);
        }
        let merged: Awaited<ReturnType<typeof runProcess>>;
        try {
          merged = await runEvolutionProcess(
            "git",
            [
              "-C",
              repository,
              "merge",
              "--no-ff",
              "-m",
              `Promote OmniCortex evolution ${basename(worktree)}`,
              commit.stdout.trim()
            ],
            repository,
            120_000
          );
        } catch (error) {
          await runProcess(
            "git",
            ["-C", repository, "merge", "--abort"],
            repository,
            30_000
          ).catch(() => undefined);
          throw error;
        }
        if (merged.exitCode !== 0) {
          await runProcess(
            "git",
            ["-C", repository, "merge", "--abort"],
            repository,
            30_000
          );
          throw new Error(
            `Candidate promotion failed and was aborted: ${merged.stderr}`
          );
        }
        const promotionCommit = await runEvolutionProcess(
          "git",
          ["-C", repository, "rev-parse", "HEAD"],
          repository,
          30_000
        );
        if (promotionCommit.exitCode !== 0) {
          throw new Error(
            `Could not resolve the promotion commit: ${promotionCommit.stderr}`
          );
        }
        let runtimeActivation: SourceRuntimeActivationResult | undefined;
        if (
          permission === "full" &&
          runtimeStage &&
          currentExecutablePath &&
          currentExecutableSha256
        ) {
          const scheduled = await this.sourceRuntime!.scheduleActivation({
            stage: runtimeStage,
            promotionCommit: promotionCommit.stdout.trim(),
            candidateCommit: commit.stdout.trim(),
            delayMs: 5_000
          });
          runtimeActivation = await this.verifyRuntimeActivation(
            scheduled,
            runtimeStage,
            promotionCommit.stdout.trim(),
            currentExecutablePath,
            currentExecutableSha256
          );
        }
        return {
          worktree,
          promoted: true,
          commit: promotionCommit.stdout.trim(),
          candidateCommit: commit.stdout.trim(),
          parentCommit: parentCommit.stdout.trim(),
          diffSha256: expected,
          evaluatorSha256: proposal.evaluatorSha256,
          brainSnapshotId: activationSnapshotId,
          runtimeActivation,
          note: runtimeActivation
            ? runtimeActivation.state === "scheduled"
              ? "Source was merged; a verified side-by-side runtime restart was scheduled without replacing the running executable."
              : "Source was merged; the verified side-by-side runtime is safely deferred in this development/test host."
            : "Source was merged; Ask/Auto promotion did not activate or replace the running executable."
        };
      } catch (error) {
        if (runtimeStage) {
          await this.sourceRuntime
            ?.abandonStage?.(
              runtimeStage,
              error instanceof Error ? error.message : String(error)
            )
            .catch(() => undefined);
        }
        throw error;
      }
    }

    if (action === "rollback") {
      const expectedCommit = argumentString(args, "expectedCommit", 64).toLocaleLowerCase();
      const parentCommit = argumentString(args, "parentCommit", 64).toLocaleLowerCase();
      if (
        !/^[a-f0-9]{40,64}$/.test(expectedCommit) ||
        !/^[a-f0-9]{40,64}$/.test(parentCommit)
      ) {
        throw new Error("Rollback requires exact promotion and parent commit hashes.");
      }
      const [repositoryStatus, head, firstParent] = await Promise.all([
        runEvolutionProcess(
          "git",
          ["-C", repository, "status", "--porcelain"],
          repository,
          30_000
        ),
        runEvolutionProcess(
          "git",
          ["-C", repository, "rev-parse", "HEAD"],
          repository,
          30_000
        ),
        runEvolutionProcess(
          "git",
          ["-C", repository, "rev-parse", `${expectedCommit}^1`],
          repository,
          30_000
        )
      ]);
      if (repositoryStatus.exitCode !== 0 || repositoryStatus.stdout.trim()) {
        throw new Error("Authorized source repository must be clean before rollback.");
      }
      if (head.exitCode !== 0 || head.stdout.trim().toLocaleLowerCase() !== expectedCommit) {
        throw new Error("Rollback only applies when the exact promoted commit is current.");
      }
      if (
        firstParent.exitCode !== 0 ||
        firstParent.stdout.trim().toLocaleLowerCase() !== parentCommit
      ) {
        throw new Error("Rollback lineage does not match the archived promotion.");
      }
      const reverted = await runEvolutionProcess(
        "git",
        [
          "-C",
          repository,
          "-c",
          "user.name=OmniCortex Evolution",
          "-c",
          "user.email=omni-evolution@local.invalid",
          "revert",
          "--no-edit",
          "-m",
          "1",
          expectedCommit
        ],
        repository,
        120_000
      );
      if (reverted.exitCode !== 0) {
        await runProcess(
          "git",
          ["-C", repository, "revert", "--abort"],
          repository,
          30_000
        ).catch(() => undefined);
        throw new Error(`Evolution rollback failed and was aborted: ${reverted.stderr}`);
      }
      const rollbackCommit = await runEvolutionProcess(
        "git",
        ["-C", repository, "rev-parse", "HEAD"],
        repository,
        30_000
      );
      if (rollbackCommit.exitCode !== 0) {
        throw new Error(`Could not resolve the rollback commit: ${rollbackCommit.stderr}`);
      }
      return {
        rolledBack: true,
        revertedCommit: expectedCommit,
        rollbackCommit: rollbackCommit.stdout.trim(),
        note: "Promotion was reverted with an auditable commit; history was not rewritten."
      };
    }

    throw new Error("Unknown source evolution action.");
  }

  private async audit(
    invocation: ToolInvocation,
    result: ToolExecutionResult
  ): Promise<void> {
    return withBrainWrite(this.service.repository, invocation.brainId, () =>
      this.auditUnlocked(invocation, result)
    );
  }

  private async auditUnlocked(
    invocation: ToolInvocation,
    result: ToolExecutionResult
  ): Promise<void> {
    const brain = await this.service.repository.get(invocation.brainId);
    const paths = Object.fromEntries(
      ["path", "cwd", "entryPath", "inputPath"]
        .map((key) => [key, invocation.arguments[key]])
        .filter((entry): entry is [string, string] => typeof entry[1] === "string")
        .map(([key, value]) => [key, value.slice(0, 1_000)])
    );
    const urlValue = typeof invocation.arguments.url === "string"
      ? (() => {
          try {
            const value = new URL(invocation.arguments.url as string);
            return `${value.origin}${value.pathname}`.slice(0, 1_000);
          } catch {
            return "invalid-url";
          }
        })()
      : undefined;
    const outputRecord =
      typeof result.output === "object" && result.output !== null
        ? (result.output as Record<string, unknown>)
        : {};
    const detail = {
      argumentKeys: Object.keys(invocation.arguments).sort(),
      argumentSha256: sha256(JSON.stringify(invocation.arguments)),
      paths,
      url: urlValue,
      changedPath:
        invocation.toolId === "windows.files" && invocation.action === "write"
          ? paths.path
          : undefined,
      exitCode:
        typeof outputRecord.exitCode === "number" ? outputRecord.exitCode : undefined,
      outputSha256:
        typeof outputRecord.sha256 === "string" ? outputRecord.sha256 : undefined,
      artifactPath:
        typeof outputRecord.artifactPath === "string"
          ? outputRecord.artifactPath.slice(0, 1_000)
          : undefined,
      worktree:
        typeof outputRecord.worktree === "string"
          ? outputRecord.worktree.slice(0, 1_000)
          : undefined,
      error: result.error?.slice(0, 1_000)
    };
    brain.journal = [
      ...(brain.journal ?? []),
      {
        id: randomUUID(),
        createdAt: result.finishedAt ?? result.startedAt,
        kind: "tool",
        summary: `${invocation.toolId}.${invocation.action}: ${result.state}.`,
        detail: JSON.stringify(detail)
      }
    ];
    brain.traces = [
      ...(brain.traces ?? []),
      {
        id: result.id,
        createdAt: result.finishedAt ?? result.startedAt,
        input: `${invocation.toolId}.${invocation.action}`,
        seed: 0,
        runtime: "adaptive-core",
        activatedConcepts: [],
        recalledIdeas: [],
        driveScores: { novelty: 0, coherence: 0, curiosity: 0 },
        branches: 1,
        selectedBranch: 0,
        steps: [
          {
            stage: "tool-permission",
            detail: "Executed through the visible per-brain authority matrix.",
            value: result.state
          },
          {
            stage: "tool-invocation",
            detail: `${invocation.toolId}.${invocation.action}; arguments ${detail.argumentSha256.slice(0, 16)}…`,
            value:
              typeof detail.changedPath === "string"
                ? detail.changedPath
                : typeof detail.url === "string"
                  ? detail.url
                  : undefined
          },
          {
            stage: "tool-result",
            detail: result.error ?? "Tool action completed and its output remained inspectable.",
            value:
              typeof detail.outputSha256 === "string"
                ? `${detail.outputSha256.slice(0, 16)}…`
                : typeof detail.exitCode === "number"
                  ? `exit ${detail.exitCode}`
                  : result.state
          }
        ],
        note:
          "Operational tool trace; it records authority, arguments by digest, visible targets, and outcomes—not hidden reasoning."
      }
    ];
    await this.service.repository.save(brain);
  }
}
