import { createHash, randomUUID } from "node:crypto";
import { spawn, type ChildProcess } from "node:child_process";
import {
  lstat,
  mkdir,
  readFile,
  readdir,
  realpath,
  rename,
  stat,
  unlink,
  writeFile
} from "node:fs/promises";
import { basename, dirname, isAbsolute, join, relative, resolve } from "node:path";
import type {
  ToolExecutionResult,
  ToolInvocation,
  ToolPermissionLevel
} from "../shared/types";
import {
  readResponseBounded,
  safeFetch,
  type BrainService,
  type RuntimeJobManager
} from "./brainService";

const MAX_TEXT_BYTES = 8 * 1024 * 1024;
const MAX_PROCESS_OUTPUT = 2 * 1024 * 1024;
const MAX_WEB_BYTES = 8 * 1024 * 1024;

interface Approval {
  brainId: string;
  toolId: string;
  action: string;
  argumentSha256: string;
  expiresAt: number;
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
    toolId === "source.self-modify"
  );
}

function toolCancellationError(): Error {
  return new Error("Tool execution was cancelled.");
}

function assertToolActive(signal: AbortSignal): void {
  if (signal.aborted) throw toolCancellationError();
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
    private readonly jobs: RuntimeJobManager
  ) {}

  async execute(invocation: ToolInvocation): Promise<ToolExecutionResult> {
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
      const needsApproval =
        permission === "ask" ||
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
        output = await this.dispatch(invocation, controller.signal);
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

  private async dispatch(invocation: ToolInvocation, signal: AbortSignal): Promise<unknown> {
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
          signal
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
        return this.sourceEvolution(invocation.action, invocation.arguments, signal);
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
    const response = await safeFetch(requested, {
      signal: AbortSignal.any([
        signal,
        AbortSignal.timeout(boundedTimeout(args.timeoutMs, 120_000))
      ]),
      headers: { Accept: "text/html, text/plain;q=0.9" }
    });
    const contentType = response.headers.get("content-type") ?? "";
    if (!contentType.includes("text/html") && !contentType.includes("text/plain")) {
      throw new Error("Browser snapshots support HTML or plain-text pages.");
    }
    const contents = await readResponseBounded(response, 4 * 1024 * 1024);
    assertToolActive(signal);
    const source = contentType.includes("text/html")
      ? contents.toString("utf8")
      : `<pre>${contents
          .toString("utf8")
          .replaceAll("&", "&amp;")
          .replaceAll("<", "&lt;")
          .replaceAll(">", "&gt;")}</pre>`;
    const inert = buildInertBrowserDocument(source, response.url || requested.href);
    const { BrowserWindow } = await import("electron");
    const browser = new BrowserWindow({
      show: false,
      width: 1280,
      height: 900,
      backgroundColor: "#ffffff",
      webPreferences: {
        partition: `omni-browser-${randomUUID()}`,
        sandbox: true,
        contextIsolation: true,
        nodeIntegration: false,
        javascript: true,
        webSecurity: true
      }
    });
    const requestSession = browser.webContents.session;
    requestSession.webRequest.onBeforeRequest((details, callback) => {
      callback({
        cancel:
          !details.url.startsWith("data:") &&
          details.url !== "about:blank"
      });
    });
    browser.webContents.setWindowOpenHandler(() => ({ action: "deny" }));
    browser.webContents.on("will-navigate", (event) => event.preventDefault());
    const abort = (): void => {
      if (!browser.isDestroyed()) browser.destroy();
    };
    signal.addEventListener("abort", abort, { once: true });
    try {
      assertToolActive(signal);
      await browser.loadURL(
        `data:text/html;base64,${Buffer.from(inert.document).toString("base64")}`
      );
      assertToolActive(signal);
      const page = (await browser.webContents.executeJavaScript(
        `(() => ({
          title: document.title.slice(0, 1000),
          text: (document.body?.innerText || "").slice(0, 200000),
          links: Array.from(document.querySelectorAll("a[href]")).slice(0, 500).map((item) => ({
            label: (item.textContent || "").trim().slice(0, 500),
            href: item.getAttribute("href") || ""
          }))
        }))()`,
        true
      )) as { title?: string; text?: string; links?: Array<{ label: string; href: string }> };
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
        status: response.status,
        finalUrl: response.url,
        title: page.title ?? "",
        text: page.text ?? "",
        links: page.links ?? [],
        artifactPath,
        mode: "script-disabled-snapshot",
        note:
          "The public page was fetched with private-network and redirect checks, then rendered locally with scripts and live navigation disabled."
      };
    } finally {
      signal.removeEventListener("abort", abort);
      requestSession.webRequest.onBeforeRequest(null);
      if (!browser.isDestroyed()) browser.destroy();
    }
  }

  private async imagine(
    brainId: string,
    action: string,
    args: Record<string, unknown>,
    signal: AbortSignal
  ): Promise<unknown> {
    if (action !== "generate") throw new Error("Unknown imagination action.");
    const modality = argumentString(args, "modality", 16);
    if (!["image", "audio", "video"].includes(modality)) {
      throw new Error("Imagination modality must be image, audio, or video.");
    }
    const job = this.jobs.generate({
      brainId,
      modality: modality as "image" | "audio" | "video",
      conceptIds: Array.isArray(args.conceptIds)
        ? args.conceptIds.filter((value): value is string => typeof value === "string").slice(0, 128)
        : undefined,
      settings:
        typeof args.settings === "object" && args.settings !== null
          ? (args.settings as Record<string, string | number | boolean>)
          : undefined
    });
    const finished = await this.jobs.wait(
      job.id,
      signal,
      boundedTimeout(args.timeoutMs, 600_000)
    );
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
      const result = await this.service.chat(fork.id, objective, signal);
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

  private async sourceEvolution(
    action: string,
    args: Record<string, unknown>,
    signal: AbortSignal
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
      return candidate;
    };

    const diffSnapshot = async (
      worktree: string
    ): Promise<{
      status: string;
      diff: string;
      untracked: Array<{ path: string; sha256: string; bytes: number }>;
      sha256: string;
    }> => {
      const [status, diff, untrackedList] = await Promise.all([
        runEvolutionProcess("git", ["-C", worktree, "status", "--short"], worktree, 30_000),
        runEvolutionProcess(
          "git",
          ["-C", worktree, "diff", "--no-ext-diff", "--binary", "HEAD"],
          worktree,
          60_000
        ),
        runEvolutionProcess(
          "git",
          ["-C", worktree, "ls-files", "--others", "--exclude-standard", "-z"],
          worktree,
          30_000
        )
      ]);
      if (status.exitCode !== 0 || diff.exitCode !== 0 || untrackedList.exitCode !== 0) {
        throw new Error(
          `Could not inspect evolution candidate: ${status.stderr || diff.stderr || untrackedList.stderr}`
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
      const material = `${status.stdout}\n${diff.stdout}\n${JSON.stringify(untracked)}`;
      return { status: status.stdout, diff: diff.stdout, untracked, sha256: sha256(material) };
    };

    if (action === "propose") {
      const objective = argumentString(args, "objective", 20_000);
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
      await atomicWrite(
        taskFile,
        `# OmniCortex source-evolution candidate\n\n${objective}\n\n` +
          "Edits belong in the isolated worktree. Run diff and test before requesting promotion.\n"
      );
      return {
        worktree,
        branch,
        taskFile,
        diff: "",
        checks: [
          {
            name: "isolated-worktree",
            passed: true,
            detail: "No running application files were overwritten."
          },
          {
            name: "promotion",
            passed: false,
            detail: "Promotion requires a passing test action and a separate exact approval."
          }
        ]
      };
    }

    if (action === "diff") {
      const worktree = await resolveCandidate();
      return { worktree, ...(await diffSnapshot(worktree)) };
    }

    if (action === "test") {
      const worktree = await resolveCandidate();
      const requested = Array.isArray(args.tests)
        ? args.tests.filter((value): value is string => typeof value === "string").slice(0, 3)
        : [];
      const names = requested.length ? requested : ["typecheck", "unit", "build"];
      const allowed = new Set(["typecheck", "unit", "build"]);
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
      const checks = [];
      for (const name of names) {
        const commandArgs =
          process.platform === "win32"
            ? ["/d", "/c", "npm.cmd", ...commands[name]!]
            : commands[name]!;
        const result = await runEvolutionProcess(
          executable,
          commandArgs,
          worktree,
          boundedTimeout(args.timeoutMs, 600_000)
        );
        checks.push({
          name,
          passed: result.exitCode === 0,
          exitCode: result.exitCode,
          stdout: result.stdout,
          stderr: result.stderr,
          truncated: result.truncated
        });
        if (result.exitCode !== 0) break;
      }
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
        stdout: diffCheck.stdout,
        stderr: diffCheck.stderr,
        truncated: diffCheck.truncated
      });
      const snapshot = await diffSnapshot(worktree);
      const passed = checks.every((check) => check.passed);
      const validationPath = join(evolutionRoot, `${basename(worktree)}.validation.json`);
      await writeFile(
        validationPath,
        JSON.stringify(
          {
            worktree,
            diffSha256: snapshot.sha256,
            passed,
            checks: checks.map(({ name, passed: checkPassed, exitCode }) => ({
              name,
              passed: checkPassed,
              exitCode
            })),
            createdAt: new Date().toISOString()
          },
          null,
          2
        ),
        { encoding: "utf8", mode: 0o600 }
      );
      return { worktree, passed, diffSha256: snapshot.sha256, validationPath, checks };
    }

    if (action === "promote") {
      const worktree = await resolveCandidate();
      const expected = argumentString(args, "expectedDiffSha256", 64).toLocaleLowerCase();
      if (!/^[a-f0-9]{64}$/.test(expected)) {
        throw new Error("expectedDiffSha256 must be a SHA-256 digest from source.self-modify.test.");
      }
      const snapshot = await diffSnapshot(worktree);
      if (snapshot.sha256 !== expected) {
        throw new Error("Evolution candidate changed after validation; test it again.");
      }
      const validationPath = join(evolutionRoot, `${basename(worktree)}.validation.json`);
      const validation = JSON.parse(await readFile(validationPath, "utf8")) as {
        worktree?: string;
        diffSha256?: string;
        passed?: boolean;
      };
      if (
        validation.passed !== true ||
        validation.diffSha256 !== expected ||
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
      await unlink(join(worktree, "OMNI_EVOLUTION_TASK.md")).catch(() => undefined);
      const added = await runEvolutionProcess(
        "git",
        ["-C", worktree, "add", "-A"],
        worktree,
        60_000
      );
      if (added.exitCode !== 0) throw new Error(`Could not stage candidate: ${added.stderr}`);
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
        throw new Error(`Candidate commit failed (an empty candidate cannot be promoted): ${committed.stderr}`);
      }
      const commit = await runEvolutionProcess(
        "git",
        ["-C", worktree, "rev-parse", "HEAD"],
        worktree,
        30_000
      );
      if (commit.exitCode !== 0) throw new Error(`Could not resolve candidate commit: ${commit.stderr}`);
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
        await runProcess("git", ["-C", repository, "merge", "--abort"], repository, 30_000);
        throw new Error(`Candidate promotion failed and was aborted: ${merged.stderr}`);
      }
      return {
        worktree,
        promoted: true,
        commit: commit.stdout.trim(),
        diffSha256: expected,
        note: "Source was merged; the running binary was not overwritten or restarted."
      };
    }

    throw new Error("Unknown source evolution action.");
  }

  private async audit(
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
