import { createHash, randomBytes } from "node:crypto";
import { request as httpRequest } from "node:http";
import { createServer, type Socket } from "node:net";
import { lstat, mkdtemp, open, rm } from "node:fs/promises";
import { join } from "node:path";
import { tmpdir } from "node:os";
import type { LocalTeacherGeneration } from "../shared/localTeacher";
import { localTeacherCanonical, localTeacherHash } from "./localTeacherFormat";
import { localTeacherDirectory } from "./localTeacherArtifacts";
import { localTeacherBaseArguments, startLocalTeacherChild, type LocalTeacherCapturedFile, type LocalTeacherChild } from "./localTeacherRunner";
import { LocalTeacherSse } from "./localTeacherSse";

export type LocalTeacherEndpoint = { socketPath: string } | { hostname: "127.0.0.1"; port: number };
export interface LocalTeacherServerOptions {
  root: string;
  sessionId: string;
  executable: string;
  modelPath: string;
  generation: LocalTeacherGeneration;
  signal: AbortSignal;
  requireCurrent(): Promise<void>;
  checkResources(pids: readonly number[], additionalBytes?: number): Promise<void>;
  reserveLogWrite(bytes: number): Promise<void>;
  /** Windows must verify this connected socket's peer belongs to the live owned child. */
  verifyTcpOwner?(child: LocalTeacherChild, port: number, socket: Socket, signal: AbortSignal): Promise<void>;
  onPhase?(phase: string): void;
}
export interface LocalTeacherServerSession {
  sessionId: string;
  argvSha256: string;
  complete(prompt: string, attemptId: string, signal: AbortSignal): Promise<{ content: LocalTeacherCapturedFile; transport: LocalTeacherCapturedFile;
    final: Record<string, unknown>; elapsedNanoseconds: string; requestSha256: string }>;
  close(reason?: Error): Promise<Awaited<LocalTeacherChild["closed"]>>;
}
export class LocalTeacherHttpStatus extends Error {
  constructor(readonly statusCode: number | undefined) { super(`Selected local teacher returned HTTP ${statusCode ?? "unknown"}; no endpoint fallback was used.`); }
}
export class LocalTeacherCompletionError extends Error {
  constructor(message: string, readonly partial: { content: LocalTeacherCapturedFile; transport: LocalTeacherCapturedFile;
    elapsedNanoseconds: string; requestSha256: string }) { super(message); }
}
export const localTeacherCompletionBody = (prompt: string, generation: LocalTeacherGeneration): string => JSON.stringify({ prompt,
  n_predict: generation.maxOutputTokens, temperature: generation.temperature, seed: generation.seed, stream: true,
  cache_prompt: false, n_keep: -1, stop: [], return_tokens: false });
async function freeLoopbackPort(): Promise<number> {
  const server = createServer();
  return new Promise((resolve, reject) => { server.once("error", reject); server.listen(0, "127.0.0.1", () => {
    const address = server.address(); if (!address || typeof address === "string") { server.close(); reject(new Error("A private local teacher port could not be selected.")); return; }
    server.close(error => error ? reject(error) : resolve(address.port));
  }); });
}
function wait(milliseconds: number, signal: AbortSignal): Promise<void> {
  signal.throwIfAborted(); return new Promise((resolve, reject) => { const timer = setTimeout(() => { signal.removeEventListener("abort", abort); resolve(); }, milliseconds);
    const abort = () => { clearTimeout(timer); signal.removeEventListener("abort", abort); reject(signal.reason); }; signal.addEventListener("abort", abort, { once: true }); });
}

/** No ambient proxy/fetch dispatcher, redirects, provider SDK, template or system prompt. */
export async function localTeacherHttp(endpoint: LocalTeacherEndpoint, path: "/completion" | "/health", method: "POST" | "GET",
  key: string, signal: AbortSignal, beforeSend: (socket: Socket) => Promise<void>, body?: string): Promise<import("node:http").IncomingMessage> {
  signal.throwIfAborted();
  return new Promise((resolve, reject) => {
    const req = httpRequest({ ...endpoint, path, method, agent: false }, response => {
      if (response.statusCode !== 200) { response.resume(); reject(new LocalTeacherHttpStatus(response.statusCode)); return; }
      resolve(response);
    });
    const abort = () => req.destroy(signal.reason instanceof Error ? signal.reason : new Error("Local teacher request stopped."));
    signal.addEventListener("abort", abort, { once: true });
    req.once("error", reject); req.once("close", () => signal.removeEventListener("abort", abort));
    req.once("socket", socket => {
      socket.once("connect", () => {
        void beforeSend(socket).then(() => {
          signal.throwIfAborted();
          // Authentication/prompt bytes are not sent until the connected
          // endpoint is proven to be our exact current owned server.
          req.setHeader("Authorization", `Bearer ${key}`); req.setHeader("Content-Type", "application/json");
          if (body !== undefined) req.setHeader("Content-Length", Buffer.byteLength(body)); req.end(body);
        }).catch(error => req.destroy(error instanceof Error ? error : new Error(String(error))));
      });
    });
  });
}

export async function openLocalTeacherServer(options: LocalTeacherServerOptions): Promise<LocalTeacherServerSession> {
  if (process.platform === "win32" && !options.verifyTcpOwner) throw new Error("This host has no exact owned-TCP verifier; local teacher server is unavailable, not publicly bound.");
  // macOS per-user tmpdir paths can exceed sockaddr_un. /tmp is a real OS
  // temporary namespace; mkdtemp creates one exact unpredictable owned folder.
  const privateRoot = await mkdtemp(join(process.platform === "win32" ? tmpdir() : "/tmp", "omni-teacher-"));
  const authPath = join(privateRoot, "key"), key = randomBytes(32).toString("hex"), auth = await open(authPath, "wx", 0o600);
  try { await auth.writeFile(`${key}\n`); await auth.sync(); } finally { await auth.close(); }
  const endpoint: LocalTeacherEndpoint = process.platform === "win32" ? { hostname: "127.0.0.1", port: await freeLoopbackPort() }
    : { socketPath: join(privateRoot, "teacher.sock") };
  const host = "socketPath" in endpoint ? endpoint.socketPath : endpoint.hostname;
  const args = [...localTeacherBaseArguments(options.modelPath, options.generation), "--host", host, "--port", String("port" in endpoint ? endpoint.port : 0),
    "--api-key-file", authPath, "--parallel", "1", "--no-agent", "--no-webui", "--no-slots", "--no-cache-prompt", "--verbosity", "3"];
  let child: LocalTeacherChild | undefined, listenObserved = false, tail = "", socketIdentity: string | undefined;
  const requireCurrent = async (): Promise<void> => { options.signal.throwIfAborted(); await options.requireCurrent();
    if (!child?.alive()) throw new Error("The exact selected local teacher child is no longer live."); };
  const beforeSend = async (socket: Socket, signal = options.signal): Promise<void> => {
    signal.throwIfAborted(); await requireCurrent();
    if ("socketPath" in endpoint) {
      const item = await lstat(endpoint.socketPath, { bigint: true });
      if (!item.isSocket() || item.isSymbolicLink() || item.uid !== BigInt(process.getuid?.() ?? Number(item.uid))) throw new Error("Private local teacher socket ownership is unverifiable.");
      const identity = `${item.dev}:${item.ino}:${item.ctimeNs}`;
      if (socketIdentity && socketIdentity !== identity) throw new Error("The private local teacher socket changed."); socketIdentity ??= identity;
    } else {
      if (!listenObserved || !options.verifyTcpOwner || socket.remoteAddress !== "127.0.0.1" || socket.remotePort !== endpoint.port)
        throw new Error("The Windows teacher port is not bound to a verified current owned child; no prompt/auth was sent.");
      await options.verifyTcpOwner(child!, endpoint.port, socket, signal);
    }
    await requireCurrent(); signal.throwIfAborted();
  };
  try {
    options.onPhase?.("loading-local-teacher");
    child = await startLocalTeacherChild({ root: options.root, relativeDirectory: `local-teachers/attempts/${options.sessionId}`, executable: options.executable,
      arguments: args, signal: options.signal, maxLogBytes: null, reserveLogWrite: options.reserveLogWrite, timeoutMs: options.generation.timeoutMs,
      requireCurrent: options.requireCurrent, checkResources: options.checkResources,
      onLog: text => { tail = (tail + text).slice(-8192); if ("port" in endpoint && tail.includes(`listening on http://127.0.0.1:${endpoint.port}`)) listenObserved = true; } });
    let ready = false;
    while (!ready) {
      await requireCurrent();
      if ("port" in endpoint && !listenObserved) { await wait(200, options.signal); continue; }
      try {
        const response = await localTeacherHttp(endpoint, "/health", "GET", key, options.signal, beforeSend), chunks: Buffer[] = []; let bytes = 0;
        for await (const chunk of response) { const data = Buffer.from(chunk); bytes += data.length; if (bytes > 4096) throw new Error("Teacher health response exceeded its exact metadata envelope."); chunks.push(data); }
        const health = JSON.parse(new TextDecoder("utf-8", { fatal: true }).decode(Buffer.concat(chunks)));
        if (health.status !== "ok") throw new Error("Teacher health schema is unsupported."); ready = true;
      } catch (error) {
        if (options.signal.aborted || !child.alive()) throw error;
        const code = (error as NodeJS.ErrnoException)?.code;
        if (!(error instanceof LocalTeacherHttpStatus && error.statusCode === 503) && !["ENOENT", "ECONNREFUSED"].includes(code ?? "")) throw error;
      }
      if (!ready) await wait(200, options.signal);
    }
    options.onPhase?.("local-teacher-ready");
    return {
      sessionId: options.sessionId, argvSha256: localTeacherHash(localTeacherCanonical(args)),
      complete: async (prompt, attemptId, signal) => {
        const promptBytes = Buffer.byteLength(prompt), maximumFrame = (options.generation.maxOutputBytes + promptBytes) * 6 + 65536;
        await options.checkResources(child?.child.pid ? [child.child.pid] : [], promptBytes * 8 + maximumFrame * 4 + 256 * 1024);
        const body = localTeacherCompletionBody(prompt, options.generation);
        const requestSha256 = localTeacherHash(body), start = process.hrtime.bigint();
        const relative = `local-teachers/attempts/${attemptId}`, directory = await localTeacherDirectory(options.root, relative);
        const contentFile = await open(join(directory, "content.bin"), "wx", 0o600), rawFile = await open(join(directory, "response.sse"), "wx", 0o600);
        const contentHash = createHash("sha256"), rawHash = createHash("sha256"); let contentBytes = 0, rawBytes = 0;
        const maximumRaw = options.generation.maxOutputBytes * 64 + promptBytes * 6 + 1024 * 1024;
        try {
          const response = await localTeacherHttp(endpoint, "/completion", "POST", key, signal, socket => beforeSend(socket, signal), body);
          if (!String(response.headers["content-type"] ?? "").startsWith("text/event-stream")) { response.destroy(); throw new Error("Selected raw completion endpoint did not return the verified SSE protocol."); }
          const parser = new LocalTeacherSse(options.generation.maxOutputBytes, maximumFrame, async bytes => {
            await contentFile.writeFile(bytes); contentHash.update(bytes); contentBytes += bytes.length;
          });
          for await (const chunk of response) {
            signal.throwIfAborted(); await requireCurrent(); const bytes = Buffer.from(chunk);
            if (rawBytes + bytes.length > maximumRaw) { response.destroy(); throw new Error("Teacher SSE transport exceeded its admitted envelope."); }
            await options.reserveLogWrite(bytes.length); await rawFile.writeFile(bytes); rawHash.update(bytes); rawBytes += bytes.length; await parser.push(bytes);
          }
          const final = parser.finish();
          const settings = final.generation_settings as Record<string, unknown> | undefined;
          if (settings?.n_predict !== undefined && settings.n_predict !== options.generation.maxOutputTokens || settings?.n_ctx !== undefined && settings.n_ctx !== options.generation.contextTokens)
            throw new Error("Teacher changed the requested prediction/context envelope; no example was learned.");
          await contentFile.sync(); await rawFile.sync();
          return { content: { relativePath: `${relative}/content.bin`, sha256: contentHash.digest("hex"), bytes: contentBytes, coverage: "complete" },
            transport: { relativePath: `${relative}/response.sse`, sha256: rawHash.digest("hex"), bytes: rawBytes, coverage: "complete" },
            final, elapsedNanoseconds: (process.hrtime.bigint() - start).toString(), requestSha256 };
        } catch (error) {
          await contentFile.sync(); await rawFile.sync();
          throw new LocalTeacherCompletionError(error instanceof Error ? error.message : String(error), {
            content: { relativePath: `${relative}/content.bin`, sha256: contentHash.digest("hex"), bytes: contentBytes, coverage: "prefix-only" },
            transport: { relativePath: `${relative}/response.sse`, sha256: rawHash.digest("hex"), bytes: rawBytes, coverage: "prefix-only" },
            elapsedNanoseconds: (process.hrtime.bigint() - start).toString(), requestSha256 });
        } finally { await contentFile.close(); await rawFile.close(); }
      },
      close: async reason => { await child!.stop(reason); const result = await child!.closed; await rm(privateRoot, { recursive: true, force: true }); return result; }
    };
  } catch (error) {
    if (child) await child.stop(error instanceof Error ? error : new Error(String(error)));
    await rm(privateRoot, { recursive: true, force: true }); throw error;
  }
}
