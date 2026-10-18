/** Tiny owned metadata/process fixtures only; no GGUF tensor inference. */
import { describe, expect, it } from "vitest";
import { mkdtemp, open, readFile, rm, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { createServer } from "node:http";
import { LocalTeacherSse } from "../src/main/localTeacherSse";
import { localTeacherHttp } from "../src/main/localTeacherServer";
import { cleanLocalTeacherEnvironment, localTeacherProbeFlags, localTeacherRequiredFlags, startLocalTeacherChild } from "../src/main/localTeacherRunner";
import { hashLocalTeacherFile, readLocalTeacherGguf } from "../src/main/localTeacherFiles";
import { localTeacherHash, localTeacherRequest } from "../src/main/localTeacherFormat";
import { validateLocalTeacherTcpOwner } from "../src/main/localTeacherTcpOwner";
const generation = { contextTokens: 32, maxOutputTokens: 8, threads: 1, temperature: .7, seed: 1, timeoutMs: 0, maxOutputBytes: 4096 };
const event = (content: string, stop = false) => `data: ${JSON.stringify({ content, stop })}\n\n`;
const u64 = (value: number) => { const result = Buffer.alloc(8); result.writeBigUInt64LE(BigInt(value)); return result; };
const u32 = (value: number) => { const result = Buffer.alloc(4); result.writeUInt32LE(value); return result; };
const str = (value: string) => Buffer.concat([u64(Buffer.byteLength(value)), Buffer.from(value)]);
const tinyGguf = () => Buffer.concat([Buffer.from("GGUF"), u32(3), u64(0), u64(3), str("general.architecture"), u32(8), str("llama"),
  str("llama.context_length"), u32(4), u32(128), str("tokenizer.ggml.tokens"), u32(9), u32(8), u64(2), str("first"), str("last")]);

describe("local teacher raw protocol and selected source boundary", () => {
  it("preserves full fragmented Unicode deltas and handles multiple complete frames before its pending-frame bound", async () => {
    const chunks: Buffer[] = [], parser = new LocalTeacherSse(100, 80, async bytes => { chunks.push(bytes); });
    const input = Buffer.from(event("ab🌷") + event("cd") + event("" , true));
    await parser.push(input.subarray(0, 27)); await parser.push(input.subarray(27));
    expect(parser.finish().stop).toBe(true); expect(Buffer.concat(chunks).toString()).toBe("ab🌷cd");
  });
  it("rejects aggregate terminal content, malformed Unicode, truncation and data after the real terminal", async () => {
    for (const raw of [event("one", true), 'data: {"content":"\\ud800","stop":false}\n\n',
      'data: {"content":"x","stop":false,"truncated":true}\n\n', event("x") + event("", true) + event("y")]) {
      const parser = new LocalTeacherSse(100, 1000, async () => undefined); await expect(parser.push(Buffer.from(raw))).rejects.toThrow();
    }
  });
  it("fails output/pending frame/incomplete terminal explicitly rather than learning a clipped prefix", async () => {
    const parser = new LocalTeacherSse(2, 100, async () => undefined); await expect(parser.push(Buffer.from(event("three")))).rejects.toThrow(/byte envelope/);
    const pending = new LocalTeacherSse(100, 10, async () => undefined); await expect(pending.push(Buffer.from("data: unsupported"))).rejects.toThrow(/parse envelope/);
    const incomplete = new LocalTeacherSse(100, 100, async () => undefined); await incomplete.push(Buffer.from(event("a"))); expect(() => incomplete.finish()).toThrow(/complete/);
  });
  it("sends no HTTP authorization or prompt bytes when the exact endpoint ownership guard fails", async () => {
    let requests = 0; const server = createServer((_request, response) => { requests++; response.end("{}"); });
    await new Promise<void>(resolve => server.listen(0, "127.0.0.1", resolve)); const address = server.address();
    if (!address || typeof address === "string") throw new Error("fixture socket unavailable");
    try { await expect(localTeacherHttp({ hostname: "127.0.0.1", port: address.port }, "/completion", "POST", "synthetic-private-auth", new AbortController().signal,
      async () => { throw new Error("fixture endpoint ownership denied"); }, "private fixture prompt")).rejects.toThrow(/ownership denied/); expect(requests).toBe(0); }
    finally { await new Promise<void>(resolve => server.close(() => resolve())); }
  });
  it("parses all header metadata including tokenizer arrays without claiming tensor payload validation", async () => {
    const directory = await mkdtemp(join(tmpdir(), "teacher-header-")), path = join(directory, "tiny.gguf");
    try { const bytes = tinyGguf(); await writeFile(path, bytes); const file = await open(path, "r");
      try { const header = await readLocalTeacherGguf(file, bytes.length, new AbortController().signal); expect(header.metadataEnd).toBe(bytes.length);
        expect(header.architecture).toBe("llama"); expect(header.geometry["llama.context_length"]).toBe(128); expect(header.tensorPayloadValidated).toBe(false); }
      finally { await file.close(); } }
    finally { await rm(directory, { recursive: true, force: true }); }
  });
  it("whole-file selected SHA reports progress and obeys cancellation without loading any tensor", async () => {
    const directory = await mkdtemp(join(tmpdir(), "teacher-sha-")), path = join(directory, "tiny.gguf"), signal = new AbortController();
    try { const bytes = tinyGguf(); await writeFile(path, bytes); const progress: number[] = [];
      const file = await hashLocalTeacherFile(path, signal.signal, async () => undefined, value => progress.push(value)); expect(file.sha256).toBe(localTeacherHash(bytes)); expect(progress.at(-1)).toBe(bytes.length);
      signal.abort(new Error("fixture selection cancelled")); await expect(hashLocalTeacherFile(path, signal.signal, async () => undefined)).rejects.toThrow(/cancelled/); }
    finally { await rm(directory, { recursive: true, force: true }); }
  });
  it("supports only flags proven by actual help and never mutes CLI token logs or fabricates raw support", () => {
    const help = localTeacherRequiredFlags("llama-server").join("\n") + "\nUNIX socket paths ending .sock";
    expect(localTeacherProbeFlags(help, "llama-server").supported).toBe(true);
    expect(localTeacherProbeFlags("--model --chat-template", "llama-cli").supported).toBe(false);
    expect(localTeacherRequiredFlags("llama-completion")).not.toContain("--log-disable");
    expect(localTeacherRequest({ brainId: "b", requestId: "r", teacherId: "t", prompts: ["  full first\n", "last🌷"], generation }).request.prompts).toEqual(["  full first\n", "last🌷"]);
  });
  it("uses a clean child env without inherited provider keys or llama defaults", () => {
    process.env.LOCAL_TEACHER_SYNTHETIC_SECRET = "not-for-child"; process.env.LLAMA_ARG_MODEL = "foreign-model";
    try { const env = cleanLocalTeacherEnvironment("/fixture/private-home"); expect(env.LOCAL_TEACHER_SYNTHETIC_SECRET).toBeUndefined(); expect(env.LLAMA_ARG_MODEL).toBeUndefined(); expect(env.HOME).toBe("/fixture/private-home"); }
    finally { delete process.env.LOCAL_TEACHER_SYNTHETIC_SECRET; delete process.env.LLAMA_ARG_MODEL; }
  });
  it("only seals a real tiny owned child after its actual exit and both captured pipes close", async () => {
    const directory = await mkdtemp(join(tmpdir(), "teacher-child-"));
    try { const child = await startLocalTeacherChild({ root: directory, relativeDirectory: "local-teachers/attempts/owned-child", executable: process.execPath,
      arguments: ["-e", "process.stdout.write('exact synthetic response'); process.stderr.write('synthetic log');"], signal: new AbortController().signal,
      maxLogBytes: 4096, timeoutMs: 3000, requireCurrent: async () => undefined, checkResources: async () => undefined });
      const result = await child.closed; expect(child.alive()).toBe(false); expect(result.exitCode).toBe(0); expect(result.stdout.coverage).toBe("complete");
      expect((await readFile(join(directory, result.stdout.relativePath))).toString()).toBe("exact synthetic response"); }
    finally { await rm(directory, { recursive: true, force: true }); }
  });
  it("bounds retained child-output coverage and stops only that owned process", async () => {
    const directory = await mkdtemp(join(tmpdir(), "teacher-output-limit-"));
    try { const child = await startLocalTeacherChild({ root: directory, relativeDirectory: "local-teachers/attempts/owned-output", executable: process.execPath,
      arguments: ["-e", "process.stdout.write('abcdef');setInterval(()=>{},1000);"], signal: new AbortController().signal,
      maxLogBytes: 3, timeoutMs: 3000, requireCurrent: async () => undefined, checkResources: async () => undefined });
      const result = await child.closed; expect(result.stdout.bytes).toBe(3); expect(result.stdout.coverage).toBe("prefix-only"); expect(result.error).toMatch(/prefix|envelope/); expect(child.alive()).toBe(false); }
    finally { await rm(directory, { recursive: true, force: true }); }
  });
  it("requires both listener and connected peer to be owned by the exact selected Windows child", () => {
    const rows = [{ OwningProcess: 10, LocalAddress: "127.0.0.1", LocalPort: 1234, State: "Listen" },
      { OwningProcess: 10, LocalAddress: "127.0.0.1", LocalPort: 1234, RemoteAddress: "127.0.0.1", RemotePort: 5678, State: "Established" }];
    expect(() => validateLocalTeacherTcpOwner(rows, 10, 1234, 5678)).not.toThrow();
    expect(() => validateLocalTeacherTcpOwner(rows, 11, 1234, 5678)).toThrow(/authenticated/);
    expect(() => validateLocalTeacherTcpOwner(rows.slice(0, 1), 10, 1234, 5678)).toThrow(/authenticated/);
  });
});
