import { createHash } from "node:crypto";
import { mkdtemp, readFile, rm, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { afterEach, describe, expect, it } from "vitest";
import { inspectPluginWasmModule } from "../src/main/pluginWasmModule";
import { runPluginWasmParser } from "../src/main/pluginWasmParser";
import { parserModule, parserWasm } from "./fixtures/pluginWasmParser";

const roots: string[] = [];
afterEach(async () => { for (const root of roots.splice(0)) await rm(root, { recursive: true, force: true }); });
async function directory(): Promise<string> { const root = await mkdtemp(join(tmpdir(), "omni-wasm-parser-")); roots.push(root); return root; }

describe("isolated executable plugin record parser", () => {
  it("rejects imported or maximum-free linear memory before a module can execute", () => {
    expect(() => inspectPluginWasmModule(parserModule(parserWasm({ maximum: undefined })))).toThrow(/explicit maximum/i);
    expect(() => inspectPluginWasmModule(parserModule(parserWasm({ maximum: 2, importedMemory: true })))).toThrow(/may not import/i);
  });

  it("streams every source byte with one-batch backpressure and commits exact source/module/output provenance", async () => {
    const root = await directory(), source = join(root, "source.lines"), body = Buffer.from('{"x":1}\n{"x":2}\n');
    await writeFile(source, body); const module = parserModule();
    const request = { pluginId: "fixture.parser", packageSha256: "a".repeat(64),
      formatId: "lines", module, sourcePath: source, sourceSha256: createHash("sha256").update(body).digest("hex"),
      sourceBytes: body.length, transactionKey: "b".repeat(64), workspaceRoot: root,
      chunkBytes: 8, physicalEnvelopeBytes: 8 * 1024 * 1024 } as const;
    const [prepared, reused] = await Promise.all([runPluginWasmParser(request), runPluginWasmParser(request)]);
    expect(await readFile(prepared.path)).toEqual(body);
    expect(prepared.provenance).toMatchObject({ sourceBytes: body.length, outputBytes: body.length,
      outputRecords: 2, moduleSha256: module.wasmSha256, complete: true });
    expect(reused.provenance).toEqual(prepared.provenance);
    await prepared.cleanup(); expect(await readFile(reused.path)).toEqual(body); await reused.cleanup();
  });

  it("terminates a nonreturning module on cancellation and leaves explicit incomplete coverage", async () => {
    const root = await directory(), source = join(root, "source.lines"), body = Buffer.from('{"x":1}\n');
    await writeFile(source, body); const controller = new AbortController();
    let entered!: () => void; const dispatched = new Promise<void>(resolve => { entered = resolve; });
    const running = runPluginWasmParser({ pluginId: "fixture.parser", packageSha256: "a".repeat(64),
      formatId: "lines", module: parserModule(parserWasm({ maximum: 2, infinite: true })),
      sourcePath: source, sourceSha256: createHash("sha256").update(body).digest("hex"), sourceBytes: body.length,
      transactionKey: "c".repeat(64), workspaceRoot: root, signal: controller.signal,
      chunkBytes: body.length, physicalEnvelopeBytes: 8 * 1024 * 1024,
      onProgress: (_value, _message, detail) => { if ((detail as { inputBytes?: number }).inputBytes === body.length) entered(); } });
    await dispatched; controller.abort(new Error("fixture cancellation"));
    await expect(running).rejects.toThrow(/fixture cancellation/i);
    const partial = JSON.parse(await readFile(join(root, "plugin-parser", `${"c".repeat(64)}.partial.json`), "utf8"));
    expect(partial).toMatchObject({ complete: false, inputBytes: body.length, outputBytes: 0, outputRecords: 0 });
  });

  it("pauses on an invalid partial-record output without claiming traversal complete", async () => {
    const root = await directory(), source = join(root, "source.lines"), body = Buffer.from('{"x":1}\n');
    await writeFile(source, body);
    await expect(runPluginWasmParser({ pluginId: "fixture.parser", packageSha256: "a".repeat(64),
      formatId: "lines", module: parserModule(), sourcePath: source,
      sourceSha256: createHash("sha256").update(body).digest("hex"), sourceBytes: body.length,
      transactionKey: "e".repeat(64), workspaceRoot: root, chunkBytes: 4,
      physicalEnvelopeBytes: 8 * 1024 * 1024 })).rejects.toThrow(/complete JSONL record boundary/i);
    const partial = JSON.parse(await readFile(join(root, "plugin-parser", `${"e".repeat(64)}.partial.json`), "utf8"));
    expect(partial).toMatchObject({ complete: false, inputBytes: 4, outputBytes: 0, outputRecords: 0 });
  });

  it("rejects declared maximum memory outside the current physical envelope", async () => {
    const root = await directory(), source = join(root, "source.lines"), body = Buffer.from('{"x":1}\n');
    await writeFile(source, body);
    await expect(runPluginWasmParser({ pluginId: "fixture.parser", packageSha256: "a".repeat(64),
      formatId: "lines", module: parserModule(parserWasm({ maximum: 256 })), sourcePath: source,
      sourceSha256: createHash("sha256").update(body).digest("hex"), sourceBytes: body.length,
      transactionKey: "d".repeat(64), workspaceRoot: root, physicalEnvelopeBytes: 1024 * 1024 })).rejects.toThrow(/physical memory/i);
  });
});
