import { createHash, randomUUID } from "node:crypto";
import { createReadStream, createWriteStream } from "node:fs";
import { lstat, mkdir, readFile, rename, rm } from "node:fs/promises";
import { freemem } from "node:os";
import { basename, join } from "node:path";
import { getHeapStatistics } from "node:v8";
import { Worker } from "node:worker_threads";
import { once } from "node:events";
import { requireDiskWrite } from "./diskSpace";
import { writeMutableFileIsolated } from "./mutableFileIsolation";
import { inspectPluginWasmModule } from "./pluginWasmModule";
import { normalizeSourceMediaType } from "../shared/sourceMediaType";
import {
  PLUGIN_WASM_PARSER_ABI,
  type PluginParserExecutionProvenance,
  type PluginParserExecutionRequest,
  type PluginPreparedTrainingSourceReady,
  type PluginTrainingModuleDefinition
} from "../shared/plugins";

const SHA256 = /^[a-f0-9]{64}$/;
const DEFAULT_CHUNK_BYTES = 256 * 1024;
const FORBIDDEN_KEYS = new Set(["__proto__", "prototype", "constructor"]);
const parserRuns = new Map<string, Promise<void>>();
const parserConsumers = new Map<string, number>();

const WORKER_SOURCE = String.raw`
const { parentPort, workerData } = require('node:worker_threads');
let instance;
const fail = error => parentPort.postMessage({ type: 'error', error: String(error && error.message || error).slice(0, 2000) });
(async()=>{
  const bytes=Buffer.from(workerData.moduleBase64,'base64'), module=await WebAssembly.compile(bytes);
  if(WebAssembly.Module.imports(module).length)throw new Error('WASM parser imports are forbidden.');
  instance=await WebAssembly.instantiate(module,{});
  const x=instance.exports;
  if(!(x.memory instanceof WebAssembly.Memory)||typeof x.omni_alloc!=='function'||typeof x.omni_parse!=='function')throw new Error('WASM parser ABI exports are missing.');
  if(x.memory.buffer.byteLength>workerData.maximumBytes)throw new Error('WASM parser exceeded its admitted linear memory.');
  parentPort.postMessage({type:'ready',memoryBytes:x.memory.buffer.byteLength});
})().catch(fail);
parentPort.on('message',message=>{
  if(!instance||!message||message.type!=='chunk')return;
  try{
    const x=instance.exports,input=new Uint8Array(message.input),length=input.byteLength;
    const pointer=Number(x.omni_alloc(length))>>>0, before=x.memory.buffer.byteLength;
    if(before>workerData.maximumBytes||pointer>before||length>before-pointer)throw new Error('WASM parser input pointer exceeds admitted memory.');
    new Uint8Array(x.memory.buffer,pointer,length).set(input);
    const packed=x.omni_parse(pointer,length,message.final?1:0);
    if(typeof packed!=='bigint')throw new Error('WASM parser returned a non-i64 output descriptor.');
    const raw=BigInt.asUintN(64,packed),outPointer=Number((raw>>32n)&0xffffffffn),outLength=Number(raw&0xffffffffn),after=x.memory.buffer.byteLength;
    if(after>workerData.maximumBytes||outPointer>after||outLength>after-outPointer)throw new Error('WASM parser output pointer exceeds admitted memory.');
    const output=new Uint8Array(outLength);output.set(new Uint8Array(x.memory.buffer,outPointer,outLength));
    if(typeof x.omni_free==='function'){
      const released=new Set();for(const [p,n] of [[pointer,length],[outPointer,outLength]]){const key=p+':'+n;if(n&& !released.has(key)){released.add(key);x.omni_free(p,n);}}
    }
    parentPort.postMessage({type:'result',sequence:message.sequence,output:output.buffer,memoryBytes:after},[output.buffer]);
  }catch(error){fail(error);}
});`;

interface ParserSpoolReceipt {
  format: "omni-plugin-parser-spool";
  version: 1;
  transactionKey: string;
  outputFile: string;
  provenance: PluginParserExecutionProvenance;
  receiptSha256: string;
}

interface ParserCoverage {
  format: "omni-plugin-parser-coverage";
  version: 1;
  inputBytes: number;
  outputBytes: number;
  outputRecords: number;
  complete: boolean;
  error?: string;
}

export interface RunPluginWasmParserInput extends PluginParserExecutionRequest {
  pluginId: string;
  packageSha256: string;
  formatId: string;
  module: PluginTrainingModuleDefinition;
  /** Test/host transport tuning only; this never limits total source bytes. */
  chunkBytes?: number;
  /** Optional already-admitted physical envelope for deterministic fixtures. */
  physicalEnvelopeBytes?: number;
}

function canonical(value: unknown): string {
  if (value === null || typeof value !== "object") return JSON.stringify(value);
  if (Array.isArray(value)) return `[${value.map(canonical).join(",")}]`;
  return `{${Object.entries(value as Record<string, unknown>)
    .sort(([left], [right]) => left < right ? -1 : left > right ? 1 : 0)
    .map(([key, child]) => `${JSON.stringify(key)}:${canonical(child)}`).join(",")}}`;
}

function sha256(value: string | Buffer): string {
  return createHash("sha256").update(value).digest("hex");
}

function receiptHash(value: Omit<ParserSpoolReceipt, "receiptSha256">): string {
  return sha256(canonical(value));
}

function exactKeys(value: object, keys: readonly string[]): boolean {
  return Object.keys(value).sort().join("\0") === [...keys].sort().join("\0");
}

function validProvenance(value: PluginParserExecutionProvenance): boolean {
  return Boolean(value && exactKeys(value, ["abi", "complete", "format", "formatId", "moduleId", "moduleSha256",
    "outputBytes", "outputRecords", "outputSha256", "packageSha256", "pluginId", "sourceBytes", "sourceSha256", "version", ...(value.inputMediaType === undefined ? [] : ["inputMediaType"])]) &&
    value.format === "omni-plugin-parser-execution" && value.version === 1 && value.abi === PLUGIN_WASM_PARSER_ABI &&
    value.complete === true && (value.inputMediaType === undefined || normalizeSourceMediaType(value.inputMediaType) === value.inputMediaType) && SHA256.test(value.packageSha256) && SHA256.test(value.moduleSha256) &&
    SHA256.test(value.sourceSha256) && SHA256.test(value.outputSha256) && Number.isSafeInteger(value.sourceBytes) &&
    value.sourceBytes >= 0 && Number.isSafeInteger(value.outputBytes) && value.outputBytes >= 0 &&
    Number.isSafeInteger(value.outputRecords) && value.outputRecords >= 0);
}

function physicalEnvelope(): number {
  const heap = getHeapStatistics(), heapFree = Math.max(0, heap.heap_size_limit - process.memoryUsage().heapUsed);
  return Math.floor(Math.min(freemem() / 8, heapFree / 2));
}

function safeJsonValue(value: unknown, depth = 0): void {
  if (depth > 64) throw new Error("Executable parser output record is nested too deeply.");
  if (value === null || typeof value === "string" || typeof value === "boolean") return;
  if (typeof value === "number") {
    if (!Number.isFinite(value)) throw new Error("Executable parser output contains a non-finite number.");
    return;
  }
  if (Array.isArray(value)) { value.forEach(child => safeJsonValue(child, depth + 1)); return; }
  if (!value || typeof value !== "object") throw new Error("Executable parser output is not JSON data.");
  for (const [key, child] of Object.entries(value as Record<string, unknown>)) {
    if (FORBIDDEN_KEYS.has(key) || key.includes("\0")) throw new Error("Executable parser output contains an unsafe key.");
    safeJsonValue(child, depth + 1);
  }
}

function validateJsonlBatch(output: Buffer): number {
  if (!output.length) return 0;
  if (output[output.length - 1] !== 0x0a) throw new Error("Executable parser output batches must end at a complete JSONL record boundary.");
  const text = new TextDecoder("utf-8", { fatal: true }).decode(output), lines = text.split("\n");
  lines.pop();
  for (const line of lines) {
    if (!line.trim()) throw new Error("Executable parser emitted an empty JSONL record.");
    const value = JSON.parse(line) as unknown;
    if (!value || typeof value !== "object" || Array.isArray(value)) throw new Error("Executable parser JSONL records must be objects.");
    safeJsonValue(value);
  }
  return lines.length;
}

async function fileSha256(path: string, signal?: AbortSignal): Promise<string> {
  const hash = createHash("sha256");
  for await (const chunk of createReadStream(path, signal ? { signal } : undefined)) { signal?.throwIfAborted(); hash.update(chunk as Buffer); }
  return hash.digest("hex");
}

async function existingPrepared(input: RunPluginWasmParserInput, outputPath: string, receiptPath: string): Promise<PluginPreparedTrainingSourceReady | undefined> {
  try {
    const raw = JSON.parse(await readFile(receiptPath, "utf8")) as ParserSpoolReceipt;
    const { receiptSha256, ...body } = raw;
    if (!exactKeys(raw, ["format", "version", "transactionKey", "outputFile", "provenance", "receiptSha256"]) ||
        !validProvenance(raw.provenance) || raw.format !== "omni-plugin-parser-spool" || raw.version !== 1 || raw.transactionKey !== input.transactionKey ||
        raw.outputFile !== basename(outputPath) || !SHA256.test(receiptSha256) || receiptHash(body) !== receiptSha256 ||
        raw.provenance.pluginId !== input.pluginId || raw.provenance.packageSha256 !== input.packageSha256 ||
        raw.provenance.formatId !== input.formatId || raw.provenance.moduleId !== input.module.id ||
        raw.provenance.moduleSha256 !== input.module.wasmSha256 || raw.provenance.sourceSha256 !== input.sourceSha256 ||
        raw.provenance.inputMediaType !== input.inputMediaType ||
        raw.provenance.sourceBytes !== input.sourceBytes || raw.provenance.complete !== true) return undefined;
    const info = await lstat(outputPath);
    if (!info.isFile() || info.isSymbolicLink() || info.size !== raw.provenance.outputBytes ||
        await fileSha256(outputPath, input.signal) !== raw.provenance.outputSha256) return undefined;
    const sourceInfo = await lstat(input.sourcePath);
    if (!sourceInfo.isFile() || sourceInfo.isSymbolicLink() || sourceInfo.size !== input.sourceBytes ||
        await fileSha256(input.sourcePath, input.signal) !== input.sourceSha256) return undefined;
    return { path: outputPath, parserFormat: "jsonl", provenance: raw.provenance,
      cleanup: async () => { await Promise.all([rm(outputPath, { force: true }), rm(receiptPath, { force: true })]); } };
  } catch (error) {
    if (input.signal?.aborted) throw input.signal.reason ?? error;
    return undefined;
  }
}

async function post(worker: Worker, message: Record<string, unknown>, transfer: ArrayBuffer[] = [], signal?: AbortSignal): Promise<Record<string, unknown>> {
  signal?.throwIfAborted();
  return new Promise((resolve, reject) => {
    const finish = (): void => { worker.off("message", onMessage); worker.off("error", onError); worker.off("exit", onExit); signal?.removeEventListener("abort", onAbort); };
    const onMessage = (value: unknown): void => {
      const row = value && typeof value === "object" ? value as Record<string, unknown> : {};
      if (row.type === "error") { finish(); reject(new Error(String(row.error ?? "Executable parser worker failed."))); }
      else { finish(); resolve(row); }
    };
    const onError = (error: Error): void => { finish(); reject(error); };
    const onExit = (code: number): void => { finish(); reject(new Error(`Executable parser worker exited before acknowledgement (${code}).`)); };
    const onAbort = (): void => { finish(); void worker.terminate(); reject(signal?.reason ?? new Error("Executable parser cancelled.")); };
    worker.on("message", onMessage); worker.once("error", onError); worker.once("exit", onExit);
    signal?.addEventListener("abort", onAbort, { once: true });
    worker.postMessage(message, transfer);
  });
}

async function withParserRun<T>(key: string, operation: () => Promise<T>): Promise<T> {
  const prior = parserRuns.get(key) ?? Promise.resolve();
  let release!: () => void; const gate = new Promise<void>(resolve => { release = resolve; });
  const tail = prior.then(() => gate); parserRuns.set(key, tail); await prior;
  try { return await operation(); }
  finally { release(); if (parserRuns.get(key) === tail) parserRuns.delete(key); }
}

export async function runPluginWasmParser(input: RunPluginWasmParserInput): Promise<PluginPreparedTrainingSourceReady> {
  const key = `${input.workspaceRoot}\0${input.transactionKey}`;
  parserConsumers.set(key, (parserConsumers.get(key) ?? 0) + 1);
  let prepared: PluginPreparedTrainingSourceReady;
  try { prepared = await withParserRun(key, () => runPluginWasmParserLocked(input)); }
  catch (error) {
    const remaining = (parserConsumers.get(key) ?? 1) - 1;
    if (remaining > 0) parserConsumers.set(key, remaining); else parserConsumers.delete(key);
    throw error;
  }
  let released = false;
  return { ...prepared, cleanup: async () => {
    if (released) return; released = true;
    const remaining = (parserConsumers.get(key) ?? 1) - 1;
    if (remaining > 0) { parserConsumers.set(key, remaining); return; }
    parserConsumers.delete(key); await prepared.cleanup();
  } };
}

async function runPluginWasmParserLocked(input: RunPluginWasmParserInput): Promise<PluginPreparedTrainingSourceReady> {
  if (!SHA256.test(input.packageSha256) || !SHA256.test(input.sourceSha256) || !SHA256.test(input.transactionKey) ||
      !Number.isSafeInteger(input.sourceBytes) || input.sourceBytes < 0 ||
      (input.inputMediaType !== undefined && normalizeSourceMediaType(input.inputMediaType) !== input.inputMediaType)) {
    throw new Error("Executable parser request identity is invalid.");
  }
  const inspected = inspectPluginWasmModule(input.module);
  const chunkBytes = input.chunkBytes ?? DEFAULT_CHUNK_BYTES;
  if (!Number.isSafeInteger(chunkBytes) || chunkBytes < 1) throw new Error("Executable parser transport batch is invalid.");
  const envelope = input.physicalEnvelopeBytes ?? physicalEnvelope();
  const required = inspected.maximumBytes + inspected.bytes.length * 4 + chunkBytes * 4;
  if (!Number.isSafeInteger(envelope) || envelope < required) {
    throw new Error(`Executable parser awaits physical memory: ${required} bytes required within the current ${Math.max(0, envelope)}-byte envelope.`);
  }
  const directory = join(input.workspaceRoot, "plugin-parser"), outputPath = join(directory, `${input.transactionKey}.jsonl`),
    receiptPath = join(directory, `${input.transactionKey}.receipt.json`), partialPath = join(directory, `${input.transactionKey}.partial.json`);
  await mkdir(directory, { recursive: true, mode: 0o700 });
  const existing = await existingPrepared(input, outputPath, receiptPath);
  if (existing) return existing;
  await Promise.all([rm(outputPath, { force: true }), rm(receiptPath, { force: true }), rm(partialPath, { force: true })]);
  const temporary = join(directory, `.${input.transactionKey}.${randomUUID()}.incomplete`);
  const output = createWriteStream(temporary, { flags: "wx", mode: 0o600 });
  let outputFailure: Error | undefined;
  output.on("error", error => { outputFailure = error; });
  const outputClosed = once(output, "close").then(() => undefined, () => undefined);
  const worker = new Worker(WORKER_SOURCE, { eval: true,
    workerData: { moduleBase64: input.module.wasmBase64, maximumBytes: inspected.maximumBytes },
    resourceLimits: { maxOldGenerationSizeMb: Math.max(16, Math.floor(envelope / (1024 * 1024))), stackSizeMb: 4 } });
  let inputBytes = 0, outputBytes = 0, outputRecords = 0, sequence = 0, closed = false;
  const sourceDigest = createHash("sha256"), outputDigest = createHash("sha256");
  const coverage = (complete: boolean, error?: string): ParserCoverage => ({ format: "omni-plugin-parser-coverage", version: 1,
    inputBytes, outputBytes, outputRecords, complete, ...(error ? { error: error.slice(0, 2_000) } : {}) });
  const publish = async (complete: boolean, error?: string): Promise<void> => {
    await input.onProgress?.(complete ? 0.25 : Math.min(0.249, input.sourceBytes ? inputBytes / input.sourceBytes * 0.249 : 0),
      complete ? `Executable parser visited all ${inputBytes.toLocaleString()} source bytes.` :
        error ? `Executable parser paused after ${inputBytes.toLocaleString()} source bytes: ${error}` :
          `Executable parser streamed ${inputBytes.toLocaleString()} of ${input.sourceBytes.toLocaleString()} source bytes.`,
      coverage(complete, error));
  };
  const writeOutput = async (bytes: Buffer): Promise<void> => {
    const records = validateJsonlBatch(bytes);
    if (!bytes.length) return;
    await requireDiskWrite(input.workspaceRoot, { operationWriteBytes: bytes.length });
    if (!output.write(bytes)) await once(output, "drain");
    outputDigest.update(bytes); outputBytes += bytes.length; outputRecords += records;
  };
  try {
    const ready = await post(worker, { type: "init" }, [], input.signal);
    if (ready.type !== "ready") throw new Error("Executable parser worker did not initialize its ABI.");
    const source = createReadStream(input.sourcePath, { highWaterMark: chunkBytes, ...(input.signal ? { signal: input.signal } : {}) });
    for await (const value of source) {
      input.signal?.throwIfAborted();
      const bytes = Buffer.from(value as Buffer), transferable = bytes.buffer.slice(bytes.byteOffset, bytes.byteOffset + bytes.byteLength);
      sourceDigest.update(bytes); inputBytes += bytes.length;
      await publish(false);
      const response = await post(worker, { type: "chunk", sequence: sequence++, input: transferable, final: false }, [transferable], input.signal);
      if (response.type !== "result") throw new Error("Executable parser worker returned no batch receipt.");
      await writeOutput(Buffer.from(response.output as ArrayBuffer)); await publish(false);
    }
    if (inputBytes !== input.sourceBytes || sourceDigest.digest("hex") !== input.sourceSha256) {
      throw new Error("Executable parser source changed or was not traversed completely.");
    }
    const empty = new ArrayBuffer(0), final = await post(worker, { type: "chunk", sequence: sequence++, input: empty, final: true }, [empty], input.signal);
    if (final.type !== "result") throw new Error("Executable parser worker returned no final receipt.");
    await writeOutput(Buffer.from(final.output as ArrayBuffer));
    output.end(); await outputClosed; closed = true;
    if (outputFailure) throw outputFailure;
    const outputSha256 = outputDigest.digest("hex");
    await rename(temporary, outputPath);
    const provenance: PluginParserExecutionProvenance = { format: "omni-plugin-parser-execution", version: 1,
      pluginId: input.pluginId, packageSha256: input.packageSha256, formatId: input.formatId,
      moduleId: input.module.id, moduleSha256: input.module.wasmSha256, abi: PLUGIN_WASM_PARSER_ABI,
      sourceSha256: input.sourceSha256, sourceBytes: input.sourceBytes,
      ...(input.inputMediaType ? { inputMediaType: input.inputMediaType } : {}),
      outputSha256, outputBytes, outputRecords, complete: true };
    const body: Omit<ParserSpoolReceipt, "receiptSha256"> = { format: "omni-plugin-parser-spool", version: 1,
      transactionKey: input.transactionKey, outputFile: basename(outputPath), provenance };
    await writeMutableFileIsolated(receiptPath, Buffer.from(`${canonical({ ...body, receiptSha256: receiptHash(body) })}\n`, "utf8"));
    await publish(true); await rm(partialPath, { force: true });
    return { path: outputPath, parserFormat: "jsonl", provenance,
      cleanup: async () => { await Promise.all([rm(outputPath, { force: true }), rm(receiptPath, { force: true }), rm(partialPath, { force: true })]); } };
  } catch (error) {
    const message = error instanceof Error ? error.message : String(error);
    if (!closed) { output.destroy(); await outputClosed; }
    await rm(temporary, { force: true });
    await writeMutableFileIsolated(partialPath, Buffer.from(`${canonical(coverage(false, message))}\n`, "utf8")).catch(() => undefined);
    await publish(false, message).catch(() => undefined);
    throw error;
  } finally {
    await worker.terminate().catch(() => undefined);
  }
}
