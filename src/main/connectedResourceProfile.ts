import { createHash } from "node:crypto";
import { constants } from "node:fs";
import { lstat, open } from "node:fs/promises";
import { freemem } from "node:os";
import { dirname, join } from "node:path";
import { getHeapStatistics } from "node:v8";
import type { SavedTensorShape } from "./nativeCheckpointMemory";

type RecordValue = Record<string, unknown>;
const object = (value: unknown): RecordValue => {
  if (!value || typeof value !== "object" || Array.isArray(value)) throw new Error("Connected resource metadata is invalid.");
  return value as RecordValue;
};
const positive = (value: unknown): number => {
  if (typeof value !== "number" || !Number.isSafeInteger(value) || value < 1) throw new Error("Connected resource geometry must be positive exact integers.");
  return value;
};
const hash = (value: Uint8Array | string) => createHash("sha256").update(value).digest("hex");
const digest = (value: unknown) => typeof value === "string" && /^[a-f0-9]{64}$/.test(value);

/** Preserve Python's numeric lexemes (1.0 must not become 1) when checking a
 * native hash-bound descriptor. JSON parsing still validates the complete file. */
function descriptorCanonical(text: string): string | undefined {
  let cursor = 0, result: string | undefined;
  const ws = () => { while (/\s/.test(text[cursor] ?? "") && cursor < text.length) cursor += 1; };
  const string = () => { const start = cursor++; while (cursor < text.length) {
    const ch = text[cursor++]; if (ch === "\\") cursor += 1;
    else if (ch === '"') return JSON.parse(text.slice(start, cursor)) as string;
  } throw new Error("Invalid native JSON string."); };
  const parse = (path: string[]): string => {
    ws(); const ch = text[cursor];
    if (ch === "{") {
      cursor += 1; ws(); const entries = new Map<string, string>();
      while (text[cursor] !== "}") {
        const key = string(); if (entries.has(key)) throw new Error("Native metadata has duplicate keys.");
        ws(); cursor += 1; entries.set(key, parse([...path, key])); ws();
        if (text[cursor] !== ",") break; cursor += 1; ws();
      }
      cursor += 1;
      const encode = (omitHash: boolean) => `{${[...entries].filter(([key]) => !omitHash || key !== "sha256")
        .sort(([a], [b]) => a < b ? -1 : a > b ? 1 : 0).map(([key, value]) => `${JSON.stringify(key)}:${value}`).join(",")}}`;
      if (path.length === 1 && path[0] === "connected_regions") result = encode(true);
      return encode(false);
    }
    if (ch === "[") { cursor += 1; ws(); const values: string[] = [];
      while (text[cursor] !== "]") { values.push(parse([...path, String(values.length)])); ws();
        if (text[cursor] !== ",") break; cursor += 1; ws(); }
      cursor += 1; return `[${values.join(",")}]`; }
    if (ch === '"') return JSON.stringify(string());
    const start = cursor; while (cursor < text.length && !/[\s,}\]]/.test(text[cursor]!)) cursor += 1;
    return text.slice(start, cursor);
  };
  parse([]); return result;
}

async function metadata(path: string): Promise<{ value: RecordValue; digest: string }> {
  const info = await lstat(path);
  if (!info.isFile() || info.isSymbolicLink() || info.size * 12 > freemem() / 2
    || info.size * 12 > (getHeapStatistics().heap_size_limit - process.memoryUsage().heapUsed) / 2) throw new Error("Connected metadata awaits safe regular-file/RAM admission.");
  const file = await open(path, constants.O_RDONLY | (process.platform === "win32" ? 0 : constants.O_NOFOLLOW));
  try {
    const first = await file.stat();
    if (first.ino !== info.ino || first.dev !== info.dev) throw new Error("Connected metadata changed while opening.");
    const bytes = await file.readFile(), text = bytes.toString("utf8"), value = object(JSON.parse(text));
    const last = await file.stat();
    if (last.size !== first.size || last.mtimeMs !== first.mtimeMs) throw new Error("Connected metadata changed while planning.");
    if (value.connected_regions !== undefined) {
      const descriptor = object(value.connected_regions), canonical = descriptorCanonical(text);
      if (!canonical || hash(canonical) !== descriptor.sha256) throw new Error("Connected descriptor checksum is invalid.");
    }
    return { value, digest: hash(bytes) };
  } finally { await file.close(); }
}

async function inside(root: string, relative: string): Promise<string> {
  const parts = relative.split("/");
  if (parts.some((part) => !part || part === "." || part === ".." || /[\\:\x00-\x1f]/.test(part))) throw new Error("Connected input path is unsafe.");
  let path = root;
  for (const part of ["", ...parts]) {
    if (part) path = join(path, part);
    const info = await lstat(path);
    if (info.isSymbolicLink()) throw new Error("Connected planning cannot follow symbolic links.");
  }
  return path;
}

export interface NativeContextGeometry {
  leaves: Array<{ prefix: string; width: number; layers: number; workspaceSlots: number }>;
  attentionLayers: number;
  kvBytesPerToken: number;
  workspaceStateBytes: number;
  /** Recurrent, regional hidden sums and connected exchange activity, not weights. */
  connectedActivityBytes: number;
}

/** Use current root headers once; immutable parent metadata supplies provenance
 * and dimensions, while current connected_layout supplies grown depths. */
export async function connectedResourceProfile(engine: string, tensors: ReadonlyMap<string, SavedTensorShape>): Promise<NativeContextGeometry> {
  const root = await metadata(join(engine, "brain.json"));
  const leaves: NativeContextGeometry["leaves"] = [];
  const seenHeaders = new Set<string>();
  let connectedActivityBytes = 0;
  const matrix = (name: string, rows: number, width: number) => {
    const sign = tensors.get(name + "._packed_forward_weight"), state = tensors.get(name + "._connection_state");
    if (!sign || sign.dtype !== "U8" || sign.shape.length !== 2 || sign.shape[0] !== rows || sign.shape[1] !== Math.ceil(width / 4)
      || (state && (state.dtype !== "U8" || state.shape[0] !== rows || state.shape[1] !== width))) throw new Error(`Connected native geometry does not match ${name}.`);
    seenHeaders.add(name + "._packed_forward_weight");
  };
  const walk = async (source: string, saved: RecordValue, layoutValue: unknown, prefix: string): Promise<{ layers: number; experts: number; width: number }> => {
    const descriptor = object(saved.connected_regions), layout = object(layoutValue);
    if (descriptor.format !== "omni-connected-native-regions" || descriptor.formatVersion !== 1 || !Array.isArray(descriptor.regions) || !descriptor.regions.length
      || layout.format !== "omni-connected-layout" || layout.formatVersion !== 1 || !Array.isArray(layout.regions) || layout.regions.length !== descriptor.regions.length
      || !Array.isArray(descriptor.parents) || descriptor.parents.length < 2) throw new Error("Connected layout/provenance is incomplete.");
    const ids = new Set<string>(); let layers = 0, experts = 0, width = 0;
    for (const [index, entry] of descriptor.regions.entries()) {
      const region = object(entry), row = object(layout.regions[index]), config = object(region.config);
      if (typeof region.regionId !== "string" || ids.has(region.regionId) || row.regionId !== region.regionId || typeof region.enginePath !== "string"
        || !["metadataSha256", "coreSha256", "plasticitySha256", "generationSha256", "neuralSha256"].every((field) => digest(region[field]))) throw new Error("Connected region provenance is invalid.");
      ids.add(region.regionId);
      const childWidth = positive(config.d_model), childLayers = positive(row.layers), baseline = positive(config.n_layers);
      if (childLayers < baseline || !Number.isSafeInteger(row.experts) || Number(row.experts) < 0) throw new Error("Connected saved growth is invalid.");
      let path: string | undefined;
      for (const base of [join(source, "connected-inputs"), dirname(source)]) {
        try { path = await inside(base, region.enginePath); break; }
        catch (error) { if ((error as NodeJS.ErrnoException).code !== "ENOENT") throw error; }
      }
      if (!path) throw new Error("Connected native input metadata is missing.");
      const original = await metadata(await inside(path, "brain.json"));
      if (original.digest !== region.metadataSha256) throw new Error("Connected native input metadata checksum changed.");
      for (const file of ["core.safetensors", "plasticity.safetensors"]) {
        if (!(await lstat(await inside(path, file))).isFile()) throw new Error("Connected input checkpoint is unavailable.");
      }
      const child = `${prefix}regions.${index}.decoder.`;
      if (original.value.connected_regions !== undefined) {
        const nested = await walk(path, original.value, row.nested, child);
        if (nested.layers !== childLayers || nested.experts !== row.experts || nested.width !== childWidth) throw new Error("Nested connected geometry disagrees with current layout.");
      } else {
        if (row.nested !== undefined) throw new Error("Ordinary region has an undeclared nested layout.");
        const originalConfig = object(original.value.config);
        if (positive(originalConfig.d_model) !== childWidth) throw new Error("Connected regional width changed outside its native descriptor.");
        if (Number(row.experts) < Number(original.value.expert_count ?? 0)) throw new Error("Connected layout removed a preserved parent expert.");
        for (let layer = 0; layer < childLayers; layer += 1) matrix(`${child}blocks.${layer}.attention.qkv`, 3 * childWidth, childWidth);
        const latent = tensors.get(`${child}global_workspace.latent_table._packed_forward_weight`);
        const slots = positive(latent?.shape[0]);
        matrix(`${child}global_workspace.latent_table`, slots, childWidth);
        const expertIds = new Set([...tensors.keys()].filter((name) => name.startsWith(child + "experts.")).map((name) => name.slice((child + "experts.").length).split(".")[0]));
        if (expertIds.size !== row.experts || [...expertIds].some((id) => !/^\d+$/.test(id!) || Number(id) >= Number(row.experts))) throw new Error("Connected expert owners differ from current layout.");
        leaves.push({ prefix: child, width: childWidth, layers: childLayers, workspaceSlots: slots });
      }
      // Regional recurrent liquid/membrane/spike caches and hidden accumulators.
      connectedActivityBytes += 4 * (4 * childWidth + 2 * positive(config.router_neurons));
      layers += childLayers; experts += Number(row.experts); width += childWidth;
    }
    const parentIds = new Set<string>();
    for (const value of descriptor.parents) {
      const parent = object(value);
      if (typeof parent.parentId !== "string" || parentIds.has(parent.parentId) || !ids.has(String(parent.regionId)) || !digest(parent.generationSha256)) throw new Error("Connected parent aliases are invalid.");
      parentIds.add(parent.parentId);
    }
    if (positive(object(saved.config).d_model) !== width) throw new Error("Connected root width does not equal its current owners.");
    connectedActivityBytes += width * 4 * 12;
    return { layers, experts, width };
  };
  const total = await walk(engine, root.value, root.value.connected_layout, "decoder.");
  for (const name of tensors.keys()) if (/^decoder\..*blocks\.\d+\.attention\.qkv\._packed_forward_weight$/.test(name) && !seenHeaders.has(name)) throw new Error("An attention owner is absent from connected layout.");
  const kvBytesPerToken = leaves.reduce((sum, leaf) => sum + leaf.layers * leaf.width * 8, 0);
  const workspaceStateBytes = leaves.reduce((sum, leaf) => sum + 4 * leaf.workspaceSlots * (2 * leaf.width + 1), 0);
  if (![total.layers, kvBytesPerToken, workspaceStateBytes, connectedActivityBytes].every(Number.isSafeInteger)) throw new Error("Connected geometry exceeds exact accounting.");
  return { leaves, attentionLayers: total.layers, kvBytesPerToken, workspaceStateBytes, connectedActivityBytes };
}
