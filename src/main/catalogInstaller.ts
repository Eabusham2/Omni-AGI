import { createHash } from "node:crypto";
import {
  mkdir,
  readFile,
  readdir,
  rename,
  rm,
  writeFile
} from "node:fs/promises";
import { basename, join } from "node:path";
import { strFromU8, unzipSync } from "fflate";
import {
  BRAIN_SCHEMA_VERSION,
  createPresetConfig,
  type ArchitecturePreset,
  type BrainConfig,
  type BuildRecipe,
  type HardwareTier,
  type InstalledModalityPack,
  type ModalityKind,
  type ModalityPackManifest,
  type ToolPermissionLevel
} from "../shared/types";

const MAX_RECIPE_BYTES = 1024 * 1024;
const MAX_PACK_BYTES = 512 * 1024 * 1024;
const MAX_PACK_UNCOMPRESSED_BYTES = 1024 * 1024 * 1024;
const MAX_PACK_FILES = 4;
const SAFE_ID = /^[a-zA-Z0-9][a-zA-Z0-9._-]{0,127}$/;
const ALLOWED_PACK_FILES = new Set([
  "manifest.json",
  "model-card.md",
  "checksums.sha256",
  "tensors/modality.safetensors"
]);
const MODALITIES: readonly ModalityKind[] = ["vision", "image", "audio", "video"];
const TENSOR_DTYPES = new Map<string, number>([
  ["BOOL", 1],
  ["U8", 1],
  ["I8", 1],
  ["F8_E4M3", 1],
  ["F8_E5M2", 1],
  ["I16", 2],
  ["U16", 2],
  ["F16", 2],
  ["BF16", 2],
  ["I32", 4],
  ["U32", 4],
  ["F32", 4],
  ["I64", 8],
  ["U64", 8],
  ["F64", 8]
]);

interface ZipDirectoryEntry {
  name: string;
  compressedBytes: number;
  uncompressedBytes: number;
}

interface ValidatedPack {
  manifest: ModalityPackManifest;
  checksum: string;
  tensorBytes: Buffer;
  modelCard: string;
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

function sha256(value: Buffer | Uint8Array | string): string {
  return createHash("sha256").update(value).digest("hex");
}

function boundedText(value: unknown, label: string, maximum = 4_000): string {
  if (typeof value !== "string") throw new Error(`${label} must be a string.`);
  const clean = value.replace(/\0/g, "").trim();
  if (!clean || clean.length > maximum) throw new Error(`${label} has an invalid length.`);
  return clean;
}

function optionalHttpsUrl(value: unknown, label: string): string | undefined {
  if (value === undefined) return undefined;
  const text = boundedText(value, label, 8_192);
  const url = new URL(text);
  if (url.protocol !== "https:") throw new Error(`${label} must use HTTPS.`);
  url.username = "";
  url.password = "";
  return url.toString();
}

function assertOnlyKeys(
  value: Record<string, unknown>,
  allowed: readonly string[],
  label: string
): void {
  const permitted = new Set(allowed);
  const unexpected = Object.keys(value).filter((key) => !permitted.has(key));
  if (unexpected.length) {
    throw new Error(`${label} contains unsupported field ${unexpected[0]}.`);
  }
}

function booleanValue(
  value: Record<string, unknown>,
  key: string,
  fallback: boolean
): boolean {
  if (value[key] === undefined) return fallback;
  if (typeof value[key] !== "boolean") throw new Error(`${key} must be boolean.`);
  return value[key];
}

function integerValue(
  value: unknown,
  label: string,
  minimum: number,
  maximum: number
): number {
  if (!Number.isInteger(value) || Number(value) < minimum || Number(value) > maximum) {
    throw new Error(`${label} must be an integer from ${minimum} to ${maximum}.`);
  }
  return Number(value);
}

function recipeMemory(
  value: unknown
): Pick<BrainConfig, "memoryRecipe" | "retainSourceText" | "memoryInjection" | "consolidation"> {
  if (!["human-consolidation", "total-recall", "synapses-only"].includes(String(value))) {
    throw new Error("Recipe memoryRecipe is invalid.");
  }
  const memoryRecipe = value as BrainConfig["memoryRecipe"];
  return {
    memoryRecipe,
    retainSourceText: memoryRecipe === "total-recall",
    memoryInjection: memoryRecipe === "total-recall" ? "working-memory" : "parameter-only",
    consolidation: memoryRecipe !== "total-recall"
  };
}

function recipeModalities(value: unknown): ModalityKind[] {
  if (value === undefined) return [];
  if (!isRecord(value)) throw new Error("Recipe architecture.modalities must be an object.");
  assertOnlyKeys(value, ["vision", "imageGeneration", "audio", "video"], "Recipe modalities");
  const selected: ModalityKind[] = [];
  if (booleanValue(value, "vision", false)) selected.push("vision");
  if (booleanValue(value, "imageGeneration", false)) selected.push("image");
  if (booleanValue(value, "audio", false)) selected.push("audio");
  if (booleanValue(value, "video", false)) selected.push("video");
  return selected;
}

function allToolPermissions(level: ToolPermissionLevel): BuildRecipe["toolPermissions"] {
  return [
    "windows.files",
    "windows.powershell",
    "code.execute",
    "web.search",
    "web.fetch",
    "browser.automation",
    "modality.imagine",
    "agent.fork",
    "source.self-modify"
  ].map((toolId) => ({ toolId, level }));
}

export function validateBuildRecipe(
  contents: Buffer,
  source = "local recipe"
): BuildRecipe {
  if (!contents.byteLength || contents.byteLength > MAX_RECIPE_BYTES) {
    throw new Error("A build recipe must be a non-empty JSON file no larger than 1 MB.");
  }
  let document: unknown;
  try {
    document = JSON.parse(contents.toString("utf8")) as unknown;
  } catch {
    throw new Error("The build recipe is not valid JSON.");
  }
  if (!isRecord(document)) throw new Error("The build recipe must be a JSON object.");
  assertOnlyKeys(
    document,
    [
      "schemaVersion",
      "id",
      "name",
      "description",
      "origin",
      "starterUrl",
      "hardwareProfile",
      "architecture",
      "memoryRecipe",
      "toolPermission",
      "license",
      "provenanceUrl"
    ],
    "Build recipe"
  );
  if (document.schemaVersion !== 1) throw new Error("Unsupported build recipe schema.");
  const id = boundedText(document.id, "Recipe id", 128);
  if (!SAFE_ID.test(id)) throw new Error("Recipe id contains unsupported characters.");
  const name = boundedText(document.name, "Recipe name", 200);
  const description = boundedText(document.description, "Recipe description", 4_000);
  if (!isRecord(document.architecture)) throw new Error("Recipe architecture must be an object.");
  const architecture = document.architecture;
  assertOnlyKeys(
    architecture,
    ["preset", "text", "spiking", "liquid", "ideas", "modalities"],
    "Recipe architecture"
  );
  if (
    !["whole-brain", "ternary", "neuromorphic", "liquid", "symbolic", "custom"].includes(
      String(architecture.preset)
    )
  ) {
    throw new Error("Recipe architecture preset is invalid.");
  }
  const preset = architecture.preset as ArchitecturePreset;
  let config: BrainConfig = {
    ...createPresetConfig(preset, name),
    description
  };

  if (architecture.text !== undefined) {
    if (!isRecord(architecture.text)) throw new Error("Recipe text settings must be an object.");
    assertOnlyKeys(
      architecture.text,
      ["vocabularySize", "contextLength", "dimension", "layers", "heads", "ternary"],
      "Recipe text settings"
    );
    // Shape settings are validated for forward compatibility. Hardware profiling owns
    // their final resolution; no code or command from the recipe is executed.
    if (architecture.text.vocabularySize !== undefined) {
      integerValue(architecture.text.vocabularySize, "vocabularySize", 259, 1_000_000);
    }
    if (architecture.text.contextLength !== undefined) {
      integerValue(architecture.text.contextLength, "contextLength", 16, 1_048_576);
    }
    if (architecture.text.dimension !== undefined) {
      integerValue(architecture.text.dimension, "dimension", 8, 65_536);
    }
    if (architecture.text.layers !== undefined) {
      integerValue(architecture.text.layers, "layers", 1, 1_024);
    }
    if (architecture.text.heads !== undefined) {
      integerValue(architecture.text.heads, "heads", 1, 1_024);
    }
    config = {
      ...config,
      ternaryWeights: booleanValue(architecture.text, "ternary", config.ternaryWeights)
    };
  }

  if (architecture.spiking !== undefined) {
    if (!isRecord(architecture.spiking)) {
      throw new Error("Recipe spiking settings must be an object.");
    }
    assertOnlyKeys(
      architecture.spiking,
      ["enabled", "neurons", "stdp", "metaplasticity", "structuralGrowth"],
      "Recipe spiking settings"
    );
    const neurons =
      architecture.spiking.neurons === undefined
        ? config.initialNeuronBudget
        : integerValue(architecture.spiking.neurons, "spiking neurons", 16, 100_000_000);
    const growth = architecture.spiking.structuralGrowth;
    if (growth !== undefined && !["fixed", "elastic", "unbounded"].includes(String(growth))) {
      throw new Error("Recipe structuralGrowth is invalid.");
    }
    config = {
      ...config,
      spikingDynamics: booleanValue(architecture.spiking, "enabled", config.spikingDynamics),
      stdpPlasticity: booleanValue(architecture.spiking, "stdp", config.stdpPlasticity),
      metaplasticity: booleanValue(
        architecture.spiking,
        "metaplasticity",
        config.metaplasticity
      ),
      initialNeuronBudget: neurons,
      growthPolicy: (growth as BrainConfig["growthPolicy"] | undefined) ?? config.growthPolicy
    };
  }

  if (architecture.liquid !== undefined) {
    if (!isRecord(architecture.liquid)) throw new Error("Recipe liquid settings must be an object.");
    assertOnlyKeys(architecture.liquid, ["enabled", "kind", "units"], "Recipe liquid settings");
    if (
      architecture.liquid.kind !== undefined &&
      !["cfc", "ltc"].includes(String(architecture.liquid.kind))
    ) {
      throw new Error("Recipe liquid kind is invalid.");
    }
    if (architecture.liquid.units !== undefined) {
      integerValue(architecture.liquid.units, "liquid units", 1, 1_000_000);
    }
    config = {
      ...config,
      liquidDynamics: booleanValue(architecture.liquid, "enabled", config.liquidDynamics),
      liquidMode:
        (architecture.liquid.kind as BrainConfig["liquidMode"] | undefined) ?? config.liquidMode
    };
  }

  if (architecture.ideas !== undefined) {
    if (!isRecord(architecture.ideas)) throw new Error("Recipe ideas settings must be an object.");
    assertOnlyKeys(
      architecture.ideas,
      ["enabled", "hypervectorDimensions"],
      "Recipe idea settings"
    );
    if (architecture.ideas.hypervectorDimensions !== undefined) {
      integerValue(
        architecture.ideas.hypervectorDimensions,
        "hypervectorDimensions",
        64,
        1_048_576
      );
    }
    config = {
      ...config,
      vectorSymbolicMemory: booleanValue(
        architecture.ideas,
        "enabled",
        config.vectorSymbolicMemory
      )
    };
  }

  config = { ...config, ...recipeMemory(document.memoryRecipe) };
  const modalities = recipeModalities(architecture.modalities);
  const origin = document.origin;
  if (!["blank", "starter"].includes(String(origin))) throw new Error("Recipe origin is invalid.");
  const starterUrl = optionalHttpsUrl(document.starterUrl, "Recipe starterUrl");
  if (origin === "starter" && !starterUrl) {
    throw new Error("A starter recipe must declare an HTTPS starterUrl.");
  }
  if (
    !["micro", "personal", "gpu", "workstation"].includes(String(document.hardwareProfile))
  ) {
    throw new Error("Recipe hardwareProfile is invalid.");
  }
  if (!["off", "ask", "auto", "full"].includes(String(document.toolPermission))) {
    throw new Error("Recipe toolPermission is invalid.");
  }
  const license =
    document.license === undefined
      ? "Undeclared; verify before use or redistribution"
      : boundedText(document.license, "Recipe license", 2_000);
  return {
    schemaVersion: 1,
    id,
    name,
    description,
    source,
    sha256: sha256(contents),
    license,
    provenanceUrl: optionalHttpsUrl(document.provenanceUrl, "Recipe provenanceUrl"),
    origin: origin as BuildRecipe["origin"],
    starterUrl,
    hardwareTier: document.hardwareProfile as HardwareTier,
    modalities,
    toolPermissions: allToolPermissions(document.toolPermission as ToolPermissionLevel),
    config
  };
}

function parseZipDirectory(buffer: Buffer): ZipDirectoryEntry[] {
  if (buffer.byteLength < 22) throw new Error("The modality pack is not a valid ZIP container.");
  const minimum = Math.max(0, buffer.byteLength - 65_557);
  let end = -1;
  for (let offset = buffer.byteLength - 22; offset >= minimum; offset -= 1) {
    if (buffer.readUInt32LE(offset) === 0x06054b50) {
      end = offset;
      break;
    }
  }
  if (end < 0) throw new Error("The modality pack ZIP directory is missing.");
  const count = buffer.readUInt16LE(end + 10);
  const directoryBytes = buffer.readUInt32LE(end + 12);
  const directoryOffset = buffer.readUInt32LE(end + 16);
  if (
    count < 1 ||
    count > MAX_PACK_FILES ||
    directoryOffset + directoryBytes > end ||
    directoryOffset < 0
  ) {
    throw new Error("The modality pack ZIP directory is invalid.");
  }
  const entries: ZipDirectoryEntry[] = [];
  let cursor = directoryOffset;
  let total = 0;
  for (let index = 0; index < count; index += 1) {
    if (cursor + 46 > end || buffer.readUInt32LE(cursor) !== 0x02014b50) {
      throw new Error("The modality pack has a malformed ZIP entry.");
    }
    const flags = buffer.readUInt16LE(cursor + 8);
    const method = buffer.readUInt16LE(cursor + 10);
    const compressedBytes = buffer.readUInt32LE(cursor + 20);
    const uncompressedBytes = buffer.readUInt32LE(cursor + 24);
    const nameLength = buffer.readUInt16LE(cursor + 28);
    const extraLength = buffer.readUInt16LE(cursor + 30);
    const commentLength = buffer.readUInt16LE(cursor + 32);
    const nameEnd = cursor + 46 + nameLength;
    if (nameEnd > end) throw new Error("The modality pack has a truncated ZIP name.");
    const name = buffer.subarray(cursor + 46, nameEnd).toString("utf8");
    if (
      flags & 0x1 ||
      ![0, 8].includes(method) ||
      !ALLOWED_PACK_FILES.has(name) ||
      name.includes("\\") ||
      name.startsWith("/") ||
      name.split("/").includes("..")
    ) {
      throw new Error(`The modality pack contains unsupported entry ${name || "(empty)"}.`);
    }
    total += uncompressedBytes;
    if (total > MAX_PACK_UNCOMPRESSED_BYTES) {
      throw new Error("The modality pack expands beyond the safe size limit.");
    }
    entries.push({ name, compressedBytes, uncompressedBytes });
    cursor = nameEnd + extraLength + commentLength;
  }
  if (cursor !== directoryOffset + directoryBytes) {
    throw new Error("The modality pack ZIP directory length is inconsistent.");
  }
  return entries;
}

function checksumRecords(text: string): Map<string, string> {
  const result = new Map<string, string>();
  for (const raw of text.split(/\r?\n/)) {
    if (!raw.trim()) continue;
    const match = /^([a-f0-9]{64}) {2}([A-Za-z0-9._/-]+)$/.exec(raw);
    if (!match || result.has(match[2]!)) throw new Error("checksums.sha256 is invalid.");
    result.set(match[2]!, match[1]!);
  }
  return result;
}

function validateSafetensors(buffer: Buffer, modalities: readonly ModalityKind[]): void {
  if (buffer.byteLength < 10) throw new Error("modality.safetensors is invalid.");
  const headerLength = Number(buffer.readBigUInt64LE(0));
  if (
    !Number.isSafeInteger(headerLength) ||
    headerLength < 2 ||
    headerLength > 16 * 1024 * 1024 ||
    8 + headerLength > buffer.byteLength
  ) {
    throw new Error("modality.safetensors has an invalid header length.");
  }
  let header: unknown;
  try {
    header = JSON.parse(buffer.subarray(8, 8 + headerLength).toString("utf8")) as unknown;
  } catch {
    throw new Error("modality.safetensors has invalid header JSON.");
  }
  if (!isRecord(header)) throw new Error("modality.safetensors has an invalid header.");
  const seen = new Set<ModalityKind>();
  let tensorCount = 0;
  for (const [name, descriptor] of Object.entries(header)) {
    if (name === "__metadata__") continue;
    tensorCount += 1;
    if (tensorCount > 100_000 || !isRecord(descriptor)) {
      throw new Error("modality.safetensors has too many or invalid tensors.");
    }
    const modality = modalities.find((kind) => name.startsWith(`modalities.${kind}.`));
    if (!modality) throw new Error(`Tensor ${name} is outside the declared modality namespaces.`);
    if (!TENSOR_DTYPES.has(String(descriptor.dtype)) || !Array.isArray(descriptor.shape)) {
      throw new Error(`Tensor ${name} has an unsupported descriptor.`);
    }
    const offsets = descriptor.data_offsets;
    if (
      !Array.isArray(offsets) ||
      offsets.length !== 2 ||
      !offsets.every(Number.isSafeInteger) ||
      Number(offsets[0]) < 0 ||
      Number(offsets[1]) < Number(offsets[0])
    ) {
      throw new Error(`Tensor ${name} has invalid data offsets.`);
    }
    let values = 1;
    for (const dimension of descriptor.shape) {
      values *= integerValue(dimension, `${name} shape`, 0, 1_000_000_000);
      if (!Number.isSafeInteger(values)) throw new Error(`Tensor ${name} is too large.`);
    }
    const expectedBytes = values * (TENSOR_DTYPES.get(String(descriptor.dtype)) as number);
    if (
      expectedBytes !== Number(offsets[1]) - Number(offsets[0]) ||
      8 + headerLength + Number(offsets[1]) > buffer.byteLength
    ) {
      throw new Error(`Tensor ${name} has inconsistent byte ranges.`);
    }
    seen.add(modality);
  }
  if (!tensorCount || modalities.some((modality) => !seen.has(modality))) {
    throw new Error("The pack must contain at least one tensor for every declared modality.");
  }
}

export function validateModalityPack(
  contents: Buffer,
  sourceLabel = "modality.omnipack"
): ValidatedPack {
  if (!contents.byteLength || contents.byteLength > MAX_PACK_BYTES) {
    throw new Error("A modality pack must be a non-empty file no larger than 512 MB.");
  }
  const directory = parseZipDirectory(contents);
  if (
    directory.length !== ALLOWED_PACK_FILES.size ||
    [...ALLOWED_PACK_FILES].some((name) => !directory.some((entry) => entry.name === name))
  ) {
    throw new Error("The modality pack is missing a required declarative file.");
  }
  let files: Record<string, Uint8Array>;
  try {
    files = unzipSync(new Uint8Array(contents));
  } catch {
    throw new Error(`${basename(sourceLabel)} is not a valid modality-pack ZIP.`);
  }
  const checksums = checksumRecords(strFromU8(files["checksums.sha256"]!));
  for (const name of ALLOWED_PACK_FILES) {
    if (name === "checksums.sha256") continue;
    const file = files[name];
    if (!file || checksums.get(name) !== sha256(file)) {
      throw new Error(`Checksum validation failed for ${name}.`);
    }
  }
  if (
    checksums.size !== ALLOWED_PACK_FILES.size - 1 ||
    [...checksums.keys()].some((name) => name === "checksums.sha256" || !ALLOWED_PACK_FILES.has(name))
  ) {
    throw new Error("checksums.sha256 must cover every declarative pack file exactly once.");
  }
  let value: unknown;
  try {
    value = JSON.parse(strFromU8(files["manifest.json"]!)) as unknown;
  } catch {
    throw new Error("The modality pack manifest is invalid JSON.");
  }
  if (!isRecord(value)) throw new Error("The modality pack manifest must be an object.");
  assertOnlyKeys(
    value,
    [
      "format",
      "formatVersion",
      "architecture",
      "architectureSchemaVersion",
      "pack",
      "licenseLedger",
      "files"
    ],
    "Modality pack manifest"
  );
  if (
    value.format !== "omni-modality-pack" ||
    value.formatVersion !== 1 ||
    value.architecture !== "OmniCortex" ||
    value.architectureSchemaVersion !== BRAIN_SCHEMA_VERSION
  ) {
    throw new Error("The modality pack targets an incompatible OmniCortex schema.");
  }
  if (!isRecord(value.pack)) throw new Error("The modality pack identity is invalid.");
  assertOnlyKeys(value.pack, ["id", "name", "version", "modalities"], "Modality pack identity");
  const id = boundedText(value.pack.id, "Pack id", 128);
  if (!SAFE_ID.test(id)) throw new Error("Pack id contains unsupported characters.");
  const name = boundedText(value.pack.name, "Pack name", 200);
  const version = boundedText(value.pack.version, "Pack version", 100);
  if (
    !Array.isArray(value.pack.modalities) ||
    value.pack.modalities.length < 1 ||
    value.pack.modalities.length > MODALITIES.length ||
    value.pack.modalities.some((item) => !MODALITIES.includes(item as ModalityKind))
  ) {
    throw new Error("The modality pack declares invalid modalities.");
  }
  const modalities = [...new Set(value.pack.modalities as ModalityKind[])];
  if (modalities.length !== value.pack.modalities.length) {
    throw new Error("The modality pack repeats a modality.");
  }
  if (!isRecord(value.licenseLedger)) throw new Error("The pack license ledger is required.");
  assertOnlyKeys(
    value.licenseLedger,
    ["license", "provenanceUrl", "sourceUrl"],
    "Modality pack license ledger"
  );
  const license = boundedText(value.licenseLedger.license, "Pack license", 2_000);
  const provenanceUrl = optionalHttpsUrl(
    value.licenseLedger.provenanceUrl,
    "Pack provenanceUrl"
  );
  const sourceUrl = optionalHttpsUrl(value.licenseLedger.sourceUrl, "Pack sourceUrl");
  if (!isRecord(value.files)) throw new Error("The pack file ledger is required.");
  assertOnlyKeys(value.files, ["model-card.md", "tensors/modality.safetensors"], "Pack files");
  for (const name of ["model-card.md", "tensors/modality.safetensors"] as const) {
    const record = value.files[name];
    const file = files[name]!;
    if (
      !isRecord(record) ||
      record.sha256 !== sha256(file) ||
      record.bytes !== file.byteLength
    ) {
      throw new Error(`The pack file ledger is inconsistent for ${name}.`);
    }
  }
  const tensorBytes = Buffer.from(files["tensors/modality.safetensors"]!);
  validateSafetensors(tensorBytes, modalities);
  const modelCard = boundedText(strFromU8(files["model-card.md"]!), "Pack model card", 256_000);
  return {
    checksum: sha256(contents),
    tensorBytes,
    modelCard,
    manifest: {
      format: "omni-modality-pack",
      formatVersion: 1,
      architecture: "OmniCortex",
      architectureSchemaVersion: BRAIN_SCHEMA_VERSION,
      pack: { id, name, version, modalities },
      licenseLedger: { license, provenanceUrl, sourceUrl },
      files: {
        "model-card.md": {
          sha256: sha256(files["model-card.md"]!),
          bytes: files["model-card.md"]!.byteLength
        },
        "tensors/modality.safetensors": {
          sha256: sha256(tensorBytes),
          bytes: tensorBytes.byteLength
        }
      }
    }
  };
}

async function atomicJson(path: string, value: unknown): Promise<void> {
  const temporary = `${path}.${process.pid}.${Date.now()}.tmp`;
  await writeFile(temporary, `${JSON.stringify(value, null, 2)}\n`, { flag: "wx" });
  await rename(temporary, path);
}

export async function stageModalityPack(
  brainDirectory: string,
  contents: Buffer,
  sourceLabel: string
): Promise<{
  manifest: ModalityPackManifest;
  packPath: string;
  stagingDirectory: string;
  result: InstalledModalityPack;
}> {
  const validated = validateModalityPack(contents, sourceLabel);
  const stagingDirectory = join(
    brainDirectory,
    "packs",
    `${validated.manifest.pack.id}-${validated.checksum.slice(0, 16)}`
  );
  await rm(stagingDirectory, { recursive: true, force: true });
  await mkdir(stagingDirectory, { recursive: true });
  const packPath = join(stagingDirectory, "modality.safetensors");
  await Promise.all([
    writeFile(packPath, validated.tensorBytes, { flag: "wx" }),
    writeFile(join(stagingDirectory, "manifest.json"), `${JSON.stringify(validated.manifest, null, 2)}\n`, {
      flag: "wx"
    }),
    writeFile(join(stagingDirectory, "model-card.md"), `${validated.modelCard.trim()}\n`, {
      flag: "wx"
    })
  ]);
  return {
    manifest: validated.manifest,
    packPath,
    stagingDirectory,
    result: {
      id: validated.manifest.pack.id,
      name: validated.manifest.pack.name,
      version: validated.manifest.pack.version,
      modalities: validated.manifest.pack.modalities,
      license: validated.manifest.licenseLedger.license,
      provenanceUrl: validated.manifest.licenseLedger.provenanceUrl,
      sourceLabel: basename(sourceLabel),
      sha256: validated.checksum,
      installedAt: new Date().toISOString()
    }
  };
}

export async function recordInstalledPack(
  brainDirectory: string,
  pack: InstalledModalityPack
): Promise<void> {
  const root = join(brainDirectory, "packs");
  await mkdir(root, { recursive: true });
  const current = await listInstalledPacks(brainDirectory);
  const next = [
    pack,
    ...current.filter((item) => item.id !== pack.id || item.sha256 !== pack.sha256)
  ].slice(0, 1_000);
  await atomicJson(join(root, "index.json"), next);
}

export async function listInstalledPacks(
  brainDirectory: string
): Promise<InstalledModalityPack[]> {
  const path = join(brainDirectory, "packs", "index.json");
  let value: unknown;
  try {
    value = JSON.parse(await readFile(path, "utf8")) as unknown;
  } catch {
    return [];
  }
  if (!Array.isArray(value)) return [];
  return value.filter((item): item is InstalledModalityPack => {
    if (!isRecord(item)) return false;
    return (
      SAFE_ID.test(String(item.id)) &&
      typeof item.name === "string" &&
      typeof item.version === "string" &&
      Array.isArray(item.modalities) &&
      item.modalities.every((modality) => MODALITIES.includes(modality as ModalityKind)) &&
      typeof item.license === "string" &&
      typeof item.sourceLabel === "string" &&
      /^[a-f0-9]{64}$/.test(String(item.sha256)) &&
      typeof item.installedAt === "string"
    );
  });
}

export async function removeStagedPack(path: string): Promise<void> {
  await rm(path, { recursive: true, force: true });
}

export async function installedPackDirectories(brainDirectory: string): Promise<string[]> {
  try {
    return await readdir(join(brainDirectory, "packs"));
  } catch {
    return [];
  }
}
