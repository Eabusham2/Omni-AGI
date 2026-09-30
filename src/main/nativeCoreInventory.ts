/** Pure constructor-layout arithmetic mirrored by omni_core/native_architecture.py. */
import { createHash } from "node:crypto";
import type { HardwareTier } from "../shared/types";

export interface NativeArchitectureShape {
  dModel: number; layers: number; feedForward: number; nHeads: number;
  vsaDimensions: number; routerNeurons: number; modalityChannels: number;
  imageSize: number; audioSamples: number; videoFrames: number;
  workingMemoryItems: number; workspaceLatents: number; vocabSize: number;
  liquidMode: "cfc" | "ltc";
}

export function nativeCoreInventory(shape: NativeArchitectureShape) {
  const { dModel: d, layers, feedForward: ff, vsaDimensions: vsa,
    routerNeurons: router, modalityChannels: channels, workspaceLatents: workspace,
    vocabSize: vocab, videoFrames: frames } = shape;
  const side = Math.floor(shape.imageSize / 4), audio = Math.floor(shape.audioSamples / 4);
  const counts = { logicalParameters: 0, projectionParameters: 0, tableParameters: 0,
    routerSynapseParameters: router * router, packedWeightBytes: 0,
    resistanceBytes: 0, packedOwners: 0, maximumProjectionBytes: 0 };
  function matrix(width: number, rows: number, bias = 0, table = false, repeat = 1) {
    const parameters = width * rows + bias;
    const packed = rows * Math.ceil(width / 4) + Math.ceil(bias / 4);
    counts.logicalParameters += repeat * parameters;
    counts[table ? "tableParameters" : "projectionParameters"] += repeat * parameters;
    counts.packedWeightBytes += repeat * packed;
    counts.resistanceBytes += repeat * (rows + Number(bias > 0));
    counts.packedOwners += repeat;
    counts.maximumProjectionBytes = Math.max(counts.maximumProjectionBytes, packed + rows + Number(bias > 0) + 8);
  }
  const linear = (width: number, rows: number, bias = false, repeat = 1, table = false) => matrix(width, rows, bias ? rows : 0, table, repeat);
  const norm = (width: number, repeat = 1) => matrix(width, 1, 0, true, repeat);
  const convolution = (inputs: number, outputs: number, kernel: number, transposed = false, repeat = 1) => matrix(transposed ? inputs : inputs * kernel, transposed ? outputs * kernel : outputs, outputs, false, repeat);
  function latentTransformer(tokens: number) {
    linear(tokens, channels, false, 1, true);
    linear(d, channels, true); linear(1, channels, true); linear(channels, channels, true);
    linear(channels, 3 * channels, true, 2); linear(channels, channels, true, 2);
    linear(channels, 3 * channels, true, 2); linear(3 * channels, channels, true, 2);
    linear(channels, channels, true);
  }
  matrix(d, vocab, 0, true); matrix(d, workspace, 0, true);
  linear(d, d, false, 5); norm(d); linear(d, d); matrix(16, 1, 0, true, 2);
  norm(d, 2 * layers); linear(d, 3 * d, false, layers); linear(d, d, false, layers);
  linear(d, 2 * ff, false, layers); linear(ff, d, false, layers); norm(d); linear(d, vocab);
  norm(d, 2); linear(d, 4 * d, true, 2); linear(4 * d, 8, true, 2);
  linear(1024, 96, true); linear(d, 96, true); linear(1024, 96);
  linear(d + 1024, d, true); matrix(d, 258, 0, true); linear(2 * d, d, true); linear(d, 257, true);
  linear(vsa, d, true); linear(d, 2 * d, true); linear(2 * d, d, true);
  linear(d, router, true); linear(router, d, true);
  if (shape.liquidMode === "ltc") {
    linear(2 * d, d, true, 2); linear(d, d, true); linear(1, d, false, 2);
  } else linear(2 * d, d, true, 3);
  linear(d, 4, true);
  convolution(3, channels, 9); convolution(channels, 2 * channels, 9); linear(2 * channels, d, true);
  convolution(3, channels, 16); convolution(channels, channels, 16); linear(32, channels, false, 1, true);
  convolution(channels, channels, 16, true); convolution(channels, 3, 16, true);
  linear(d, channels, true); linear(channels, d, true); latentTransformer(side * side);
  convolution(1, channels, 4); convolution(channels, channels, 4);
  linear(32, channels, false, 1, true); linear(16, channels, false, 1, true);
  convolution(channels, channels, 4, true); convolution(channels, 1, 4, true);
  linear(d, channels * audio, true); linear(channels, d, true); latentTransformer(audio);
  linear(d, channels * frames * side * side, true);
  convolution(channels, channels, 3, false, 2); convolution(channels, channels, 9, false, 2);
  linear(2 * channels, channels, true, 2);
  convolution(channels, channels, 16, true); convolution(channels, 3, 16, true);
  convolution(3, channels, 16); convolution(channels, channels, 16); linear(channels, d, true); linear(d, 3, true);
  counts.logicalParameters += router * router;
  counts.packedWeightBytes += router * Math.ceil(router / 4);
  const gainAndRateBytes = 8 * counts.packedOwners;
  // STDP counters plus active-prefix and initial region-end controls.
  const routerNonweightTensorBytes = 10 * router * router + 16 * router + 32;
  const freshRouteControlBytes = 4096 + 24 + 16;
  const liquidStateBytes = 4 * d;
  return { ...counts, gainAndRateBytes, routerNonweightTensorBytes, freshRouteControlBytes,
    liquidStateBytes, staticNonweightTensorBytes: counts.resistanceBytes + gainAndRateBytes + routerNonweightTensorBytes + freshRouteControlBytes + liquidStateBytes };
}

export type NativeCoreInventory = ReturnType<typeof nativeCoreInventory>;

export interface NativeArchitectureDescriptor {
  format: "omni-main-selected-native-architecture";
  formatVersion: 1;
  architecture: "OmniCortex";
  externalPretrainedWeights: false;
  hardwareTier: HardwareTier;
  shape: NativeArchitectureShape;
  inventory: NativeCoreInventory;
  sizing: Record<string, string | number | boolean | null>;
  qualityEvidence: "unmeasured-native-quality-deferred";
  sha256: string;
}

function canonical(value: unknown): string {
  if (value === null || typeof value !== "object") {
    if (typeof value === "number" && (!Number.isFinite(value) || !Number.isSafeInteger(value))) {
      throw new Error("Native descriptor numbers must be safe whole integers.");
    }
    return JSON.stringify(value);
  }
  if (Array.isArray(value)) return `[${value.map(canonical).join(",")}]`;
  return `{${Object.keys(value).sort().map((key) => `${JSON.stringify(key)}:${canonical((value as Record<string, unknown>)[key])}`).join(",")}}`;
}

export function nativeArchitectureSha256(descriptor: Omit<NativeArchitectureDescriptor, "sha256"> | NativeArchitectureDescriptor): string {
  const { sha256: _ignored, ...payload } = descriptor as NativeArchitectureDescriptor;
  return createHash("sha256").update(canonical(payload)).digest("hex");
}

export function sealNativeArchitecture(descriptor: Omit<NativeArchitectureDescriptor, "sha256">): NativeArchitectureDescriptor {
  return { ...descriptor, sha256: nativeArchitectureSha256(descriptor) };
}

export function validateNativeArchitectureDescriptor(value: unknown): NativeArchitectureDescriptor {
  const record = (candidate: unknown): candidate is Record<string, unknown> =>
    typeof candidate === "object" && candidate !== null && !Array.isArray(candidate);
  if (!record(value) || value.format !== "omni-main-selected-native-architecture" || value.formatVersion !== 1
      || value.architecture !== "OmniCortex" || value.externalPretrainedWeights !== false
      || value.qualityEvidence !== "unmeasured-native-quality-deferred"
      || !["micro", "personal", "gpu", "workstation"].includes(String(value.hardwareTier))
      || "foundationModelId" in value || "foundation_model_id" in value) {
    throw new Error("Invalid trusted native architecture identity/quality descriptor.");
  }
  if (!record(value.shape) || !record(value.inventory) || !record(value.sizing)) {
    throw new Error("Native architecture shape, inventory and sizing must be objects.");
  }
  const shapeValue = value.shape;
  const inventoryValue = value.inventory;
  const keys = ["dModel", "layers", "feedForward", "nHeads", "vsaDimensions", "routerNeurons", "modalityChannels",
    "imageSize", "audioSamples", "videoFrames", "workingMemoryItems", "workspaceLatents", "vocabSize"] as const;
  if (Object.keys(shapeValue).length !== keys.length + 1 || keys.some((key) =>
    typeof shapeValue[key] !== "number" || !Number.isSafeInteger(shapeValue[key]) || Number(shapeValue[key]) < 1)) {
    throw new Error("Native architecture dimensions must be positive safe integers.");
  }
  const shape = shapeValue as unknown as NativeArchitectureShape;
  if (shape.dModel % shape.nHeads !== 0 || (shape.dModel / shape.nHeads) % 2 !== 0
      || shape.vocabSize !== 261 || shape.vsaDimensions <= 8 || shape.routerNeurons < 2
      || shape.imageSize < 8 || shape.imageSize % 4 !== 0 || shape.audioSamples < 4 || shape.audioSamples % 4 !== 0
      || shape.videoFrames < 2 || !["cfc", "ltc"].includes(shape.liquidMode)
      || shape.workspaceLatents !== Math.max(8, Math.floor(shape.workingMemoryItems / 4))) {
    throw new Error("Native architecture dimensions violate the constructor contract.");
  }
  const exact = nativeCoreInventory(shape);
  if (Object.keys(inventoryValue).length !== Object.keys(exact).length
      || Object.entries(exact).some(([key, count]) => !Number.isSafeInteger(count) || inventoryValue[key] !== count)) {
    throw new Error("Native architecture exact inventory does not match its shape.");
  }
  if (Object.values(value.sizing).some((entry) => entry !== null && typeof entry !== "string"
      && typeof entry !== "boolean" && !(typeof entry === "number" && Number.isSafeInteger(entry)))) {
    throw new Error("Native architecture sizing must be flat canonical JSON primitives.");
  }
  const descriptor = value as unknown as NativeArchitectureDescriptor;
  if (typeof descriptor.sha256 !== "string" || !/^[0-9a-f]{64}$/.test(descriptor.sha256)
      || nativeArchitectureSha256(descriptor) !== descriptor.sha256) {
    throw new Error("Native architecture descriptor hash mismatch.");
  }
  return descriptor;
}
