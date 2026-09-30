import rawArchitecture from "../../architecture/omnicortex-ground-up-v1.json";
import type { HardwareTier, WorkingMemoryMode } from "../shared/types";
import {
  nativeCoreInventory, sealNativeArchitecture,
  type NativeArchitectureDescriptor, type NativeArchitectureShape
} from "./nativeCoreInventory";
import { validateNativeProjectionComputeProfile, type NativeProjectionComputeProfile } from "./nativeComputeMeasurement";

const MIB = 1024 ** 2;

export interface GroundUpArchitectureProfile {
  format: "omni-ground-up-architecture-profile";
  formatVersion: 1;
  architecture: "OmniCortex";
  origin: "ground-up-random-initialization";
  externalPretrainedWeights: false;
  hardwareTier: HardwareTier;
  dModel: number;
  layers: number;
  feedForward: number;
  vsaDimensions: number;
  routerNeurons: number;
  workingMemoryItems: number;
  workspaceLatents: number;
  exactLogicalParameterCount: number;
  exactPackedProjectionParameterCount: number;
  exactPackedTableParameterCount: number;
  exactPackedTernaryParameterCount: number;
  packedTernaryWeightBytes: number;
  packedWorkspaceTableBytes: number;
  packedMetaplasticityReserveBytes: number;
  fixedControlBufferReserveBytes: number;
  packedUpdateScratchBytes: number;
  residentInferenceStateBytes: number;
  minimumTrainingStateBytes: number;
  checkpointTensorBytes: number;
  parameterCountBasis: "architecture-logical-neural-elements";
  checkpointByteBasis: "packed-weights-plus-nonweight-state-reserve";
  nativeArchitecture?: NativeArchitectureDescriptor;
}

type RawTier = {
  dModel: number;
  layers: number;
  feedForward: number;
  vsaDimensions: number;
  routerNeurons: number;
  autoWorkingMemoryItems: number;
  baseParametersExcludingWorkspaceLatents: number;
  exactPackedProjectionParameters: number;
  exactPackedTableParametersExcludingWorkspaceLatents: number;
  packedControlParameters: number;
  packedControlBytes: number;
  packedBytesExcludingWorkspaceLatents: number;
};

type RawArchitecture = {
  format: string;
  formatVersion: number;
  architecture: string;
  origin: string;
  externalPretrainedWeights: boolean;
  workspaceLatentRule: {
    minimumLatents: number;
    workingMemoryItemsPerLatent: number;
  };
  storageAccounting: {
    packedTernaryBitsPerWeight: number;
    routerEligibilityBytesPerSynapse: number;
    routerStabilityAndUsageBytesPerSynapse: number;
    fixedControlBufferReserveBytes: number;
    packedProjectionUpdateRowBlock: number;
    packedProjectionScratchBytesPerBlockElement: number;
  };
  profiles: Record<HardwareTier, RawTier>;
};

const architecture = rawArchitecture as RawArchitecture;

function positiveSafeInteger(value: number, label: string): number {
  if (!Number.isSafeInteger(value) || value < 1) {
    throw new Error(`OmniCortex architecture ${label} is invalid.`);
  }
  return value;
}

function validateArchitecture(): void {
  if (
    architecture.format !== "omni-ground-up-architecture-profile" ||
    architecture.formatVersion !== 1 ||
    architecture.architecture !== "OmniCortex" ||
    architecture.origin !== "ground-up-random-initialization" ||
    architecture.externalPretrainedWeights !== false ||
    architecture.storageAccounting.packedTernaryBitsPerWeight !== 2
  ) {
    throw new Error("OmniCortex architecture identity is invalid.");
  }
  positiveSafeInteger(
    architecture.workspaceLatentRule.minimumLatents,
    "minimum workspace latent count"
  );
  positiveSafeInteger(
    architecture.workspaceLatentRule.workingMemoryItemsPerLatent,
    "workspace divisor"
  );
  for (const [key, value] of Object.entries(architecture.storageAccounting)) {
    positiveSafeInteger(value, `storageAccounting.${key}`);
  }
  for (const tier of ["micro", "personal", "gpu", "workstation"] as const) {
    const profile = architecture.profiles[tier];
    if (!profile) throw new Error(`OmniCortex architecture has no ${tier} profile.`);
    for (const [key, value] of Object.entries(profile)) {
      positiveSafeInteger(value, `${tier}.${key}`);
    }
    const basePackedCount =
      profile.exactPackedProjectionParameters +
      profile.exactPackedTableParametersExcludingWorkspaceLatents +
      profile.routerNeurons ** 2;
    // The initial decoder has one packed RMSNorm gain per channel in the
    // global workspace, two action heads, final norm, and two per block.
    // Two 16-digit packed scalar gains add 32 logical ternary values. Fresh
    // expert-route count is zero; grown routes enter runtime counts later.
    const normCount = 2 * profile.layers + 4;
    const expectedControlParameters = normCount * profile.dModel + 32;
    const expectedControlBytes =
      normCount * Math.ceil(profile.dModel / 4) + 8;
    if (
      basePackedCount !==
        profile.baseParametersExcludingWorkspaceLatents ||
      profile.packedControlParameters !== expectedControlParameters ||
      profile.packedControlBytes !== expectedControlBytes ||
      profile.exactPackedTableParametersExcludingWorkspaceLatents <
        profile.packedControlParameters ||
      profile.packedBytesExcludingWorkspaceLatents <
        Math.ceil(basePackedCount *
          architecture.storageAccounting.packedTernaryBitsPerWeight / 8)
    ) {
      throw new Error(`OmniCortex architecture ${tier} logical and packed counts disagree.`);
    }
  }
}

validateArchitecture();

export function defaultGroundUpWorkingMemoryItems(
  hardwareTier: HardwareTier,
  mode: WorkingMemoryMode = "auto"
): number {
  const auto = architecture.profiles[hardwareTier].autoWorkingMemoryItems;
  return mode === "extended" ? auto * 4 : auto;
}

export function groundUpArchitectureProfile(
  hardwareTier: HardwareTier,
  workingMemoryItems = defaultGroundUpWorkingMemoryItems(hardwareTier)
): GroundUpArchitectureProfile {
  const selectedItems = positiveSafeInteger(
    Math.floor(workingMemoryItems),
    "working-memory item count"
  );
  const source = architecture.profiles[hardwareTier];
  const workspaceLatents = Math.max(
    architecture.workspaceLatentRule.minimumLatents,
    Math.floor(
      selectedItems /
        architecture.workspaceLatentRule.workingMemoryItemsPerLatent
    )
  );
  const exactLogicalParameterCount = positiveSafeInteger(
    source.baseParametersExcludingWorkspaceLatents +
      workspaceLatents * source.dModel,
    "logical neural parameter count"
  );
  const exactPackedProjectionParameterCount =
    source.exactPackedProjectionParameters;
  const exactPackedTableParameterCount =
    source.exactPackedTableParametersExcludingWorkspaceLatents +
      workspaceLatents * source.dModel;
  const routerSynapses = source.routerNeurons * source.routerNeurons;
  const exactPackedTernaryParameterCount =
    exactPackedProjectionParameterCount +
    exactPackedTableParameterCount + routerSynapses;
  const accounting = architecture.storageAccounting;
  if (exactPackedTernaryParameterCount !== exactLogicalParameterCount) {
    throw new Error("OmniCortex learned weights must all be packed ternary.");
  }
  // The measured base includes row padding and packed biases. Workspace
  // rows are derived from the selected working-memory capacity, rather than
  // charging a floating table for every learned latent slot.
  const packedWorkspaceTableBytes = positiveSafeInteger(
    workspaceLatents * Math.ceil(
      source.dModel * accounting.packedTernaryBitsPerWeight / 8
    ),
    "packed workspace table byte count"
  );
  const packedTernaryWeightBytes = positiveSafeInteger(
    source.packedBytesExcludingWorkspaceLatents +
      packedWorkspaceTableBytes,
    "packed ternary weight byte count"
  );
  // The worker checkpoints one uint8 resistance per packed output row, not
  // per learned weight. A static profile cannot list each modality/module
  // row here, so budget the known workspace rows plus conservative estimates
  // for narrower projection/table rows and 16 KiB for skinny/bias rows. This
  // is non-weight headroom, not an exact row inventory or an extra parameter.
  const conservativeRowWidth = Math.max(4, Math.floor(source.dModel / 4));
  const packedMetaplasticityReserveBytes =
    workspaceLatents +
    Math.ceil(source.exactPackedProjectionParameters / conservativeRowWidth) +
    Math.ceil(source.exactPackedTableParametersExcludingWorkspaceLatents /
      conservativeRowWidth) +
    16 * 1024;
  const fixedControlBufferReserveBytes =
    accounting.fixedControlBufferReserveBytes;
  const routerPlasticityBytes = routerSynapses * (
    accounting.routerEligibilityBytesPerSynapse +
    accounting.routerStabilityAndUsageBytesPerSynapse
  );
  // The packed gradient learner decodes only bounded row blocks. This is
  // transient scratch, not another resident dense projection master or Adam
  // state. The general activation budget is added by ResourcePlanner.
  const packedUpdateScratchBytes = Math.max(
    4 * MIB,
    accounting.packedProjectionUpdateRowBlock *
      Math.max(
        source.dModel,
        source.feedForward,
        source.vsaDimensions,
        source.routerNeurons
      ) * accounting.packedProjectionScratchBytesPerBlockElement
  );
  const residentInferenceStateBytes = positiveSafeInteger(
    packedTernaryWeightBytes +
      packedMetaplasticityReserveBytes +
      fixedControlBufferReserveBytes + routerPlasticityBytes,
    "resident inference-state byte count"
  );
  // Bounded packed-update scratch is transient training RAM, not a resident
  // dense master, dense gradient, or optimizer moment for learned weights.
  const minimumTrainingStateBytes = positiveSafeInteger(
    residentInferenceStateBytes + packedUpdateScratchBytes,
    "minimum training-state byte count"
  );
  // Durable generations contain packed learned weights plus non-weight
  // metaplastic/router/control state. Scratch and activations do not enter.
  const checkpointTensorBytes = Math.max(
    MIB,
    positiveSafeInteger(
      residentInferenceStateBytes + MIB,
      "checkpoint tensor byte count"
    )
  );

  return {
    format: "omni-ground-up-architecture-profile",
    formatVersion: 1,
    architecture: "OmniCortex",
    origin: "ground-up-random-initialization",
    externalPretrainedWeights: false,
    hardwareTier,
    dModel: source.dModel,
    layers: source.layers,
    feedForward: source.feedForward,
    vsaDimensions: source.vsaDimensions,
    routerNeurons: source.routerNeurons,
    workingMemoryItems: selectedItems,
    workspaceLatents,
    exactLogicalParameterCount,
    exactPackedProjectionParameterCount,
    exactPackedTableParameterCount,
    exactPackedTernaryParameterCount,
    packedTernaryWeightBytes,
    packedWorkspaceTableBytes,
    packedMetaplasticityReserveBytes,
    fixedControlBufferReserveBytes,
    packedUpdateScratchBytes,
    residentInferenceStateBytes,
    minimumTrainingStateBytes,
    checkpointTensorBytes,
    parameterCountBasis: "architecture-logical-neural-elements",
    checkpointByteBasis: "packed-weights-plus-nonweight-state-reserve"
  };
}

/** Preserve the current shipped policy while a NEW sizing policy is unselected. */
export function legacyNativeArchitectureDescriptor(hardwareTier: HardwareTier, items: number): NativeArchitectureDescriptor {
  const source = architecture.profiles[hardwareTier];
  const media = { micro: [8, 64, 2, 8], personal: [16, 256, 4, 16],
    gpu: [32, 512, 6, 24], workstation: [32, 1024, 8, 32] }[hardwareTier]!;
  const shape: NativeArchitectureShape = {
    dModel: source.dModel, layers: source.layers, feedForward: source.feedForward,
    nHeads: source.dModel <= 64 ? 4 : 8, vsaDimensions: source.vsaDimensions,
    routerNeurons: source.routerNeurons, modalityChannels: media[3]!, imageSize: media[0]!,
    audioSamples: media[1]!, videoFrames: media[2]!, workingMemoryItems: items,
    workspaceLatents: Math.max(8, Math.floor(items / 4)), vocabSize: 261, liquidMode: "cfc"
  };
  return sealNativeArchitecture({
    format: "omni-main-selected-native-architecture", formatVersion: 1, architecture: "OmniCortex",
    externalPretrainedWeights: false, hardwareTier, shape, inventory: nativeCoreInventory(shape),
    sizing: { policy: "existing-profile-preserved-new-sizing-policy-unpromoted",
      isNewCapacityPolicy: false, throughputMeasured: false },
    qualityEvidence: "unmeasured-native-quality-deferred"
  });
}

/** Inputs are main-process measurements/plan commitments, not renderer shapes. */
export interface NativeSizingInput {
  hardwareTier: HardwareTier;
  workingMemoryItems: number;
  selectedSystemRamBudgetBytes: number;
  runtimeBaselineReserveBytes: number;
  selectedStoragePoolBytes: number;
  measuredStorageBytesPerSecond: number;
  estimatedTrainingSourceBytes: number;
  acceleratorAvailable: boolean;
  liquidMode?: "cfc" | "ltc";
  estimatedNeuralGrowthReserveBytes?: number;
  trainingScratchReserveBytes?: number;
  /** Resident device-derived context commitment; never a context upper cap. */
  baselineContextTokens?: number;
  baselineResidentWorkingMemoryItems?: number;
}

/** Default new-brain policy: total selected RAM, resident activity and slack. */
export function ramFirstNativeArchitectureProfile(input: NativeSizingInput): GroundUpArchitectureProfile {
  const baselineContextTokens = positiveSafeInteger(Math.floor(input.baselineContextTokens ?? 1), "resident baseline context");
  return deriveNativeArchitectureProfile(input, undefined, { baselineContextTokens });
}

export function capacityDerivedNativeArchitectureProfile(input: NativeSizingInput): GroundUpArchitectureProfile {
  return deriveNativeArchitectureProfile(input);
}

/** Internal candidate only. The work budget must be chosen by trusted main. */
export function balancedMeasuredNativeArchitectureProfile(
  input: NativeSizingInput,
  measurement: NativeProjectionComputeProfile,
  primitiveWorkBudgetMicroseconds: number
): GroundUpArchitectureProfile {
  const profile = validateNativeProjectionComputeProfile(measurement);
  positiveSafeInteger(primitiveWorkBudgetMicroseconds, "primitive proxy work budget");
  const budget = BigInt(profile.projectionMacsPerSecond) * BigInt(primitiveWorkBudgetMicroseconds) / 1_000_000n;
  return deriveNativeArchitectureProfile(input, { profile, primitiveWorkBudgetMicroseconds,
    maximumTraversalMacsProxy: Number(budget > BigInt(Number.MAX_SAFE_INTEGER) ? BigInt(Number.MAX_SAFE_INTEGER) : budget) });
}

function deriveNativeArchitectureProfile(input: NativeSizingInput, compute?: {
  profile: NativeProjectionComputeProfile;
  primitiveWorkBudgetMicroseconds: number;
  maximumTraversalMacsProxy: number;
}, ramFirst?: { baselineContextTokens: number }): GroundUpArchitectureProfile {
  const tier = input.hardwareTier;
  const base = architecture.profiles[tier];
  const items = positiveSafeInteger(Math.floor(input.workingMemoryItems), "working memory");
  const residual = Math.max(0, Math.floor(input.selectedSystemRamBudgetBytes) - Math.floor(input.runtimeBaselineReserveBytes));
  const corePartition = Math.floor(residual * 0.55);
  const sourceBytes = Math.max(0, Math.floor(input.estimatedTrainingSourceBytes));
  const transferPartition = residual - corePartition - Math.floor(residual * 0.30);
  const activityPartition = Math.floor(residual * 0.30);
  const estimatedNeuralGrowthReserve = Math.max(0, Math.floor(input.estimatedNeuralGrowthReserveBytes ?? 0));
  const trainingScratchReserve = Math.max(0, Math.floor(input.trainingScratchReserveBytes ?? 0));
  const checkpointCopiesReserve = 8;
  const storageTarget = Math.max(0, Math.floor(
    (input.selectedStoragePoolBytes - estimatedNeuralGrowthReserve - trainingScratchReserve) / checkpointCopiesReserve));
  // Auto's pool grows to publish the selected RAM baseline. A tiny bootstrap
  // pool (or temporary busy-app free RAM) must not choose a tiny cortex.
  const target = ramFirst ? corePartition : Math.max(0, Math.min(corePartition, storageTarget));
  let layers = ramFirst
    ? Math.max(2, 2 + Math.floor(Math.log2(1 + residual / (256 * MIB))))
    : Math.max(base.layers, base.layers + Math.floor(Math.log2(1 + target / Math.max(MIB, base.packedBytesExcludingWorkspaceLatents))));
  const media = {
    micro: { image: 8, audio: 64, frames: 2, channels: 8 },
    personal: { image: 16, audio: 256, frames: 4, channels: 16 },
    gpu: { image: 32, audio: 512, frames: 6, channels: 24 },
    workstation: { image: 32, audio: 1024, frames: 8, channels: 32 }
  }[tier];
  function shapeForWidth(width: number): NativeArchitectureShape {
    let heads = width <= 64 ? 4 : 8;
    if (width >= 1024 && width % 16 === 0) heads = 16;
    return {
      dModel: width, layers, feedForward: width * 3, nHeads: heads,
      vsaDimensions: Math.max(128, width * 4),
      routerNeurons: Math.max(base.routerNeurons, Math.floor(width / 4 / 8) * 8),
      modalityChannels: Math.max(media.channels, Math.floor(width / 4 / 8) * 8),
      imageSize: media.image, audioSamples: media.audio, videoFrames: media.frames,
      workingMemoryItems: items, workspaceLatents: Math.max(8, Math.floor(items / 4)),
      vocabSize: 261, liquidMode: input.liquidMode ?? "cfc"
    };
  }
  function cost(shape: NativeArchitectureShape) {
    const inventory = nativeCoreInventory(shape);
    const derivedControlBytes = 4 * inventory.packedOwners + 8 * (2 * shape.layers + 6)
      + 2 * (shape.dModel / shape.nHeads) * shape.layers;
    const residentTensorAndStateReserve = inventory.packedWeightBytes + inventory.staticNonweightTensorBytes + derivedControlBytes + 16 * 1024;
    const checkpointReserve = residentTensorAndStateReserve + MIB;
    const workspaceBytes = shape.workspaceLatents * Math.ceil(shape.dModel / 4);
    const packedRowScratchReserve = 64 * 32 * Math.max(shape.dModel, shape.feedForward, shape.vsaDimensions, 1024, shape.audioSamples / 4, shape.imageSize ** 2 / 16);
    // Same one-query/one-key conservative first-order tile geometry as the
    // activity pager, plus a bounded transfer chunk. Not a neural benchmark.
    const minimumAttentionTileBytes = 80 * shape.dModel + 73 * shape.nHeads;
    const minimumTransferComputeReserve = packedRowScratchReserve + minimumAttentionTileBytes + 4 * MIB;
    const minimumReturnedOutputReserve = 4 * 8 * (shape.dModel * (3 * shape.layers + 12) + 6 * shape.feedForward + shape.vocabSize);
    const residentModelWithHeadroomReserve = Math.max(Math.ceil(residentTensorAndStateReserve * 1.2), residentTensorAndStateReserve + packedRowScratchReserve);
    const residentItemBytesEstimate = Math.max(4096, 4 * shape.dModel + 512);
    const baselineItems = Math.min(shape.workingMemoryItems, input.baselineResidentWorkingMemoryItems ?? shape.workingMemoryItems);
    const baselineLatents = Math.max(8, Math.floor(baselineItems / 4));
    const baselineWorkspaceRuntimeReserve = 4 * baselineLatents * (2 * shape.dModel + 1)
      + Math.ceil(baselineLatents / 256) * 8192 + 4096 * shape.layers;
    const baselineKvAndIndexReserve = (ramFirst?.baselineContextTokens ?? 0) * (8 * shape.layers * shape.dModel + 96 + 16 * shape.layers);
    const baselineFastItemReserve = baselineItems * residentItemBytesEstimate;
    const baselineActivityWithHeadroomReserve = Math.ceil((baselineWorkspaceRuntimeReserve + baselineKvAndIndexReserve + baselineFastItemReserve) * 1.2);
    // Timing/STDP transient arrays are not a floating learned-weight master.
    // Cross-backend allocator/graph costs remain an explicit estimate.
    const boundedTrainingWithHeadroomReserve = Math.ceil((minimumTransferComputeReserve + minimumReturnedOutputReserve
      + 32 * shape.routerNeurons ** 2 + 4 * MIB) * 1.2);
    // This is a declared work *proxy*, not actual chat execution work. It
    // counts all projection entries plus one recurrent traversal, even though
    // an ordinary text turn does not execute every modality or every owner.
    const traversalMacsProxy = inventory.projectionParameters + shape.routerNeurons ** 2;
    return { inventory, checkpointReserve, nativeKernelReserve: checkpointReserve - workspaceBytes, traversalMacsProxy,
      residentTensorAndStateReserve, residentModelWithHeadroomReserve, baselineActivityWithHeadroomReserve,
      baselineWorkspaceRuntimeReserve, baselineKvAndIndexReserve, baselineFastItemReserve, residentItemBytesEstimate,
      boundedTrainingWithHeadroomReserve,
      packedRowScratchReserve, minimumAttentionTileBytes, minimumTransferComputeReserve, minimumReturnedOutputReserve };
  }
  function fitsPhysical(shape: NativeArchitectureShape): boolean {
    const value = cost(shape);
    return Object.values(value.inventory).every(Number.isSafeInteger)
      && (ramFirst ? value.residentModelWithHeadroomReserve <= corePartition
        && value.baselineActivityWithHeadroomReserve <= activityPartition
        && value.boundedTrainingWithHeadroomReserve <= transferPartition
        : value.checkpointReserve <= target
          && value.minimumTransferComputeReserve <= transferPartition
          && value.minimumReturnedOutputReserve <= activityPartition);
  }
  function fits(shape: NativeArchitectureShape): boolean {
    return fitsPhysical(shape) && (!compute || cost(shape).traversalMacsProxy <= compute.maximumTraversalMacsProxy);
  }
  if (compute) {
    // Joint depth/width admission against the explicit primitive-work proxy;
    // there is no fixed parameter-count or 2%/8% model-core fraction ceiling.
    const physicalDepth = layers;
    let lowerDepth = base.layers, upperDepth = physicalDepth + 1;
    while (upperDepth - lowerDepth > 1) {
      layers = Math.floor((lowerDepth + upperDepth) / 2);
      if (fits(shapeForWidth(base.dModel))) lowerDepth = layers;
      else upperDepth = layers;
    }
    layers = lowerDepth;
  }
  const minimumWidth = ramFirst ? 32 : base.dModel;
  const minimum = shapeForWidth(minimumWidth);
  let low = Math.floor(minimumWidth / 32), high = Math.max(low + 1, low * 2);
  while (fits(shapeForWidth(high * 32))) {
    high *= 2;
  }
  while (high - low > 1) {
    const middle = Math.floor((high + low) / 2);
    if (fits(shapeForWidth(middle * 32))) low = middle;
    else high = middle;
  }
  const shape = shapeForWidth(low * 32);
  const { inventory, checkpointReserve } = cost(shape);
  const derivedControlBytes = 4 * inventory.packedOwners + 8 * (2 * shape.layers + 6)
    + 2 * (shape.dModel / shape.nHeads) * shape.layers;
  if (Object.values(inventory).some((value) => !Number.isSafeInteger(value))) {
    throw new Error("Selected native architecture exceeds exact integer accounting.");
  }
  const descriptor = sealNativeArchitecture({
    format: "omni-main-selected-native-architecture", formatVersion: 1,
    architecture: "OmniCortex", externalPretrainedWeights: false,
    hardwareTier: tier, shape, inventory,
    sizing: {
      policy: ramFirst ? "total-envelope-ram-first-resident-baseline-headroom-v1" : compute ? "shared-envelope-measured-primitive-work-proxy-candidate-v1" : "shared-envelope-physical-capacity-admission-v2",
      selectedSystemRamBudgetBytes: Math.floor(input.selectedSystemRamBudgetBytes),
      runtimeBaselineReserveBytes: Math.floor(input.runtimeBaselineReserveBytes),
      baselineEvidence: "prebuild-runtime-reserve-not-worker-rss-measurement",
      corePartitionBytes: corePartition, corePartitionPercent: 55,
      workingActivityPartitionPercent: 30, trainingTransferPartitionPercent: 15,
      selectedStoragePoolBytes: Math.floor(input.selectedStoragePoolBytes),
      measuredStorageBytesPerSecond: Math.floor(input.measuredStorageBytesPerSecond),
      estimatedTrainingSourceBytes: sourceBytes,
      sourceEvidence: "source-byte-estimate-not-token-coverage",
      estimatedNeuralGrowthReserveBytes: estimatedNeuralGrowthReserve,
      trainingScratchReserveBytes: trainingScratchReserve,
      checkpointCopiesReserve,
      targetCheckpointReserveBytes: target,
      transferPartitionBytes: transferPartition,
      activityPartitionBytes: activityPartition,
      minimumAttentionTileBytes: cost(shape).minimumAttentionTileBytes,
      packedRowScratchReserveBytes: cost(shape).packedRowScratchReserve,
      minimumTransferComputeReserveBytes: cost(shape).minimumTransferComputeReserve,
      minimumReturnedTrainingOutputReserveBytes: cost(shape).minimumReturnedOutputReserve,
      packedWorkspaceTableBytesExact: shape.workspaceLatents * Math.ceil(shape.dModel / 4),
      workspaceReserveSeparatelyCharged: true,
      selectedNativeKernelReserveBytes: cost(shape).nativeKernelReserve,
      selectedCheckpointReserveBytes: checkpointReserve,
      staticDerivedControlTensorBytesExact: derivedControlBytes,
      rngAndPlatformStateReserveBytes: 16 * 1024,
      allocatorAndDynamicActivityBytesExact: false,
      minimumShapeCheckpointReserveBytes: cost(minimum).checkpointReserve,
      fitsPolicyTarget: fits(shape),
      fitsPhysicalTarget: fitsPhysical(shape),
      fitsPrimitiveWorkProxyBudget: !compute || cost(shape).traversalMacsProxy <= compute.maximumTraversalMacsProxy,
      ...(ramFirst ? {
        baselineSizingAssumesOtherApplicationsClosed: true,
        transientAvailableRamUsedForShape: false, ramPercentageIsCeilingNotUsageTarget: true,
        baselineContextTokens: ramFirst.baselineContextTokens,
        baselineContextReferenceEvidence: "conservative-existing-planner-2048x24-or32-not-chosen-neural-shape",
        baselineModelNormallySpills: false, baselineActivityNormallySpills: false,
        residentModelWithHeadroomReserveBytes: cost(shape).residentModelWithHeadroomReserve,
        baselineActivityWithHeadroomReserveBytes: cost(shape).baselineActivityWithHeadroomReserve,
        baselineWorkspaceRuntimeReserveBytes: cost(shape).baselineWorkspaceRuntimeReserve,
        baselineKvAndIndexReserveBytes: cost(shape).baselineKvAndIndexReserve,
        baselineFastItemReserveBytes: cost(shape).baselineFastItemReserve,
        estimatedResidentMemoryItemBytes: cost(shape).residentItemBytesEstimate,
        boundedTrainingWithHeadroomReserveBytes: cost(shape).boundedTrainingWithHeadroomReserve,
        allocatorAndBurstHeadroomRatioPercent: 20,
        ramFirstBaselineEstimatedWorkingSetBytes: input.runtimeBaselineReserveBytes + cost(shape).residentModelWithHeadroomReserve
          + cost(shape).baselineActivityWithHeadroomReserve + cost(shape).boundedTrainingWithHeadroomReserve,
        ramFirstUnusedCeilingBytes: Math.max(0, input.selectedSystemRamBudgetBytes - input.runtimeBaselineReserveBytes
          - cost(shape).residentModelWithHeadroomReserve - cost(shape).baselineActivityWithHeadroomReserve - cost(shape).boundedTrainingWithHeadroomReserve),
        workingSetIsOsRssGuarantee: false, depthPolicy: "logarithmic-total-ram-working-set-lanes-not-quality-optimum",
        autoStoragePoolMayGrowForSelectedShape: true,
        pressureCompressionAndManualOrLargeWorkloadSpillAllowed: true,
      } : {}),
      acceleratorAvailable: input.acceleratorAvailable,
      computeEvidence: ramFirst ? "total-ram-resident-working-set-source-policy-not-neural-performance-measurement"
        : compute ? "bounded-current-packed-primitive-measured-not-neural-throughput" : "availability-and-tier-proxy-not-throughput-benchmark",
      ...(compute ? {
        primitiveKernel: compute.profile.kernel, primitiveKernelRevision: compute.profile.kernelRevision,
        primitiveMeasuredAt: compute.profile.measuredAt, primitiveRequestedDevice: compute.profile.requestedDevice,
        primitiveActualDevice: compute.profile.actualDevice, primitiveFallbackReason: compute.profile.fallbackReason,
        primitiveActivityDtype: compute.profile.activityDtype, primitiveIntegerResultDtype: compute.profile.integerResultDtype,
        intermediateAccumulationDtypeVerified: false,
        primitiveMacsPerRunExact: compute.profile.projectionMacsPerRun,
        primitiveMedianRunMicroseconds: compute.profile.medianRunMicroseconds,
        primitiveMacsPerSecondMeasured: compute.profile.projectionMacsPerSecond,
        mainSelectedPrimitiveWorkBudgetMicroseconds: compute.primitiveWorkBudgetMicroseconds,
        maximumTraversalMacsProxy: compute.maximumTraversalMacsProxy,
        selectedTraversalMacsProxy: cost(shape).traversalMacsProxy,
        traversalWorkBasis: "all-projection-entries-plus-one-router-traversal-proxy-not-actual-chat-path",
        wideShapeLatencyExtrapolationVerified: false, neuralTokensPerSecondMeasured: false,
        sourceCorpusTimeEstimated: false, qualityOptimizationVerified: false,
      } : {}),
      declaredCapacityShrinksWithLivePressure: false,
      tradeoff: ramFirst ? "resident-baseline-context-and-free-headroom-before-extra-width-not-quality-or-throughput-proof" : compute ? "explicit-primitive-work-proxy-versus-physical-capacity-not-a-neural-latency-or-quality-optimum" : "largest-physically-admitted-native-shape-not-a-throughput-or-quality-optimum",
      selectionMode: ramFirst ? "ram-first-headroom-default" : compute ? "balanced-measured-primitive-candidate" : "physical-capacity-candidate",
      isAutomaticDefault: Boolean(ramFirst)
    },
    qualityEvidence: "unmeasured-native-quality-deferred"
  });
  const scratch = Math.max(4 * MIB, 64 * Math.max(shape.dModel, shape.feedForward,
    shape.vsaDimensions, shape.routerNeurons, 1024, shape.imageSize ** 2 / 16,
    shape.audioSamples / 4) * 32);
  const inference = inventory.packedWeightBytes + inventory.staticNonweightTensorBytes + derivedControlBytes + 16 * 1024;
  return {
    format: "omni-ground-up-architecture-profile", formatVersion: 1,
    architecture: "OmniCortex", origin: "ground-up-random-initialization",
    externalPretrainedWeights: false, hardwareTier: tier,
    dModel: shape.dModel, layers: shape.layers, feedForward: shape.feedForward,
    vsaDimensions: shape.vsaDimensions, routerNeurons: shape.routerNeurons,
    workingMemoryItems: items, workspaceLatents: shape.workspaceLatents,
    exactLogicalParameterCount: inventory.logicalParameters,
    exactPackedProjectionParameterCount: inventory.projectionParameters,
    exactPackedTableParameterCount: inventory.tableParameters,
    exactPackedTernaryParameterCount: inventory.logicalParameters,
    packedTernaryWeightBytes: inventory.packedWeightBytes,
    packedWorkspaceTableBytes: shape.workspaceLatents * Math.ceil(shape.dModel / 4),
    packedMetaplasticityReserveBytes: inventory.resistanceBytes,
    fixedControlBufferReserveBytes: inventory.staticNonweightTensorBytes - inventory.resistanceBytes + derivedControlBytes + 16 * 1024,
    packedUpdateScratchBytes: scratch,
    residentInferenceStateBytes: inference, minimumTrainingStateBytes: inference + scratch,
    checkpointTensorBytes: checkpointReserve,
    parameterCountBasis: "architecture-logical-neural-elements",
    checkpointByteBasis: "packed-weights-plus-nonweight-state-reserve",
    nativeArchitecture: descriptor
  };
}
