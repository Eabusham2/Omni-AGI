import rawArchitecture from "../../architecture/omnicortex-ground-up-v1.json";
import type { HardwareTier, WorkingMemoryMode } from "../shared/types";

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
