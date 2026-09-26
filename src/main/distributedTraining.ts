import { readFile } from "node:fs/promises";
import { dirname, join, resolve } from "node:path";

export interface DistributedTrainingEnvironment {
  detected: boolean;
  worldSize: number;
  rank: number;
  localRank: number;
  localWorldSize: number;
  masterAddress?: string;
  masterPort?: number;
}

export interface DistributedTrainingLaunchRequest {
  python: string;
  projectRoot: string;
  datasetPath: string;
  outputPath: string;
  runPath?: string;
  processes?: number | "gpu";
  epochs?: number;
  profile?: "micro" | "personal" | "gpu" | "workstation";
  strategy?: "auto" | "ddp" | "fsdp";
  amp?: "auto" | "off" | "fp16" | "bf16";
  replaceOutput?: boolean;
}

export interface DistributedTrainingInvocation {
  command: string;
  args: string[];
  cwd: string;
  runPath: string;
}

const integer = (
  environment: NodeJS.ProcessEnv,
  name: string,
  fallback: number,
): number => {
  const raw = environment[name];
  if (raw === undefined || raw.trim() === "") return fallback;
  const value = Number(raw);
  if (!Number.isSafeInteger(value)) throw new Error(`${name} must be an integer`);
  return value;
};

export const detectDistributedTrainingEnvironment = (
  environment: NodeJS.ProcessEnv = process.env,
): DistributedTrainingEnvironment => {
  const worldSize = integer(environment, "WORLD_SIZE", 1);
  const rank = integer(environment, "RANK", 0);
  const localRank = integer(environment, "LOCAL_RANK", 0);
  const localWorldSize = integer(environment, "LOCAL_WORLD_SIZE", worldSize);
  const masterPort = integer(environment, "MASTER_PORT", 0);
  if (worldSize < 1) throw new Error("WORLD_SIZE must be positive");
  if (rank < 0 || rank >= worldSize) throw new Error("RANK must be within WORLD_SIZE");
  if (localWorldSize < 1 || localRank < 0 || localRank >= localWorldSize) {
    throw new Error("LOCAL_RANK must be within LOCAL_WORLD_SIZE");
  }
  if (masterPort < 0 || masterPort > 65_535) throw new Error("MASTER_PORT is invalid");
  return {
    detected:
      worldSize > 1 ||
      environment.TORCHELASTIC_RUN_ID !== undefined ||
      environment.LOCAL_RANK !== undefined,
    worldSize,
    rank,
    localRank,
    localWorldSize,
    ...(environment.MASTER_ADDR?.trim()
      ? { masterAddress: environment.MASTER_ADDR.trim() }
      : {}),
    ...(masterPort ? { masterPort } : {}),
  };
};

export const defaultDistributedRunPath = (outputPath: string): string => {
  const target = resolve(outputPath);
  const name = target.split(/[\\/]/).pop() || "omni-brain";
  return join(dirname(target), `.${name}.distributed-run`);
};

export const buildDistributedTrainingInvocation = (
  request: DistributedTrainingLaunchRequest,
): DistributedTrainingInvocation => {
  const projectRoot = resolve(request.projectRoot);
  const outputPath = resolve(request.outputPath);
  const runPath = resolve(request.runPath || defaultDistributedRunPath(outputPath));
  const processes = request.processes ?? "gpu";
  if (typeof processes === "number" && (!Number.isSafeInteger(processes) || processes < 1)) {
    throw new Error("distributed process count must be a positive integer");
  }
  const args = [
    "-m",
    "torch.distributed.run",
    "--standalone",
    "--nproc-per-node",
    String(processes),
    join(projectRoot, "engine", "distributed_train.py"),
    "train",
    "--dataset",
    resolve(request.datasetPath),
    "--output",
    outputPath,
    "--run-dir",
    runPath,
    "--epochs",
    String(request.epochs ?? 1),
    "--profile",
    request.profile ?? "gpu",
    "--strategy",
    request.strategy ?? "auto",
    "--amp",
    request.amp ?? "auto",
  ];
  if (request.replaceOutput) args.push("--replace-output");
  return { command: request.python, args, cwd: projectRoot, runPath };
};

export const readDistributedTrainingStatus = async (
  runPath: string,
): Promise<Record<string, unknown>> => {
  try {
    const text = await readFile(join(resolve(runPath), "distributed-status.json"), "utf8");
    const value: unknown = JSON.parse(text);
    if (
      !value ||
      typeof value !== "object" ||
      (value as { format?: unknown }).format !== "omni-distributed-training-status" ||
      (value as { formatVersion?: unknown }).formatVersion !== 1
    ) {
      throw new Error("distributed training status format is invalid");
    }
    return value as Record<string, unknown>;
  } catch (error) {
    if ((error as NodeJS.ErrnoException).code === "ENOENT") {
      return { state: "not-started", environment: detectDistributedTrainingEnvironment() };
    }
    throw error;
  }
};
