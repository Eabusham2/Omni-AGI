import { describe, expect, it } from "vitest";

import {
  buildDistributedTrainingInvocation,
  detectDistributedTrainingEnvironment,
} from "../src/main/distributedTraining";

describe("distributed training launch configuration", () => {
  it("detects torchrun rank identity without silently repairing bad values", () => {
    expect(
      detectDistributedTrainingEnvironment({
        WORLD_SIZE: "4",
        RANK: "2",
        LOCAL_RANK: "0",
        LOCAL_WORLD_SIZE: "2",
        MASTER_ADDR: "10.0.0.5",
        MASTER_PORT: "29400",
      }),
    ).toEqual({
      detected: true,
      worldSize: 4,
      rank: 2,
      localRank: 0,
      localWorldSize: 2,
      masterAddress: "10.0.0.5",
      masterPort: 29400,
    });
    expect(() =>
      detectDistributedTrainingEnvironment({ WORLD_SIZE: "2", RANK: "2" }),
    ).toThrow(/RANK must be within WORLD_SIZE/);
  });

  it("builds an argument array without a shell and preserves paths as values", () => {
    const launch = buildDistributedTrainingInvocation({
      python: "python",
      projectRoot: "/workspace/Omni AGI",
      datasetPath: "/datasets/full folder",
      outputPath: "/brains/rented run",
      processes: 8,
      epochs: 3,
      strategy: "auto",
      amp: "bf16",
    });
    expect(launch.command).toBe("python");
    expect(launch.args).toContain("/datasets/full folder");
    expect(launch.args).toContain("/brains/rented run");
    expect(launch.args).toContain("8");
    expect(launch.args).not.toContain("pretrained");
    expect(launch.args).not.toContain("rlhf");
  });
});
