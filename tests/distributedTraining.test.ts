import { tmpdir } from "node:os";
import { join } from "node:path";
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
    const projectRoot = join(tmpdir(), "Omni AGI");
    const datasetPath = join(tmpdir(), "datasets", "full folder");
    const outputPath = join(tmpdir(), "brains", "rented run");
    const launch = buildDistributedTrainingInvocation({
      python: "python",
      projectRoot,
      datasetPath,
      outputPath,
      processes: 8,
      epochs: 3,
      strategy: "auto",
      amp: "bf16",
    });
    expect(launch.command).toBe("python");
    expect(launch.cwd).toBe(projectRoot);
    expect(launch.args).toContain(datasetPath);
    expect(launch.args).toContain(outputPath);
    expect(launch.args).toContain("8");
    expect(launch.args).not.toContain("pretrained");
    expect(launch.args).not.toContain("rlhf");
  });
});
