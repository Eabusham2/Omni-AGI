import { existsSync } from "node:fs";
import { spawnSync } from "node:child_process";

const requested = process.env.OMNI_PYTHON;
const candidates = [
  requested,
  process.platform === "win32" ? "python" : "/usr/bin/python3",
  "python3",
  "python"
].filter(Boolean);

let selected;
for (const candidate of candidates) {
  if (candidate.includes("/") && !existsSync(candidate)) continue;
  const probe = spawnSync(candidate, ["-c", "import torch; print(torch.__version__)"], {
    encoding: "utf8",
    shell: false
  });
  if (probe.status === 0) {
    selected = candidate;
    break;
  }
}

if (!selected) {
  console.error(
    "No Python runtime with PyTorch was found. Set OMNI_PYTHON or install engine/requirements.txt."
  );
  process.exit(1);
}

console.log(`Running OmniCortex tests with ${selected}`);
const result = spawnSync(
  selected,
  ["-m", "unittest", "discover", "-s", "engine/tests", "-p", "test_*.py", "-v"],
  {
    stdio: "inherit",
    shell: false,
    env: { ...process.env, PYTHONPATH: "engine" }
  }
);
process.exit(result.status ?? 1);
