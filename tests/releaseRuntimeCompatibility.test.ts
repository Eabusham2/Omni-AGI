import { spawnSync } from "node:child_process";
import { readFileSync } from "node:fs";
import { resolve } from "node:path";
import { describe, expect, it } from "vitest";

const read = (path: string): string => readFileSync(resolve(path), "utf8");

function workflowJob(source: string, jobName: string): string {
  const lines = source.split(/\r?\n/u);
  const start = lines.findIndex((line) => line === `  ${jobName}:`);
  expect(start).toBeGreaterThanOrEqual(0);
  let end = start + 1;
  while (end < lines.length && !/^  [A-Za-z0-9_-]+:\s*$/u.test(lines[end] ?? "")) end += 1;
  return lines.slice(start, end).join("\n");
}

function workflowStep(job: string, stepName: string): string {
  const lines = job.split(/\r?\n/u);
  const start = lines.findIndex((line) => line === `      - name: ${stepName}`);
  expect(start).toBeGreaterThanOrEqual(0);
  let end = start + 1;
  while (end < lines.length && !(lines[end] ?? "").startsWith("      - ")) end += 1;
  return lines.slice(start, end).join("\n");
}

describe("native release Python compatibility", () => {
  it("keeps every top-level runtime dependency exact and preserves Intel fallbacks", () => {
    const requirements = read("engine/requirements.txt");
    for (const line of requirements.split(/\r?\n/u)) {
      const requirement = line.trim();
      if (!requirement || requirement.startsWith("#")) continue;
      expect(requirement).toMatch(/^[A-Za-z0-9][A-Za-z0-9_.-]*==[^;\s]+(?:; .+)?$/u);
    }
    expect(requirements).toContain(
      'torch==2.2.2; platform_system == "Darwin" and platform_machine == "x86_64"'
    );
    expect(requirements).toContain(
      'torch==2.10.0; platform_system != "Darwin"'
    );
    expect(requirements).toContain(
      'transformers==4.57.6; platform_system == "Darwin" and platform_machine == "x86_64"'
    );
    expect(requirements).toContain(
      'transformers==5.15.0; platform_system != "Darwin" or platform_machine != "x86_64"'
    );
    expect(requirements).toContain(
      'pyarrow==17.0.0; platform_system == "Darwin" and platform_machine == "x86_64"'
    );
  });

  it("ships a complete hash-locked Python 3.11 wheel closure for every worker target", () => {
    const python =
      process.env.OMNI_BUILD_PYTHON ?? (process.platform === "win32" ? "python" : "python3");
    const verification = spawnSync(
      python,
      ["scripts/install-engine-lock.py", "--verify-locks"],
      { cwd: resolve("."), encoding: "utf8" }
    );
    expect(verification.status, verification.stderr || verification.stdout).toBe(0);
    expect(verification.stdout).toContain("Verified 5 engine locks (236 target package pins).");

    for (const [target, packageCount] of Object.entries({
      "linux-aarch64": 48,
      "linux-x86_64": 48,
      "macos-arm64": 49,
      "macos-x86_64": 40,
      "windows-x86_64": 51
    })) {
      const lock = read(`engine/locks/${target}-py311.lock`);
      expect(lock).toContain("--require-hashes");
      expect(lock).toContain("--only-binary=:all:");
      expect(lock.match(/^[-a-z0-9]+==[^\s]+ --hash=sha256:[0-9a-f]{64}$/gmu)).toHaveLength(
        packageCount
      );
      expect(lock).toContain("pyinstaller==6.21.0");
    }

    expect(read("engine/locks/macos-x86_64-py311.lock")).toContain("pyarrow==17.0.0");
    expect(read("engine/locks/windows-x86_64-py311.lock")).toContain(
      "torch==2.10.0+cpu"
    );
  });

  it("installs or verifies the selected lock before PyInstaller runs", () => {
    const posix = read("scripts/build-engine-posix.sh");
    const windows = read("scripts/build-engine.ps1");
    for (const build of [posix, windows]) {
      expect(build).toContain("install-engine-lock.py");
      expect(build).toContain("--verify-only");
      expect(build).not.toContain("pyinstaller>=");
    }
  });

  it("pins and reuses the selected lock in every desktop CI and release job", () => {
    const contracts = [
      {
        path: ".github/workflows/windows.yml",
        jobName: "test-and-package",
        pythonVersion: "3.11.9",
        lockedSteps: [
          "Build self-contained brain worker",
          "Package unsigned signed-ready Windows installer and archive"
        ]
      },
      {
        path: ".github/workflows/macos.yml",
        jobName: "test-and-package",
        pythonVersion: "3.11.9",
        lockedSteps: ["Package native macOS application"]
      },
      {
        path: ".github/workflows/linux.yml",
        jobName: "test-and-package",
        pythonVersion: "3.11.16",
        lockedSteps: ["Package native Linux application"]
      },
      {
        path: ".github/workflows/release.yml",
        jobName: "windows",
        pythonVersion: "3.11.9",
        lockedSteps: ["Package Windows"]
      },
      {
        path: ".github/workflows/release.yml",
        jobName: "macos",
        pythonVersion: "3.11.9",
        lockedSteps: ["Package, sign, and notarize macOS when credentials exist"]
      },
      {
        path: ".github/workflows/release.yml",
        jobName: "linux",
        pythonVersion: "3.11.16",
        lockedSteps: ["Package Linux"]
      }
    ];

    for (const { path, jobName, pythonVersion, lockedSteps } of contracts) {
      const job = workflowJob(read(path), jobName);
      expect(job.match(/python-version:/gu)).toHaveLength(1);
      expect(job).toContain(`python-version: "${pythonVersion}"`);
      expect(job).toContain("cache-dependency-path: |");
      expect(job).toContain("engine/requirements.txt");
      expect(job).toContain("engine/requirements-build.txt");
      expect(job).toContain("engine/locks/*.lock");
      expect(job.match(/python scripts\/install-engine-lock\.py/gu)).toHaveLength(1);
      expect(job).toContain("python -m pip check");
      expect(job).toContain("transformers.is_torch_available()");
      expect(job).not.toContain("python -m pip install -r engine/requirements.txt");
      expect(job).not.toContain("--index-url https://download.pytorch.org");
      expect(job).not.toContain("matrix.torch_version");
      expect(job).not.toContain("matrix.numpy_constraint");
      for (const stepName of lockedSteps) {
        expect(workflowStep(job, stepName)).toContain(
          'OMNI_SKIP_BUILD_DEPENDENCY_INSTALL: "1"'
        );
      }
    }

    const windowsRelease = workflowJob(read(".github/workflows/release.yml"), "windows");
    expect(windowsRelease).toContain("runner: windows-11-arm");
    expect(windowsRelease).toContain("node_arch: arm64");
    expect(windowsRelease).toContain("python_arch: x64");
    expect(windowsRelease).toContain("worker_arch: x64");
  });
});
