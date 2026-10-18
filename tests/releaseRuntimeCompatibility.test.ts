import { spawnSync } from "node:child_process";
import { readFileSync } from "node:fs";
import { createRequire } from "node:module";
import { resolve } from "node:path";
import { pathToFileURL } from "node:url";
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
  it("delegates DMG construction once to the installed hash-pinned vendor with bounded ordinary detach retries", async () => {
    const wrapper = await import(pathToFileURL(resolve("scripts/dmgbuild-with-detach-retries.mjs")).href) as {
      pinnedDmgbuildToolset(architecture: string): { releaseName: string; filenameWithExt: string; checksums: Record<string, string> };
      runDmgbuild(args: string[], options: Record<string, unknown>): Promise<number>;
    };
    const vendorSource = readFileSync(createRequire(import.meta.url).resolve("dmg-builder"), "utf8");
    for (const architecture of ["x64", "arm64"]) {
      const toolset = wrapper.pinnedDmgbuildToolset(architecture);
      expect(vendorSource).toContain(toolset.releaseName);
      for (const [file, checksum] of Object.entries(toolset.checksums)) { expect(vendorSource).toContain(file); expect(vendorSource).toContain(checksum); }
    }
    const calls: Array<{ command: string; args: string[]; options: Record<string, unknown> }> = [];
    const args = ["-s", resolve("work/owned-settings.json"), "Owned fixture", resolve("work/owned-output.dmg")];
    const result = await wrapper.runDmgbuild(args, { platform: "darwin", architecture: "x64",
      download: async (value: unknown) => { expect(value).toEqual(wrapper.pinnedDmgbuildToolset("x64")); return resolve("work/owned-vendor"); },
      execute: (command: string, invocation: string[], options: Record<string, unknown>) => {
        calls.push({ command, args: invocation, options });
        return { status: 0, stdout: invocation[0] === "--help" ? "--detach-retries {1,...,30}" : "" };
      } });
    expect(result).toBe(0); expect(calls).toHaveLength(2);
    expect(calls[0]!.args).toEqual(["--help"]); expect(calls[0]!.options.timeout).toBe(10_000);
    expect(calls[1]!.args).toEqual(["--detach-retries", "30", ...args]);
    expect(calls[1]!.options).toMatchObject({ stdio: "inherit", shell: false });
    expect(calls.every(call => call.command === resolve("work/owned-vendor/dmgbuild"))).toBe(true);
    const source = read("scripts/package-posix.sh");
    expect(source).toContain('if [[ -z "${CUSTOM_DMGBUILD_PATH:-}" ]]');
    expect(source).toContain('export CUSTOM_DMGBUILD_PATH="${REPO_ROOT}/scripts/dmgbuild-with-detach-retries.mjs"');
    expect(source).toContain('--config.mac.notarize=true');
    expect(source).toContain('npx electron-builder "${BUILDER_ARGS[@]}"');
  });

  it("keeps a failed DMG fatal and refuses unsupported detach controls without rebuilding or unmounting devices", async () => {
    const wrapper = await import(pathToFileURL(resolve("scripts/dmgbuild-with-detach-retries.mjs")).href) as {
      runDmgbuild(args: string[], options: Record<string, unknown>): Promise<number>;
    };
    const args = ["-s", resolve("work/owned-settings.json"), "Owned fixture", resolve("work/owned-output.dmg")];
    let images = 0, help = "--detach-retries";
    const options = { platform: "darwin", architecture: "arm64", download: async () => resolve("work/owned-vendor"),
      execute: (_command: string, invocation: string[]) => invocation[0] === "--help" ? { status: 0, stdout: help } : (images++, { status: 16 }) };
    expect(await wrapper.runDmgbuild(args, options)).toBe(16); expect(images).toBe(1);
    help = "no such option";
    await expect(wrapper.runDmgbuild(args, options)).rejects.toThrow("does not advertise"); expect(images).toBe(1);
    await expect(wrapper.runDmgbuild(args, { ...options, platform: "linux" })).rejects.toThrow("requires macOS");
    await expect(wrapper.runDmgbuild(["--detach-retries", "999", ...args], options)).rejects.toThrow("Unexpected");
    expect(images).toBe(1);
    const source = read("scripts/dmgbuild-with-detach-retries.mjs");
    expect(source).not.toContain('"hdiutil"'); expect(source).not.toContain('"diskutil"');
    expect(source).not.toContain('"-force"');
  });

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
    expect(requirements).not.toMatch(/^transformers==/mu);
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
    expect(verification.stdout).toContain("Verified 5 engine locks (159 target package pins).");

    for (const [target, packageCount] of Object.entries({
      "linux-aarch64": 31,
      "linux-x86_64": 31,
      "macos-arm64": 32,
      "macos-x86_64": 32,
      "windows-x86_64": 33
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
      for (const module of ["jsonschema", "jsonschema_specifications", "referencing", "attrs", "rpds"]) {
        expect(build).toContain(`--collect-all ${module}`);
      }
    }
  });

  it("preserves every schema dependency license in the canonical legal payload", () => {
    const notices = read("THIRD_PARTY_NOTICES.md");
    for (const [name, copyright] of Object.entries({
      attrs: "Hynek Schlawack and the attrs contributors",
      jsonschema: "2013 Julian Berman",
      "jsonschema-specifications": "2022 Julian Berman",
      referencing: "2022 Julian Berman",
      "rpds-py": "2023 Julian Berman"
    })) {
      const license = `licenses/${name}-MIT.txt`;
      expect(notices).toContain(license);
      expect(notices).toContain(`https://pypi.org/project/${name}/`);
      expect(read(license)).toContain(copyright);
      expect(read(license)).toContain("Permission is hereby granted");
      expect(read(license)).toContain("THE SOFTWARE IS PROVIDED");
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
      expect(job).toContain("torch.isfinite(torch.empty(1).fill_(1))");
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
