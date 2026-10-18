import { spawnSync } from "node:child_process";
import { createHash } from "node:crypto";
import { mkdtemp, readFile, rm } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join, resolve } from "node:path";
import { describe, expect, it } from "vitest";

const python = process.env.OMNI_BUILD_PYTHON ?? (process.platform === "win32" ? "python" : "python3");
const builder = resolve("scripts/build-native-kernels.py");

describe("native worker kernel packaging", () => {
  it("targets the Python process on an ARM64 Windows host and replaces a mismatched developer environment", () => {
    const result = spawnSync(python, ["-I", "-c", `
import importlib.util, os, pathlib, subprocess, sys, tempfile
from unittest.mock import patch
spec = importlib.util.spec_from_file_location("omni_kernel_builder", sys.argv[1])
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
with patch.object(module.sys, "platform", "win32"), patch.object(module.sysconfig, "get_platform", return_value="win-amd64"), patch.object(module.platform, "machine", return_value="ARM64"):
    assert module.runtime_architecture() == "x86_64"
with tempfile.TemporaryDirectory(prefix="omni-vs-fixture-") as temporary:
    root = pathlib.Path(temporary)
    locator = root / "Microsoft Visual Studio/Installer/vswhere.exe"
    locator.parent.mkdir(parents=True)
    locator.touch()
    installation = root / "Visual Studio"
    developer = installation / "Common7/Tools/VsDevCmd.bat"
    developer.parent.mkdir(parents=True)
    developer.touch()
    replies = [subprocess.CompletedProcess([], 0, str(installation) + "\\n", ""),
        subprocess.CompletedProcess([], 0, "Path=selected-x64-tools\\r\\nVSCMD_ARG_TGT_ARCH=x64\\r\\nPRIVATE_SIGNING_VALUE=fixture-only\\r\\n", "")]
    with patch.dict(os.environ, {"ProgramFiles(x86)": str(root), "PATH": "wrong-arm64-tools", "VSCMD_ARG_TGT_ARCH": "arm64"}, clear=True), patch.object(module.shutil, "which", return_value="selected-cl.exe"), patch.object(module.subprocess, "run", side_effect=replies) as run:
        environment = module.windows_msvc_environment("x86_64")
        assert environment["PATH"] == "selected-x64-tools"
        assert environment["VSCMD_ARG_TGT_ARCH"] == "x64"
        assert "OMNI_KERNEL_VSDEVCMD" not in environment
        assert len(run.call_args_list) == 2
        assert "-arch=x64" in run.call_args_list[1].args[0][-1]
        assert run.call_args_list[1].kwargs["capture_output"] is True
        assert run.call_args_list[1].kwargs["encoding"] == "utf-16-le"
assert "torch" not in sys.modules
print("x64 interpreter target retained; private developer environment selected")
`, builder], { cwd: resolve("."), encoding: "utf8", timeout: 10_000 });
    expect(result.status, result.stderr || result.stdout).toBe(0);
    expect(result.stdout.trim()).toBe("x64 interpreter target retained; private developer environment selected");
    expect(result.stdout).not.toContain("PRIVATE_SIGNING_VALUE");
    expect(result.stdout).not.toContain("fixture-only");
  });

  it("builds a standalone library and validates its loader ABI before publishing the hashed artifact", async () => {
    const directory = await mkdtemp(join(tmpdir(), "omni-kernel-package-"));
    try {
      const result = spawnSync(python, [builder, "--output-directory", directory], {
        cwd: resolve("."), encoding: "utf8", timeout: 150_000
      });
      expect(result.status, result.stderr || result.stdout).toBe(0);
      const manifest = JSON.parse(await readFile(join(directory, "manifest.json"), "utf8"));
      expect(manifest).toMatchObject({ abi: 1, platform: process.platform });
      expect(["x86_64", "arm64"]).toContain(manifest.architecture);
      expect(manifest.sourceSha256).toBe(createHash("sha256").update(await readFile(resolve("engine/omni_core/kernels/weighted_ternary.cpp"))).digest("hex"));
      expect(manifest.sha256).toBe(createHash("sha256").update(await readFile(join(directory, manifest.file))).digest("hex"));
    } finally {
      await rm(directory, { recursive: true, force: true });
    }
  }, 150_000);
});
