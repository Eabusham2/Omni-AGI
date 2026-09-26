import { readFileSync } from "node:fs";
import { resolve } from "node:path";
import { describe, expect, it } from "vitest";

const root = resolve(import.meta.dirname, "..");

function read(relativePath: string): string {
  return readFileSync(resolve(root, relativePath), "utf8");
}

function readJson<T>(relativePath: string): T {
  return JSON.parse(read(relativePath)) as T;
}

function parseStableVersion(version: string): [number, number, number] {
  const parts = version.split(".");
  if (
    parts.length !== 3 ||
    parts.some((part) => !/^(0|[1-9]\d*)$/u.test(part))
  ) {
    throw new Error(`Expected a stable MAJOR.MINOR.PATCH version, received ${version}.`);
  }
  const major = Number(parts[0]);
  const minor = Number(parts[1]);
  const patch = Number(parts[2]);
  if (
    ![major, minor, patch].every(Number.isSafeInteger) ||
    minor >= 100 ||
    patch >= 100
  ) {
    throw new Error(`Version cannot map to a stable mobile build number: ${version}.`);
  }
  return [major, minor, patch];
}

describe("project integrity", () => {
  it("keeps the application independent from hosted or third-party chat runtimes", () => {
    const packageDocument = readJson<{
      dependencies: Record<string, string>;
      devDependencies: Record<string, string>;
    }>("package.json");
    const dependencies = {
      ...packageDocument.dependencies,
      ...packageDocument.devDependencies
    };

    expect(Object.keys(dependencies)).not.toContain("openai");
    expect(Object.keys(dependencies)).not.toContain("@anthropic-ai/sdk");
    expect(Object.keys(dependencies)).not.toContain("ollama");
    expect(read("src/shared/types.ts")).toContain(
      'export type InferenceRuntime = "adaptive-core";'
    );
    expect(read("docs/ARCHITECTURE.md")).toContain(
      "External models are not used to create or answer for a new brain."
    );
  });

  it("ships the noncommercial terms separately from commercial permission", () => {
    const packageDocument = readJson<{
      license: string;
      build: { extraResources: Array<{ from: string; to: string }> };
    }>("package.json");
    const noncommercial = read("LICENSE.md");
    const commercial = read("COMMERCIAL_LICENSE.md");
    const notices = read("THIRD_PARTY_NOTICES.md");

    expect(packageDocument.license).toBe("SEE LICENSE IN LICENSE.md");
    expect(noncommercial).toContain("PolyForm Noncommercial License 1.0.0");
    expect(noncommercial).toContain("Noncommercial Purpose");
    expect(commercial).toContain("Commercial and for-profit use");
    expect(notices).toContain("BitNet");
    expect(notices).toContain("snnTorch");
    expect(notices).toContain("NCPS");
    expect(notices).toContain("imageio-ffmpeg");
    expect(notices).toContain("GPL-2.0-or-later");
    expect(notices).toContain("corresponding-source");
    expect(read("licenses/imageio-ffmpeg-BSD-2-Clause.txt")).toContain(
      "Copyright (c) 2019-2025, imageio"
    );
    expect(read("licenses/GPL-2.0.txt")).toContain(
      "GNU GENERAL PUBLIC LICENSE"
    );
    expect(packageDocument.build.extraResources).toContainEqual({
      from: "THIRD_PARTY_NOTICES.md",
      to: "licenses/THIRD_PARTY_NOTICES.md"
    });
    expect(packageDocument.build.extraResources).toContainEqual({
      from: "licenses/imageio-ffmpeg-BSD-2-Clause.txt",
      to: "licenses/imageio-ffmpeg-BSD-2-Clause.txt"
    });
    expect(packageDocument.build.extraResources).toContainEqual({
      from: "licenses/GPL-2.0.txt",
      to: "licenses/GPL-2.0.txt"
    });
  });

  it("keeps portable application resources free of local Python test caches", () => {
    const packageDocument = readJson<{
      build: {
        extraResources: Array<{ from: string; filter?: string[] }>;
      };
    }>("package.json");
    const engineResource = packageDocument.build.extraResources.find(
      (resource) => resource.from === "engine"
    );

    expect(engineResource?.filter).toEqual(
      expect.arrayContaining([
        "!tests{,/**/*}",
        "!**/__pycache__{,/**/*}",
        "!**/.pytest_cache{,/**/*}",
        "!**/.mypy_cache{,/**/*}",
        "!**/.ruff_cache{,/**/*}",
        "!**/*.pyc",
        "!**/*.pyo"
      ])
    );
  });

  it("requires provenance and license labels for every catalog entry", () => {
    const catalog = readJson<{
      schemaVersion: number;
      entries: Array<{
        id: string;
        kind: string;
        sourceUrl: string;
        license: string;
      }>;
    }>("catalog/catalog.json");

    expect(catalog.schemaVersion).toBe(1);
    expect(catalog.entries.length).toBeGreaterThanOrEqual(4);
    for (const entry of catalog.entries) {
      expect(entry.id).toMatch(/^[a-z0-9-]+$/);
      expect(["brain", "recipe", "dataset", "modality-pack"]).toContain(entry.kind);
      expect(entry.sourceUrl.trim().length).toBeGreaterThan(0);
      expect(entry.license.trim().length).toBeGreaterThan(0);
    }
  });

  it("defines structured protocols and explicit permission defaults for every tool", () => {
    const catalog = readJson<{
      schemaVersion: number;
      tools: Array<{
        id: string;
        defaultGrant: string;
        actions: Record<string, { input: object; output: object }>;
      }>;
    }>("tools/catalog.json");

    expect(catalog.schemaVersion).toBe(1);
    expect(catalog.tools.map((tool) => tool.id)).toEqual(
      expect.arrayContaining([
        "system.files",
        "system.shell",
        "code.execute",
        "web.search",
        "web.fetch",
        "browser.automation",
        "device.input",
        "modality.imagine",
        "studio.ui",
        "brain.history",
        "agent.fork",
        "source.self-modify"
      ])
    );
    for (const tool of catalog.tools) {
      expect(["off", "ask", "auto", "full"]).toContain(tool.defaultGrant);
      expect(Object.keys(tool.actions).length).toBeGreaterThan(0);
      for (const action of Object.values(tool.actions)) {
        expect(action.input).toBeTypeOf("object");
        expect(action.output).toBeTypeOf("object");
      }
    }
  });

  it("keeps stable Build automatic, removable, and free of personality controls", () => {
    const renderer = read("src/renderer/src/App.tsx");
    const preload = read("src/preload/index.ts");
    const ipc = read("src/main/ipc.ts");
    const repository = read("src/main/brainRepository.ts");
    const service = read("src/main/brainService.ts");

    // Every range input is a device-capacity control: Build context, RAM, and
    // shared storage, then the same three controls in Device settings.
    // Personality, curiosity, noise, and plasticity are not user sliders.
    expect(renderer.match(/type="range"/g)).toHaveLength(6);
    expect(renderer).toContain('className="memory-capacity-slider"');
    expect(renderer).toContain('aria-label="Omni share of safely available memory"');
    expect(renderer).toContain('aria-label="Advanced Omni RAM cap percentage"');
    expect(renderer).not.toContain("Internal freedom");
    expect(renderer).not.toContain("curiosityDrive");
    expect(renderer).not.toContain("crawlConcurrency");
    expect(renderer).toContain('imagination: "auto"');
    expect(renderer).toContain("Remove only changes this build list");
    expect(renderer).toContain("window.omni?.data.discardBuildResource");
    expect(renderer).toContain("window.omni.brain.duplicate");
    expect(renderer).toContain("Your instances");
    expect(renderer).toContain("STORED RECOVERY ORIGIN · NOT A RUNNING BRAIN");
    expect(renderer).toContain("Confirmation {stage} of 3");
    expect(renderer).toContain("PERMANENTLY DELETE ${target.name}");
    expect(renderer).toContain("Duplicate instance");
    expect(renderer).not.toContain("const first = starters[0]");
    expect(preload).toContain("selectBuildResources");
    expect(preload).toContain("startBuildResource");
    expect(ipc).toContain("buildSelections.delete");
    expect(repository).toContain("copy-on-write neural storage");
    expect(repository).toContain("deduplicateImmutableOrigin");
    expect(repository).toContain("permanentlyDeleteInstance");
    expect(repository).toContain("collectUnreferencedBlobs");
    expect(service).toContain("deduplicateImmutableOrigin(brain.id)");
  });

  it("exposes ordered mid-turn neural streaming without prose action parsing", () => {
    const types = read("src/shared/types.ts");
    const channels = read("src/shared/ipc.ts");
    const preload = read("src/preload/index.ts");
    const ipc = read("src/main/ipc.ts");
    const controller = read("src/main/chatActionController.ts");
    const protocol = read("docs/STREAMING_PROTOCOL.md");

    expect(types).toContain('type: "chat-token"');
    expect(types).toContain('type: "modality-preview"');
    expect(types).toContain('type: "chat-phase"');
    expect(types).toContain('phase: "reply-complete-learning"');
    expect(types).toContain('"action-result-learning"');
    expect(types).toContain("onStream(listener:");
    expect(channels).toContain('streamEvent: "omni:chat:stream-event"');
    expect(channels).toContain('cancel: "omni:chat:cancel"');
    expect(preload).toContain("onStream: (listener)");
    expect(preload).toContain("IPC.chat.streamEvent");
    expect(ipc).toContain('actions.on("stream", streamListener)');
    expect(controller).not.toContain("parseHumanAction");
    expect(controller).not.toContain("parseModelActions");
    expect(protocol).toContain("monotonically increasing");
    expect(protocol).toContain("never parsed into actions");
  });

  it("keeps voice delivery and steering typed, bounded, and presentation-only", () => {
    const types = read("src/shared/types.ts");
    const voiceTypes = read("src/shared/liveVoice.ts");
    const voice = read("src/renderer/src/liveVoiceController.ts");
    const renderer = read("src/renderer/src/App.tsx");
    const controller = read("src/main/chatActionController.ts");

    expect(voiceTypes).toContain('LiveVoiceDeliveryMode = "live" | "buffered"');
    expect(voiceTypes).toContain('LiveVoicePace = "slow" | "normal" | "fast"');
    expect(voice).toContain("maxLiveSpeechQueue");
    expect(voice).toContain("skippedCharacters");
    expect(voice).toContain("LIVE_VOICE_PACE_RATES");
    expect(renderer).toContain("Neural listening");
    expect(renderer).toContain("Neural voice");
    expect(renderer).toContain("platform STT + TTS");
    expect(types).toContain('kind: "steer"');
    expect(types).toContain("replacesTurnId");
    expect(controller).toContain("turnMetadata");
    expect(renderer).toContain("Ctrl/Cmd Enter steers");
    expect(renderer).toContain('kind: "steer"');
    expect(renderer).toContain('className="composer__steer-choice"');
    expect(renderer).toContain('className="send-button composer__queue-choice"');
    expect(renderer).toContain("the prior request remains visible and the loaded mind stays warm");
    expect(renderer).not.toContain("[Steer]");
  });

  it("keeps working-memory, imagination, and dataset status evidence-bounded", () => {
    const renderer = read("src/renderer/src/App.tsx");
    const readme = read("README.md");
    const audit = read("docs/COMPLETION_AUDIT.md");
    const compliance = read("docs/ORIGINAL_REQUEST_COMPLIANCE.md");

    expect(renderer).not.toContain("activity={0.82}");
    expect(renderer).not.toContain("Everything outside learned parameters");
    expect(renderer).not.toContain("Default response budget");
    expect(renderer).not.toContain("Learned sources");
    expect(renderer).toContain("Baseline response budget");
    expect(renderer).toContain("adjusted per turn from neural state");
    expect(renderer).toMatch(/textTurnVisiblyActive\s*\?\s*"Forming response"/);
    expect(renderer).not.toContain("Pondering turn");
    expect(renderer).toContain("Idle / ready");
    expect(renderer).not.toContain("<i /> pondering");
    expect(renderer).not.toContain("Following a quieter association");
    expect(renderer).toMatch(/cancellingTurn\s*\?\s*"Stopping"/);
    expect(renderer).toMatch(/chatQueue\s*\?\s*"Queued"/);
    expect(renderer).toMatch(/pondering\s*\?\s*"Pondering"/);
    expect(renderer).toContain(': "Forming a response…"');
    expect(renderer).toContain("Encoded source ledger");
    expect(renderer).toContain("incomplete/resumable");
    expect(renderer).toContain("Progressive imagination");
    expect(readme).toContain(
      "Compatible imported checkpoints retain their recorded context/model shape"
    );
    expect(readme).toContain(
      "native architecture's exact parameter inventory directly from the tier and selected recurrent/paged working-memory population"
    );
    expect(readme).toContain(
      "Token context and recurrent memory items are separate"
    );
    expect(readme).toContain("neither frame-synchronous");
    expect(compliance).toContain("committed deterministic manifest/source snapshot");
    expect(compliance).toContain("traversal accounting does not claim");
    expect(audit).not.toContain("passed 103 tests");
    expect(audit).not.toContain("passed 106/106");
    expect(audit).toContain("final merged commit");
  });

  it("keeps both Windows architectures in continuous packaging coverage", () => {
    const workflow = read(".github/workflows/windows.yml");
    const packageDocument = readJson<{
      scripts: Record<string, string>;
      devDependencies: Record<string, string>;
      build: { win: { target: string[] }; nsis: { useZip: boolean } };
    }>("package.json");
    const lockDocument = readJson<{
      packages: Record<string, { version?: string }>;
    }>("package-lock.json");

    expect(workflow).toContain('runs-on: ${{ matrix.runner }}');
    expect(workflow).toContain("arch: x64");
    expect(workflow).toContain("runner: windows-latest");
    expect(workflow).toContain("arch: arm64");
    expect(workflow).toContain("runner: windows-11-arm");
    expect(workflow).toContain("node_arch: arm64");
    expect(workflow).toContain("python_arch: x64");
    expect(workflow).toContain("worker_arch: x64");
    expect(workflow).toContain('architecture: ${{ matrix.node_arch }}');
    expect(workflow).toContain('architecture: ${{ matrix.python_arch }}');
    expect(workflow).toContain('python-version: "3.11.9"');
    expect(workflow).toContain("cache-dependency-path: |");
    expect(workflow).toContain("engine/requirements-build.txt");
    expect(workflow).toContain("engine/locks/*.lock");
    expect(workflow).toContain("python scripts/install-engine-lock.py");
    expect(workflow).toContain("python -m pip check");
    expect(workflow).toContain("transformers.is_torch_available()");
    expect(workflow.match(/OMNI_SKIP_BUILD_DEPENDENCY_INSTALL: "1"/gu)).toHaveLength(2);
    expect(workflow).not.toContain("python -m pip install -r engine/requirements.txt");
    expect(workflow).not.toContain("--index-url https://download.pytorch.org");
    expect(workflow).toContain("npm run test:python:portable");
    expect(workflow).toContain("npm run build:engine:win");
    expect(workflow).toContain("smoke-engine.ps1");
    expect(workflow).toContain("smoke-windows-package.ps1");
    expect(workflow).not.toContain("secrets.");
    expect(workflow).toContain('OMNI_EXPECT_SIGNED: "0"');
    expect(workflow).toContain(
      "windows-package-smoke-${{ matrix.arch }}.json"
    );
    expect(read("scripts/smoke-engine.ps1")).toContain(
      'Invoke-WorkerRpc -Id "health" -Method "health"'
    );
    expect(read("scripts/smoke-engine.ps1")).toContain(
      'Invoke-WorkerRpc -Id "create" -Method "create"'
    );
    expect(read("scripts/smoke-engine.ps1")).toContain(
      'Invoke-WorkerRpc -Id "load" -Method "load"'
    );
    expect(read("scripts/smoke-windows-package.ps1")).toContain(
      '-ArgumentList @("/S", "/D=$InstallRoot")'
    );
    expect(read("scripts/smoke-windows-package.ps1")).toContain(
      '"Omni AGI Studio.exe"'
    );
    expect(read("scripts/smoke-windows-package.ps1")).toContain(
      "Get-InstalledAppExecutable"
    );
    expect(read("scripts/smoke-windows-package.ps1")).toContain(
      "Installed desktop executable disappeared during the worker smoke."
    );
    expect(read("scripts/smoke-windows-package.ps1")).toContain(
      "Get-PeArchitecture"
    );
    expect(read("scripts/smoke-windows-package.ps1")).toContain(
      "Remove-TreeWithRetry"
    );
    expect(packageDocument.scripts["package:win"]).toContain("-Arch x64");
    expect(packageDocument.scripts["package:win:arm64"]).toContain("-Arch arm64");
    expect(packageDocument.build.win.target).toEqual(expect.arrayContaining(["nsis", "zip"]));
    expect(packageDocument.build.nsis.useZip).toBe(false);
    // 26.15.6 fixed NSIS archive filters that skipped ARM64 PE and native files.
    expect(packageDocument.devDependencies["electron-builder"]).toBe("26.15.7");
    expect(lockDocument.packages["node_modules/electron-builder"]?.version).toBe(
      "26.15.7"
    );
    expect(lockDocument.packages["node_modules/app-builder-lib"]?.version).toBe(
      "26.15.7"
    );
    expect(read("playwright.config.ts")).toContain(
      "retries: process.env.CI ? 1 : 0"
    );
    const electronE2e = read("tests/e2e/electron.spec.ts");
    expect(electronE2e).toContain("electron.launch({");
    expect(electronE2e).toContain("executablePath,");
    expect(electronE2e).not.toContain("chromium.connectOverCDP");
    expect(electronE2e).not.toContain("process.arch === \"arm64\"");
  });

  it("documents the exact Auto training and checkpoint recovery contract", () => {
    const training = read("docs/TRAINING.md");
    const planner = read("docs/RESOURCE_MEMORY_PLANNER.md");
    const requirements = read("docs/REQUIREMENTS.md");
    const completion = read("docs/COMPLETION_AUDIT.md");
    const prose = (value: string): string => value.replace(/\s+/gu, " ");

    expect(prose(training)).toContain("stable `v1.1.0` training and recovery contract");
    expect(prose(training)).toContain(
      "largest safe physical size that exactly divides the target"
    );
    expect(prose(training)).toContain("`4 × 4`, `2 × 8`, or `1 × 16`");
    expect(prose(training)).toContain("every 512 committed source records");
    expect(prose(training)).toContain(
      "Optimizer moments and step counters are persisted as typed safe tensors"
    );
    expect(prose(planner)).toContain("`train_batch_size * gradient_accumulation`");
    expect(prose(planner)).toContain("`4 → 2 → 1`");
    expect(prose(planner)).toContain(
      "default cadence of 512 committed source records"
    );
    expect(prose(requirements)).toContain(
      "Auto corpus training preserves an exact logical optimizer target under RAM pressure"
    );
    expect(prose(completion)).toContain(
      "Full mutable generations normally commit every 512 source records"
    );
  });

  it("defines the stable v1 release across Windows, macOS, and Linux", () => {
    const packageDocument = readJson<{
      version: string;
      description: string;
      scripts: Record<string, string>;
      build: {
        mac: {
          target: string[];
          artifactName: string;
          extendInfo: Record<string, string>;
        };
        linux: { target: string[]; artifactName: string };
      };
    }>("package.json");
    const packageLock = readJson<{
      version: string;
      packages: Record<string, { version: string }>;
    }>("package-lock.json");
    const macos = read(".github/workflows/macos.yml");
    const linux = read(".github/workflows/linux.yml");
    const release = read(".github/workflows/release.yml");

    expect(packageDocument.version).toBe("1.1.0");
    expect(packageLock.version).toBe(packageDocument.version);
    expect(packageLock.packages[""]?.version).toBe(packageDocument.version);
    expect(packageDocument.description).toContain("cross-platform");
    expect(packageDocument.build.mac.target).toEqual(
      expect.arrayContaining(["dmg", "zip"])
    );
    expect(packageDocument.build.linux.target).toEqual(
      expect.arrayContaining(["AppImage", "deb", "tar.gz"])
    );
    expect(packageDocument.build.mac.artifactName).toContain("macOS-${arch}");
    expect(packageDocument.build.mac.extendInfo).toMatchObject({
      NSMicrophoneUsageDescription: expect.any(String),
      NSCameraUsageDescription: expect.any(String),
      NSScreenCaptureUsageDescription: expect.any(String),
      NSAppleEventsUsageDescription: expect.any(String)
    });
    expect(read("build/entitlements.mac.plist")).toContain(
      "com.apple.security.automation.apple-events"
    );
    expect(packageDocument.build.linux.artifactName).toContain("Linux-${arch}");
    for (const script of [
      "package:mac:x64",
      "package:mac:arm64",
      "package:linux:x64",
      "package:linux:arm64",
      "verify:release"
    ]) {
      expect(packageDocument.scripts[script]).toBeTypeOf("string");
    }

    expect(macos).toContain("runner: macos-15-intel");
    expect(macos).toContain("runner: macos-15");
    expect(macos).toContain('python-version: "3.11.9"');
    expect(macos).toContain("python scripts/install-engine-lock.py");
    expect(macos).toContain("engine/requirements-build.txt");
    expect(macos).toContain("engine/locks/*.lock");
    expect(macos).toContain('OMNI_SKIP_BUILD_DEPENDENCY_INSTALL: "1"');
    expect(macos).not.toContain("matrix.torch_version");
    expect(macos).not.toContain("matrix.numpy_constraint");
    expect(macos).toContain("package:mac:${{ matrix.arch }}");
    expect(macos).toContain("--desktop-e2e");
    expect(linux).toContain("runner: ubuntu-24.04");
    expect(linux).toContain("runner: ubuntu-24.04-arm");
    expect(linux).toContain('python-version: "3.11.16"');
    expect(linux).toContain("python scripts/install-engine-lock.py");
    expect(linux).toContain("engine/requirements-build.txt");
    expect(linux).toContain("engine/locks/*.lock");
    expect(linux).toContain('OMNI_SKIP_BUILD_DEPENDENCY_INSTALL: "1"');
    expect(linux).toContain("package:linux:${{ matrix.arch }}");
    const posixSmoke = read("scripts/smoke-posix-package.mjs");
    expect(posixSmoke).toContain("xvfb-run");
    expect(posixSmoke).toContain('rm(debRoot, { recursive: true, force: true })');
    expect(posixSmoke).toContain('rm(appImageRoot, { recursive: true, force: true })');
    expect(posixSmoke).toContain("Desktop smoke resources:");
    expect(release).toContain('tags: ["v*.*.*"]');
    expect(release).toContain(
      'git merge-base --is-ancestor "$VERIFIED_COMMIT" origin/main'
    );
    expect(release).toContain("commit: ${{ steps.release.outputs.commit }}");
    expect(release).not.toContain("ref: ${{ needs.verify.outputs.tag }}");
    expect(release).toContain("verify-release-artifacts.mjs");
    expect(release.match(/python-version: "3\.11\.9"/gu)).toHaveLength(2);
    expect(release.match(/python-version: "3\.11\.16"/gu)).toHaveLength(1);
    expect(release.match(/python scripts\/install-engine-lock\.py/gu)).toHaveLength(3);
    expect(release.match(/engine\/requirements-build\.txt/gu)).toHaveLength(3);
    expect(release.match(/engine\/locks\/\*\.lock/gu)).toHaveLength(3);
    expect(release.match(/OMNI_SKIP_BUILD_DEPENDENCY_INSTALL: "1"/gu)).toHaveLength(3);
    expect(release).not.toContain("python -m pip install -r engine/requirements.txt");
    expect(release).not.toContain("matrix.torch_version");
    expect(release).not.toContain("matrix.numpy_constraint");
    expect(release).toContain("Canonicalize public names");
    expect(release).toContain(
      "(cd artifacts && sha256sum --check SHA256SUMS.txt)"
    );
    expect(release).toContain("OMNI_RELEASE_CSC_LINK");
    expect(release).toContain(
      'if [[ -n "${OMNI_RELEASE_CSC_LINK:-}" ]]; then export CSC_LINK='
    );
    expect(release).toContain("OMNI_RELEASE_APPLE_API_KEY_P8");
    expect(release).toContain("omni-notary-api-key.p8");
    expect(release).toContain("base64.b64decode");
    expect(release).toContain('export APPLE_API_KEY="$OMNI_NOTARY_KEY_FILE"');
    expect(release).toContain("destination.chmod(0o600)");
    expect(release).toContain("unset OMNI_RELEASE_APPLE_API_KEY_P8");
    expect(release).not.toContain(
      'export APPLE_API_KEY="$OMNI_RELEASE_APPLE_API_KEY"'
    );
    expect(release).not.toMatch(
      /^\s+CSC_LINK: \$\{\{ secrets\.MACOS_CSC_LINK \}\}$/m
    );
    expect(read("scripts/package-posix.sh")).toContain(
      '[[ -n "${CSC_LINK:-}" ]] || unset CSC_LINK'
    );
    expect(read("scripts/verify-release-artifacts.mjs")).toContain(
      "SHA256SUMS.txt"
    );
    expect(read("scripts/verify-release-artifacts.mjs")).toContain(
      "publicReleaseAssetName"
    );
    expect(read("scripts/build-engine-posix.sh")).toContain(
      "--collect-all torch"
    );
    expect(read("scripts/package-posix.sh")).toContain(
      "requires a matching host"
    );
    expect(read("scripts/verify-release.mjs")).toContain(
      "requestedTag === expectedTag"
    );
  });

  it("ships native Android and iOS companions for the same desktop brain", () => {
    const packageDocument = readJson<{
      version: string;
      scripts: Record<string, string>;
    }>("package.json");
    const gateway = read("src/main/mobileGateway.ts");
    const mcpClient = read("src/main/mcpClient.ts");
    const enginePackage = read("engine/omni_core/__init__.py");
    const androidBuild = read("mobile/android/app/build.gradle.kts");
    const androidClient = read(
      "mobile/android/app/src/main/java/ai/omniagi/companion/OmniGatewayClient.kt"
    );
    const iosClient = read("mobile/ios/OmniCompanion/GatewayClient.swift");
    const iosViewModel = read("mobile/ios/OmniCompanion/CompanionViewModel.swift");
    const iosInfo = read("mobile/ios/OmniCompanion/Info.plist");
    const iosProject = read("mobile/ios/OmniCompanion.xcodeproj/project.pbxproj");
    const iosWorkflow = read(".github/workflows/ios.yml");
    const [major, minor, patch] = parseStableVersion(packageDocument.version);
    const mobileBuildNumber = major * 10_000 + minor * 100 + patch;

    expect(gateway).toContain("exposes the existing BrainService");
    expect(gateway).toContain("never instantiates or copies a model");
    expect(gateway).toContain("experienceMatch");
    expect(enginePackage).toContain(`__version__ = "${packageDocument.version}"`);
    expect(mcpClient).toContain(
      `clientInfo: { name: "Omni AGI Studio", version: "${packageDocument.version}" }`
    );
    expect(androidBuild).toContain(`versionCode = ${mobileBuildNumber}`);
    expect(androidBuild).toContain(`versionName = "${packageDocument.version}"`);
    expect(androidClient).toContain("setChunkedStreamingMode");
    expect(iosClient).toContain("uploadTask(with: request, fromFile: fileURL)");
    expect(iosClient).toContain('setValue("Bearer \\(token)"');
    expect(iosViewModel).toContain("KeychainStore.save");
    expect(iosViewModel).toContain("queuedInputs.append");
    expect(iosInfo).toContain("NSAllowsLocalNetworking");
    expect(iosInfo).not.toContain("NSAllowsArbitraryLoads");
    expect(iosInfo).toContain("<string>$(MARKETING_VERSION)</string>");
    expect(iosInfo).toContain("<string>$(CURRENT_PROJECT_VERSION)</string>");
    expect(
      [...iosProject.matchAll(/\bMARKETING_VERSION = ([^;]+);/gu)].map(
        (match) => match[1]
      )
    ).toEqual([packageDocument.version, packageDocument.version]);
    expect(
      [...iosProject.matchAll(/\bCURRENT_PROJECT_VERSION = ([^;]+);/gu)].map(
        (match) => match[1]
      )
    ).toEqual([String(mobileBuildNumber), String(mobileBuildNumber)]);
    expect(iosWorkflow).toContain("xcrun simctl bootstatus");
    expect(iosWorkflow).toContain("npm run test:ios");
    expect(iosWorkflow).toContain("package:ios:unsigned");
    for (const script of [
      "package:android:debug",
      "test:android",
      "package:ios:simulator",
      "package:ios:unsigned",
      "package:ios:signed",
      "test:ios"
    ]) {
      expect(packageDocument.scripts[script]).toBeTypeOf("string");
    }
  });
});
