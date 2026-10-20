import { spawnSync } from "node:child_process";
import { mkdtemp, readFile, rm } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join, resolve } from "node:path";
import { expect, it } from "vitest";

it("keeps package CI on fixed source contracts and health-only worker paths", async () => {
  const root = resolve("."), script = join(root, "scripts/run-python-contracts.mjs");
  const listed = spawnSync(process.execPath, [script, "--list"], { cwd: root, encoding: "utf8", shell: false, timeout: 15_000 });
  expect(listed.status, listed.stderr).toBe(0);
  const manifest = JSON.parse(listed.stdout) as { files: Array<{ path: string; sha256: string }> };
  expect(manifest).toMatchObject({ fullTestDiscovery: false, neuralRuntimeImportsAllowed: false, brainCreated: false, trainingPerformed: false });
  expect(manifest.files.map(value => value.path)).toEqual([
    "engine/tests/test_native_compute_tensor_bytes.py",
    "engine/tests/test_manual_regional_execution.py",
    "engine/tests/test_manual_regional_boundaries.py",
    "engine/tests/test_whole_path_memory_scheduler.py",
    "engine/tests/test_reply_execution_planner.py",
    "engine/tests/test_mixed_region_schedule.py",
    "engine/tests/test_outer_learning_pipeline.py",
    "engine/tests/test_native_kernel_packaging_integrity.py",
    "engine/tests/test_retained_context_capacity.py",
    "engine/tests/test_ponder_episode_provenance.py",
    "engine/tests/test_stdio_shutdown_reserve.py",
    "engine/tests/test_live_sensory_handoff.py"
  ]);
  expect(manifest.files.every(value => /^[a-f0-9]{64}$/.test(value.sha256))).toBe(true);
  for (const name of ["windows", "macos", "linux", "release"]) {
    const workflow = await readFile(join(root, `.github/workflows/${name}.yml`), "utf8");
    expect(workflow.match(/^        run: npm run test:python:ci:contracts$/gm)).toHaveLength(name === "release" ? 3 : 1);
    expect(workflow).not.toMatch(/^\s*run: npm run test:python(?:\s|$)/m);
    expect(workflow).not.toMatch(/^\s*run: .*unittest discover/m);
  }
  const posix = await readFile(join(root, "scripts/smoke-posix-package.mjs"), "utf8");
  expect(posix).toContain('"--health-only"');
  const windows = await readFile(join(root, "scripts/smoke-windows-package.ps1"), "utf8");
  expect(windows.match(/-HealthOnly\b/g)).toHaveLength(2);
  const rejected = spawnSync(process.execPath, [script, "engine/tests/test_chat_single_pass.py"], { cwd: root, encoding: "utf8", shell: false, timeout: 15_000 });
  expect(rejected.status).not.toBe(0);
  expect(rejected.stderr).toContain("fixed selection");
});

it("registers focused chronology, source and surface cases once in reachable bounded selectors without discovery", async () => {
  const root = resolve("."), document = JSON.parse(await readFile(join(root, "package.json"), "utf8")) as { scripts: Record<string, string> };
  const visited = new Set<string>(), pending = ["test:node:ci"];
  while (pending.length) {
    const name = pending.shift()!;
    if (visited.has(name)) continue;
    visited.add(name);
    const command = document.scripts[name];
    expect(command, `Missing fixed CI selector ${name}`).toBeTypeOf("string");
    if (typeof command !== "string") throw new Error(`Missing fixed CI selector ${name}`);
    for (const match of command.matchAll(/\bnpm run (test:node[^\s;&]*)/g)) pending.push(match[1]!);
  }
  expect(visited).toEqual(new Set(["test:node:ci", "test:node:v2:contracts", "test:node:v2:tasks", "test:node:v2:completion", "test:node:v2:integration-gaps"]));
  const selected = [...visited].flatMap(name => document.scripts[name]!.match(/tests\/[A-Za-z0-9_./-]+\.test\.tsx?\b/g) ?? []);
  for (const path of [
    "tests/ponderEpisodeProvenance.test.ts", "tests/chatPonderGroups.test.ts", "tests/chatReplyProjection.test.ts", "tests/conversationNoticePlacement.test.tsx", "tests/responsiveLayout.test.ts",
    "tests/intrinsicImprovementPolicy.test.ts", "tests/learningQuarantine.test.ts", "tests/librarySelection.test.ts", "tests/pluginModuleAuthoring.test.tsx",
    "tests/taskFolderAttachments.test.ts", "tests/trainingEvolutionWorkspace.test.tsx", "tests/universalSourcePicker.test.tsx",
    "tests/appearanceAuditObservation.test.ts", "tests/teacherStatsTools.test.ts", "tests/apiTeacherToolAuthority.test.ts",
    "tests/systemStatsService.test.ts", "tests/systemTemperatureReader.test.ts", "tests/automationResidencySample.test.ts",
    "tests/demoChatTruth.test.tsx", "tests/liveObservationBackend.test.ts", "tests/livePerceptionController.test.ts"
  ]) {
    expect(selected.filter(value => value === path), `Focused case must be selected exactly once: ${path}`).toHaveLength(1);
    expect((await readFile(join(root, path), "utf8")).length).toBeGreaterThan(0);
  }
  for (const name of visited) {
    const command = document.scripts[name]!;
    expect(command).toContain("--maxWorkers=");
    expect(command).not.toMatch(/\bvitest run(?:\s*(?:&&|$))/);
    expect(command).not.toMatch(/(?:test:python:portable|unittest discover|engine\/tests)/);
  }
  const python = await readFile(join(root, "scripts/run-python-contracts.py"), "utf8");
  expect(python).toContain('sys.meta_path.insert(0, NoNeuralImports())');
  expect(python).toContain('"torch", "tensorflow", "jax", "mlx", "transformers", "safetensors", "omni_core"');
  expect(python).not.toMatch(/unittest\.defaultTestLoader\.discover\(/);
  expect(await readFile(join(root, "scripts/run-python-contracts.mjs"), "utf8")).toContain('["-I",');
});

it("does not persist the release write credential and retains exact repository/tag binding", async () => {
  const workflow = await readFile(resolve(".github/workflows/release.yml"), "utf8"), publish = workflow.split("\n  publish:\n")[1]!;
  expect(publish).toContain("persist-credentials: false");
  expect(publish).toContain("ref: ${{ needs.verify.outputs.commit }}");
  expect(publish).toContain("pattern: omni-release-*"); expect(workflow).not.toContain("omni-v1-");
  expect(publish).toContain('GIT_CONFIG_KEY_0="http.${GITHUB_SERVER_URL%/}/.extraheader"');
  expect(publish).toContain('GIT_CONFIG_VALUE_0="AUTHORIZATION: basic $omni_release_git_auth"');
  expect(publish).toContain('omni_release_expected_origin="${GITHUB_SERVER_URL%/}/$GITHUB_REPOSITORY"');
  expect(publish).toContain('test "$(git rev-parse --verify "$RELEASE_TAG^{commit}")" = "$RELEASE_COMMIT"');
  expect(publish).not.toMatch(/git config.*(?:token|extraheader|credential\.helper)/i);
  const directory = await mkdtemp(join(tmpdir(), "omni-release-runtime-auth-"));
  try {
    const initialized = spawnSync("git", ["init", "--quiet", directory], { encoding: "utf8", shell: false, timeout: 10_000 });
    expect(initialized.status, initialized.stderr).toBe(0);
    const key = "http.https://github.com/.extraheader", fake = "AUTHORIZATION: basic OWNED_INERT_FIXTURE_NOT_A_CREDENTIAL";
    const transient = spawnSync("git", ["config", "--get", key], { cwd: directory, encoding: "utf8", shell: false, timeout: 10_000,
      env: { ...process.env, GIT_CONFIG_COUNT: "1", GIT_CONFIG_KEY_0: key, GIT_CONFIG_VALUE_0: fake } });
    expect(transient.status, transient.stderr).toBe(0); expect(transient.stdout.trim()).toBe(fake);
    expect(await readFile(join(directory, ".git/config"), "utf8")).not.toContain(fake);
    const persisted = spawnSync("git", ["config", "--local", "--get", key], { cwd: directory, encoding: "utf8", shell: false, timeout: 10_000,
      env: { ...process.env, GIT_CONFIG_COUNT: "0" } });
    expect(persisted.status).toBe(1);
  } finally { await rm(directory, { recursive: true, force: true }); }
});
