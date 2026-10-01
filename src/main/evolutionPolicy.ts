import { createHash } from "node:crypto";

/**
 * This policy is evaluated by the already-running application, outside an
 * isolated candidate worktree. A candidate may improve ordinary source and add
 * tests, but it may not rewrite the evaluator, permission boundary, rollback
 * implementation, immutable-origin repository, or tests that existed when the
 * candidate was forked.
 */
export const EVOLUTION_EVALUATOR_VERSION = 2;

export const EVOLUTION_TEST_NAMES = ["typecheck", "unit", "build"] as const;

export const EVOLUTION_PROTECTED_PATHS = [
  "src/main/evolutionPolicy.ts",
  "src/main/evolutionController.ts",
  "src/main/toolExecutor.ts",
  "src/main/sourceRuntimeContract.ts",
  "src/main/sourceRuntimeLifecycle.ts",
  "src/main/brainRepository.ts",
  "src/main/index.ts",
  "src/main/ipc.ts",
  "src/preload/index.ts",
  "engine/omni_core/evolution.py",
  "engine/omni_core/paired_geometry_statistics.py",
  "engine/omni_core/geometry_holdout_evaluation.py",
  "engine/omni_core/registered_geometry_holdouts.py",
  "engine/omni_core/evolution_anchors.py",
  "engine/omni_core/architecture_migration.py",
  "package.json",
  "package-lock.json",
  "scripts/build-engine.ps1",
  "scripts/build-engine-posix.sh",
  "scripts/package-windows.ps1",
  "scripts/package-posix.sh",
  "vitest.config.ts",
  "tsconfig.json",
  "tsconfig.node.json",
  "tsconfig.web.json"
] as const;

export const EVOLUTION_BENCHMARK_DOMAINS = [
  "capability",
  "retention",
  "modality",
  "tools",
  "latency",
  "resources",
  "evolution-integrity"
] as const;

const POLICY_DOCUMENT = {
  version: EVOLUTION_EVALUATOR_VERSION,
  tests: EVOLUTION_TEST_NAMES,
  protectedPaths: EVOLUTION_PROTECTED_PATHS,
  benchmarkDomains: EVOLUTION_BENCHMARK_DOMAINS,
  baseline: "authorized-parent-commit",
  candidateBoundary: "isolated-git-worktree",
  promotion: "exact-diff-and-evaluator-hash-with-observed-paired-benefit",
  rollback: "exact-promotion-and-first-parent"
};

export const EVOLUTION_POLICY_SHA256 = createHash("sha256")
  .update(JSON.stringify(POLICY_DOCUMENT))
  .digest("hex");

/** Fixed functional checks are evidence of their result, not neural quality. */
export function sourceImprovementEvidence(baseline: unknown, candidate: unknown) {
  const parse = (input: unknown) => {
    const pairs = new Map<string, boolean>();
    if (!Array.isArray(input)) return pairs;
    for (const raw of input) {
      if (!raw || typeof raw !== "object" || Array.isArray(raw)) continue;
      const value = raw as Record<string, unknown>;
      if (!EVOLUTION_TEST_NAMES.includes(value.name as typeof EVOLUTION_TEST_NAMES[number])) continue;
      if (pairs.has(String(value.name)) || typeof value.passed !== "boolean" ||
          !Number.isSafeInteger(value.exitCode) ||
          (value.passed ? value.exitCode !== 0 : ![1, 2].includes(Number(value.exitCode)))) {
        return new Map<string, boolean>();
      }
      pairs.set(String(value.name), value.passed);
    }
    return pairs;
  };
  const before = parse(baseline), after = parse(candidate);
  const complete = EVOLUTION_TEST_NAMES.every(name => before.has(name) && after.has(name));
  const improvedChecks = complete ? EVOLUTION_TEST_NAMES.filter(name => !before.get(name) && after.get(name)) : [];
  const regressions = complete ? EVOLUTION_TEST_NAMES.filter(name => before.get(name) && !after.get(name)) : [];
  return {
    method: "paired-immutable-functional-checks" as const,
    allRequiredPairs: complete,
    improvedChecks,
    regressions,
    passed: complete && improvedChecks.length > 0 && regressions.length === 0,
    neuralQualityEstablished: false,
    timingBenefitEstablished: false
  };
}

export function isProtectedEvolutionPath(
  path: string,
  baselineTestPaths: ReadonlySet<string>
): boolean {
  const normalized = path.replaceAll("\\", "/").replace(/^\.\/+/, "");
  return (
    baselineTestPaths.has(normalized) ||
    EVOLUTION_PROTECTED_PATHS.some(
      (protectedPath) =>
        normalized === protectedPath || normalized.startsWith(`${protectedPath}/`)
    )
  );
}
