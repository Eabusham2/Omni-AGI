import { createHash } from "node:crypto";

/**
 * This policy is evaluated by the already-running application, outside an
 * isolated candidate worktree. A candidate may improve ordinary source and add
 * tests, but it may not rewrite the evaluator, permission boundary, rollback
 * implementation, immutable-origin repository, or tests that existed when the
 * candidate was forked.
 */
export const EVOLUTION_EVALUATOR_VERSION = 1;

export const EVOLUTION_TEST_NAMES = ["typecheck", "unit", "build"] as const;

export const EVOLUTION_PROTECTED_PATHS = [
  "src/main/evolutionPolicy.ts",
  "src/main/evolutionController.ts",
  "src/main/toolExecutor.ts",
  "src/main/brainRepository.ts",
  "src/main/ipc.ts",
  "src/preload/index.ts",
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
  promotion: "exact-diff-and-evaluator-hash",
  rollback: "exact-promotion-and-first-parent"
};

export const EVOLUTION_POLICY_SHA256 = createHash("sha256")
  .update(JSON.stringify(POLICY_DOCUMENT))
  .digest("hex");

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
