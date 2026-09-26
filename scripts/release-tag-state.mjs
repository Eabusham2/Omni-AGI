import { spawnSync } from "node:child_process";

function runGit(cwd, args) {
  return spawnSync("git", args, {
    cwd,
    encoding: "utf8",
    stdio: ["ignore", "pipe", "pipe"]
  });
}

function gitFailure(result, action) {
  const detail = (result.stderr || result.stdout || "unknown Git error").trim();
  return new Error(`${action}: ${detail}`);
}

/**
 * A release tag may be absent while preparing a release, or it may already
 * resolve to the checked-out commit during release CI. It must never be reused
 * for a different commit because published release tags are immutable.
 */
export function verifyReleaseTagBinding(tag, { cwd = process.cwd() } = {}) {
  if (!/^v[0-9]+\.[0-9]+\.[0-9]+$/u.test(tag)) {
    throw new Error(`Release tag ${String(tag)} is not vMAJOR.MINOR.PATCH.`);
  }

  const repository = runGit(cwd, ["rev-parse", "--is-inside-work-tree"]);
  if (repository.status !== 0 || repository.stdout.trim() !== "true") {
    throw gitFailure(repository, "Release verification requires a Git worktree");
  }

  const tagReference = `refs/tags/${tag}`;
  const tagProbe = runGit(cwd, ["show-ref", "--verify", "--quiet", tagReference]);
  if (tagProbe.status === 1) {
    return { state: "unbound", tag };
  }
  if (tagProbe.status !== 0) {
    throw gitFailure(tagProbe, `Could not inspect ${tagReference}`);
  }

  const tagCommit = runGit(cwd, ["rev-parse", "--verify", `${tagReference}^{commit}`]);
  if (tagCommit.status !== 0) {
    throw gitFailure(tagCommit, `Could not resolve ${tagReference}`);
  }
  const headCommit = runGit(cwd, ["rev-parse", "--verify", "HEAD^{commit}"]);
  if (headCommit.status !== 0) {
    throw gitFailure(headCommit, "Could not resolve the current commit");
  }

  const tagged = tagCommit.stdout.trim();
  const current = headCommit.stdout.trim();
  if (tagged !== current) {
    throw new Error(
      `Release tag ${tag} already points to ${tagged}, not current commit ${current}. ` +
        "Use a new stable-v1 patch version; do not move or reuse an immutable tag."
    );
  }
  return { state: "current", tag, commit: current };
}
