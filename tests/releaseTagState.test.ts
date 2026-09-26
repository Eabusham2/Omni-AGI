import { spawnSync } from "node:child_process";
import { mkdtempSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join, resolve } from "node:path";
import { pathToFileURL } from "node:url";
import { afterEach, describe, expect, it } from "vitest";

type TagBinding = {
  state: "unbound" | "current";
  tag: string;
  commit?: string;
};

type TagVerifier = (
  tag: string,
  options: { cwd: string }
) => TagBinding;

const temporaryDirectories: string[] = [];

function git(cwd: string, ...args: string[]): string {
  const result = spawnSync("git", args, { cwd, encoding: "utf8" });
  if (result.status !== 0) {
    throw new Error(result.stderr || result.stdout || `git ${args.join(" ")} failed`);
  }
  return result.stdout.trim();
}

async function loadVerifier(): Promise<TagVerifier> {
  const modulePath = pathToFileURL(
    resolve("scripts/release-tag-state.mjs")
  ).href;
  const loaded = (await import(modulePath)) as {
    verifyReleaseTagBinding: TagVerifier;
  };
  return loaded.verifyReleaseTagBinding;
}

afterEach(() => {
  while (temporaryDirectories.length > 0) {
    rmSync(temporaryDirectories.pop() as string, { recursive: true, force: true });
  }
});

describe("immutable release tag gate", () => {
  it("allows an unbound/current tag and rejects one bound to another commit", async () => {
    const directory = mkdtempSync(join(tmpdir(), "omni-release-tag-"));
    temporaryDirectories.push(directory);
    git(directory, "init", "--initial-branch=main");
    git(directory, "config", "user.name", "Release Gate Test");
    git(directory, "config", "user.email", "release-gate@example.invalid");
    writeFileSync(join(directory, "release.txt"), "first\n");
    git(directory, "add", "release.txt");
    git(directory, "commit", "-m", "first");

    const verify = await loadVerifier();
    expect(verify("v1.1.0", { cwd: directory })).toEqual({
      state: "unbound",
      tag: "v1.1.0"
    });

    git(directory, "tag", "-a", "v1.1.0", "-m", "immutable release");
    const taggedCommit = git(directory, "rev-parse", "HEAD");
    expect(verify("v1.1.0", { cwd: directory })).toEqual({
      state: "current",
      tag: "v1.1.0",
      commit: taggedCommit
    });

    writeFileSync(join(directory, "release.txt"), "second\n");
    git(directory, "add", "release.txt");
    git(directory, "commit", "-m", "second");
    expect(() => verify("v1.1.0", { cwd: directory })).toThrow(
      /already points to .* not current commit .*do not move or reuse an immutable tag/i
    );
  });
});
