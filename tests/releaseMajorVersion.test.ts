import { readFileSync } from "node:fs";
import { resolve } from "node:path";
import { pathToFileURL } from "node:url";
import { expect, it } from "vitest";

it("admits stable v2 without weakening exact version or immutable tag gates", async () => {
  const { stableReleaseVersion } = await import(pathToFileURL(resolve("scripts/release-version-policy.mjs")).href);
  expect(stableReleaseVersion("2.0.0")).toEqual({ major: 2, minor: 0, patch: 0, tag: "v2.0.0", androidVersionCode: 20_000 });
  expect(stableReleaseVersion("1.1.1").androidVersionCode).toBe(10_101);
  for (const invalid of ["0.9.0", "2.0.0-beta", "02.0.0", "2.01.0", "2.0", "2.100.0", "210001.0.0"]) {
    expect(() => stableReleaseVersion(invalid)).toThrow();
  }
  const source = readFileSync(resolve("scripts/verify-release.mjs"), "utf8");
  expect(source).toContain("stableReleaseVersion(packageDocument.version)");
  expect(source).toContain("packageLock.version === packageDocument.version");
  expect(source).toContain("requestedTag === expectedTag");
  expect(source).toContain("verifyReleaseTagBinding(requestedTag)");
});
