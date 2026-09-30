import { createHash } from "node:crypto";
import { readFile } from "node:fs/promises";
import { describe, expect, it } from "vitest";
import { bundledVideoRuntimeManifest } from "../src/main/videoRuntimeManifest";

const sha = (value: string) => createHash("sha256").update(value).digest("hex");

describe("catalog JSON line endings versus exact policy evidence", () => {
  it("binds manifest semantics across LF/CRLF without weakening payload or policy byte hashes", async () => {
    const [manifestText, policyText] = await Promise.all([
      readFile("licenses/ffmpeg-runtime-manifest.json", "utf8"),
      readFile("licenses/ffmpeg-runtime-policy.json", "utf8")
    ]);
    const lf = manifestText.replace(/\r\n/g, "\n"), crlf = lf.replace(/\n/g, "\r\n");
    expect(sha(lf)).not.toBe(sha(crlf));
    expect(sha(JSON.stringify(JSON.parse(lf)))).toBe(sha(JSON.stringify(JSON.parse(crlf))));
    const policy = JSON.parse(policyText);
    expect(policy.executable.automaticProvisioning.manifestContentSha256)
      .toBe(sha(JSON.stringify(JSON.parse(crlf))));
    expect(bundledVideoRuntimeManifest().artifacts).toHaveLength(6);
  });

  it("keeps exact policy-source restoration narrow and hashes parsed manifest in both verifiers", async () => {
    const [manifestSource, compliance, release, windows, attributes] = await Promise.all([
      readFile("src/main/videoRuntimeManifest.ts", "utf8"), readFile("scripts/verify-packaged-compliance.mjs", "utf8"),
      readFile(".github/workflows/release.yml", "utf8"), readFile(".github/workflows/windows.yml", "utf8"),
      readFile(".gitattributes", "utf8")
    ]);
    expect(manifestSource).toContain('update(JSON.stringify(bundledManifest))');
    expect(compliance).toContain('sha256(JSON.stringify(runtimeManifest))');
    expect(attributes).toContain("licenses/ffmpeg-runtime-policy.json text eol=lf");
    for (const workflow of [release, windows]) {
      expect(workflow).toContain('const path="licenses/ffmpeg-runtime-policy.json"');
      expect(workflow).toContain('["show","HEAD:"+path]');
      expect(workflow).toContain("FFmpeg policy differs from the verified source commit.");
    }
  });
});
