import { readFileSync } from "node:fs";
import { resolve } from "node:path";
import { describe, expect, it } from "vitest";

function source(path: string): string {
  return readFileSync(resolve(import.meta.dirname, "..", path), "utf8");
}

describe("mobile companion transport security", () => {
  it("disables Android release cleartext and pins HTTPS in the client", () => {
    const build = source("mobile/android/app/build.gradle.kts");
    const client = source(
      "mobile/android/app/src/main/java/ai/omniagi/companion/OmniGatewayClient.kt"
    );
    const network = source("mobile/android/app/src/main/res/xml/network_security_config.xml");
    expect(build).toMatch(
      /release\s*\{[\s\S]*?manifestPlaceholders\["usesCleartextTraffic"\]\s*=\s*"false"/u
    );
    expect(client).toContain("HTTP companion access is limited to loopback or the Android emulator");
    expect(client).toContain("certificateFingerprint(leaf) != expected");
    expect(client).toContain("connection.sslSocketFactory = context.socketFactory");
    expect(client).toContain("connection.instanceFollowRedirects = false");
    expect(client).toContain('connection.setRequestProperty("authorization", "Bearer $currentToken")');
    expect(network).toContain('<base-config cleartextTrafficPermitted="false"');
    expect(network).toContain(">127.0.0.1</domain>");
    expect(network).toContain(">10.0.2.2</domain>");
    expect(network).not.toContain(">192.168.");
  });

  it("keeps iOS ATS globally strict and pins normal plus upload sessions", () => {
    const plist = source("mobile/ios/OmniCompanion/Info.plist");
    const client = source("mobile/ios/OmniCompanion/GatewayClient.swift");
    expect(plist).not.toContain("NSAllowsArbitraryLoads");
    expect(plist).not.toContain("NSAllowsArbitraryLoadsInWebContent");
    expect(client).toContain("HTTP companion access is limited to loopback");
    expect(client).toContain("answerPinnedChallenge(");
    expect(client.match(/answerPinnedChallenge\(/gu)).toHaveLength(3);
    expect(client).toContain('request.setValue("Bearer \\(token)", forHTTPHeaderField: "authorization")');
  });

  it("advertises certificate identity in Studio without exposing its key", () => {
    const gateway = source("src/main/mobileGateway.ts");
    const identity = source("src/main/mobileTlsIdentity.ts");
    const renderer = source("src/renderer/src/App.tsx");
    expect(gateway).toContain("certificate-pinned-tls");
    expect(gateway).toContain("#sha256=${certificateSha256}");
    expect(gateway).toContain("Bearer ");
    expect(identity).toContain("mode: 0o600");
    expect(renderer).toContain("TLS SHA-256");
    expect(renderer).toContain("HTTP on loopback and the Android emulator only");
  });
});
