import { X509Certificate } from "node:crypto";
import { mkdtemp, readFile, rm, stat } from "node:fs/promises";
import { createServer } from "node:https";
import { request } from "node:https";
import type { TLSSocket } from "node:tls";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { afterEach, describe, expect, it } from "vitest";
import {
  createMobileTlsIdentity,
  loadOrCreateMobileTlsIdentity
} from "../src/main/mobileTlsIdentity";

const roots: string[] = [];

afterEach(async () => {
  await Promise.all(roots.splice(0).map((root) =>
    rm(root, { recursive: true, force: true })
  ));
});

describe("mobile gateway TLS identity", () => {
  it("creates a valid pinned P-256 server certificate", async () => {
    const identity = createMobileTlsIdentity(["192.168.1.25"]);
    const certificate = new X509Certificate(identity.certificatePem);
    expect(certificate.fingerprint256.replaceAll(":", "").toLowerCase()).toBe(
      identity.certificateSha256
    );
    expect(certificate.subjectAltName).toContain("IP Address:192.168.1.25");

    const server = createServer({
      key: identity.privateKeyPem,
      cert: identity.certificatePem,
      minVersion: "TLSv1.2"
    }, (_request, response) => response.end("secure"));
    await new Promise<void>((resolve) => server.listen(0, "127.0.0.1", resolve));
    try {
      const address = server.address();
      if (!address || typeof address === "string") throw new Error("TLS test server has no port.");
      const observed = await new Promise<{ body: string; fingerprint: string }>((resolve, reject) => {
        const call = request({
          hostname: "127.0.0.1",
          port: address.port,
          rejectUnauthorized: false
        }, (response) => {
          const chunks: Buffer[] = [];
          const fingerprint = (response.socket as TLSSocket)
            .getPeerX509Certificate()
            ?.fingerprint256.replaceAll(":", "").toLowerCase() ?? "";
          response.on("data", (chunk) => chunks.push(Buffer.from(chunk)));
          response.on("end", () => {
            resolve({
              body: Buffer.concat(chunks).toString("utf8"),
              fingerprint
            });
          });
        });
        call.once("error", reject);
        call.end();
      });
      expect(observed).toEqual({
        body: "secure",
        fingerprint: identity.certificateSha256
      });
    } finally {
      await new Promise<void>((resolve) => server.close(() => resolve()));
    }
  });

  it("persists one private identity with owner-only permissions", async () => {
    const root = await mkdtemp(join(tmpdir(), "omni-mobile-tls-"));
    roots.push(root);
    const first = await loadOrCreateMobileTlsIdentity(root, ["192.168.1.10"]);
    const second = await loadOrCreateMobileTlsIdentity(root, ["192.168.1.99"]);
    expect(second.certificateSha256).toBe(first.certificateSha256);
    const path = join(root, "mobile-tls-identity.json");
    expect((await stat(path)).mode & 0o777).toBe(0o600);
    expect(await readFile(path, "utf8")).toContain("BEGIN PRIVATE KEY");
  });
});
