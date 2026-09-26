import { readFileSync } from "node:fs";
import { resolve } from "node:path";
import { afterEach, describe, expect, it, vi } from "vitest";
import {
  assertSafeRemoteUrl,
  safeFetch
} from "../src/main/brainService";
import { webLearningUrlAllowed } from "../src/renderer/src/remoteUrlPolicy";

afterEach(() => {
  vi.unstubAllGlobals();
});

describe("remote and loopback web-learning URL policy", () => {
  it("admits only explicit 127/8, ::1, and safely resolved localhost over HTTP", async () => {
    const local = { allowLoopback: true } as const;
    await expect(assertSafeRemoteUrl(
      new URL("http://127.0.0.2:41839/index.html"),
      local
    )).resolves.toBe("loopback");
    await expect(assertSafeRemoteUrl(
      // WHATWG canonicalizes this decimal IPv4 spelling to 127.0.0.1.
      new URL("http://2130706433:41839/index.html"),
      local
    )).resolves.toBe("loopback");
    await expect(assertSafeRemoteUrl(
      new URL("http://[0:0:0:0:0:0:0:1]:41839/index.html"),
      local
    )).resolves.toBe("loopback");
    await expect(assertSafeRemoteUrl(
      new URL("http://localhost.:41839/index.html"),
      {
        allowLoopback: true,
        resolveHostname: async () => ["127.8.9.10", "::1"]
      }
    )).resolves.toBe("loopback");

    await expect(assertSafeRemoteUrl(
      new URL("http://localhost:41839/index.html"),
      {
        allowLoopback: true,
        resolveHostname: async () => ["127.0.0.1", "10.0.0.4"]
      }
    )).rejects.toThrow(/exclusively to 127\/8 or ::1/i);
    await expect(assertSafeRemoteUrl(
      new URL("http://127.0.0.1:41839/index.html")
    )).rejects.toThrow(/loopback network URLs are not allowed/i);
  });

  it("rejects private, encoded-host, suffix, and IPv6-mapped bypasses", async () => {
    const blocked = [
      "http://10.0.0.1/private",
      "http://192.168.1.2/private",
      "http://127.0.0.1.evil.example/private",
      "http://localhost.evil.example/private",
      "http://[::ffff:127.0.0.1]/private",
      "http://[fe80::1]/private",
      "http://[::]/private"
    ];
    for (const value of blocked) {
      await expect(assertSafeRemoteUrl(new URL(value), {
        allowLoopback: true,
        resolveHostname: async () => ["93.184.216.34"]
      }), value).rejects.toThrow();
    }
    await expect(assertSafeRemoteUrl(
      new URL("https://[::ffff:127.0.0.1]/private"),
      { allowLoopback: true }
    )).rejects.toThrow(/private or reserved/i);
    await expect(assertSafeRemoteUrl(
      new URL("https://192.0.2.9/documentation"),
      { allowLoopback: true }
    )).rejects.toThrow(/private or reserved/i);
    await expect(assertSafeRemoteUrl(
      new URL("https://example.test/resource"),
      { resolveHostname: async () => ["93.184.216.34"] }
    )).resolves.toBe("remote");
  });

  it("blocks remote redirects into loopback before issuing the second request", async () => {
    const fetchMock = vi.fn(async () => new Response(null, {
      status: 302,
      headers: { Location: "http://2130706433:41839/admin" }
    }));
    vi.stubGlobal("fetch", fetchMock);

    await expect(safeFetch(
      new URL("https://93.184.216.34/start"),
      {},
      5,
      { allowLoopback: true }
    )).rejects.toThrow(/remote URL redirect cannot target a loopback/i);
    expect(fetchMock).toHaveBeenCalledOnce();
  });

  it("allows a user-started loopback crawl to follow a validated loopback redirect", async () => {
    const fetchMock = vi.fn()
      .mockResolvedValueOnce(new Response(null, {
        status: 302,
        headers: { Location: "http://127.0.0.2:41839/final" }
      }))
      .mockResolvedValueOnce(new Response("local source", { status: 200 }));
    vi.stubGlobal("fetch", fetchMock);

    const response = await safeFetch(
      new URL("http://127.0.0.1:41839/start"),
      {},
      5,
      { allowLoopback: true }
    );
    expect(response.status).toBe(200);
    await expect(response.text()).resolves.toBe("local source");
    expect(fetchMock).toHaveBeenCalledTimes(2);
  });

  it("keeps Build and Data URL affordances aligned with the main policy", () => {
    for (const value of [
      "https://docs.example.com",
      "http://127.0.0.42:41839/index.html",
      "http://127.1:41839/index.html",
      "http://2130706433:41839/index.html",
      "http://[::1]:41839/index.html",
      "http://localhost.:41839/index.html"
    ]) {
      expect(webLearningUrlAllowed(value), value).toBe(true);
    }
    for (const value of [
      "http://example.com",
      "http://10.0.0.1",
      "http://127.0.0.1.evil.example",
      "http://localhost.evil.example",
      "http://[::ffff:127.0.0.1]",
      "ftp://127.0.0.1/file",
      "https://user:secret@example.com"
    ]) {
      expect(webLearningUrlAllowed(value), value).toBe(false);
    }

    const app = readFileSync(
      resolve(import.meta.dirname, "../src/renderer/src/App.tsx"),
      "utf8"
    );
    expect(app).toContain("if (!webLearningUrlAllowed(url)) return;");
    expect(app).toContain("const crawlUrlAllowed = webLearningUrlAllowed(crawlUrl);");
    expect(app).toContain("HTTP is accepted only for verified loopback hosts.");
    expect(app).not.toContain('disabled={!/^https:\\/\\//i.test(webDraft.trim())}');
  });
});
