import { mkdtemp, readFile, rm } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { afterEach, expect, test, vi } from "vitest";
import { ToolExecutor } from "../src/main/toolExecutor";

const browserState = vi.hoisted(() => ({
  captures: vi.fn(() => ({ toPNG: () => Buffer.from("screen fixture") })),
  scripts: [] as string[]
}));

vi.mock("../src/main/brainService", async (importOriginal) => ({
  ...await importOriginal<typeof import("../src/main/brainService")>(),
  assertSafeRemoteUrl: vi.fn(async () => "remote")
}));

vi.mock("electron", () => ({
  BrowserWindow: class {
    private url = "https://example.com";
    private destroyed = false;
    webContents = {
      session: {
        webRequest: { onBeforeRequest: vi.fn() },
        setPermissionRequestHandler: vi.fn(),
        on: vi.fn(),
        removeListener: vi.fn()
      },
      setWindowOpenHandler: vi.fn(),
      on: vi.fn(),
      isLoading: () => false,
      getURL: () => this.url,
      sendInputEvent: vi.fn(),
      capturePage: browserState.captures,
      executeJavaScript: vi.fn(async (script: string) => {
        browserState.scripts.push(script);
        return script.includes("document.title.slice")
          ? { title: "Fixture", text: "page body", links: [] }
          : true;
      })
    };
    async loadURL(url: string): Promise<void> { this.url = url; }
    isDestroyed(): boolean { return this.destroyed; }
    destroy(): void { this.destroyed = true; }
  }
}));

const folders: string[] = [];
afterEach(async () => {
  browserState.captures.mockClear();
  browserState.scripts.length = 0;
  for (const folder of folders.splice(0)) await rm(folder, { recursive: true, force: true });
});

async function runBrowser(steps: unknown) {
  const folder = await mkdtemp(join(tmpdir(), "omni-browser-policy-"));
  folders.push(folder);
  const executor = new ToolExecutor({
    repository: { brainDirectory: () => folder }
  } as never, {} as never);
  const browser = executor as unknown as {
    browser: (
      brainId: string, action: string, args: Record<string, unknown>,
      signal: AbortSignal
    ) => Promise<Record<string, unknown>>;
  };
  const result = await browser.browser(
    "fixture", "task", { url: "https://example.com", steps },
    new AbortController().signal
  );
  return { result, folder };
}

test("URL-only browsing returns page text without an unrequested screenshot", async () => {
  const { result } = await runBrowser([]);
  expect(result).toMatchObject({ finalUrl: "https://example.com/", text: "page body" });
  expect(result).not.toHaveProperty("artifactPath");
  expect(browserState.captures).not.toHaveBeenCalled();
});

test("typing preserves existing input unless clear is explicitly true", async () => {
  await runBrowser([{ kind: "type", selector: "#field", value: "new" }]);
  expect(browserState.scripts.find((script) => script.includes("insertText")))
    .toContain('"clear":false');
  expect(browserState.captures).not.toHaveBeenCalled();
});

test("an explicit clear request is honored", async () => {
  await runBrowser([{ kind: "type", selector: "#field", value: "new", clear: true }]);
  expect(browserState.scripts.find((script) => script.includes("insertText")))
    .toContain('"clear":true');
});

test("an explicit screenshot step produces one artifact", async () => {
  const { result } = await runBrowser([{ kind: "screenshot" }]);
  expect(browserState.captures).toHaveBeenCalledOnce();
  expect(result.artifactPath).toMatch(/\.png$/);
  expect(await readFile(result.artifactPath as string, "utf8"))
    .toBe("screen fixture");
});

test("browser input rejects malformed or overlong steps rather than silently skipping them", async () => {
  await expect(runBrowser([null])).rejects.toThrow(/invalid typed operation/);
  await expect(runBrowser([{ kind: "unknown" }])).rejects.toThrow(/invalid typed operation/);
  await expect(runBrowser([{ kind: "click", selector: "#go", extra: "hidden" }]))
    .rejects.toThrow(/unexpected field/);
  await expect(runBrowser([{ kind: "type", selector: "#go", value: "x", clear: "yes" }]))
    .rejects.toThrow(/type step is invalid/);
  await expect(runBrowser([{ kind: "wait", milliseconds: 60_000 }]))
    .rejects.toThrow(/wait step is invalid/);
  await expect(runBrowser([{ kind: "navigate", url: "http://example.com" }]))
    .rejects.toThrow(/navigation URL is invalid/);
  await expect(runBrowser(Array.from({ length: 201 }, () => ({ kind: "screenshot" }))))
    .rejects.toThrow(/at most 200/);
  expect(browserState.captures).not.toHaveBeenCalled();
});
