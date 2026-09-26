import { readFileSync } from "node:fs";
import { resolve } from "node:path";
import { describe, expect, it } from "vitest";

const main = readFileSync(resolve(process.cwd(), "src/main/index.ts"), "utf8");

describe("desktop shutdown IPC lifecycle", () => {
  it("keeps renderer IPC handlers alive through asynchronous before-quit cleanup", () => {
    const beforeQuit = main.indexOf('app.on("before-quit"');
    const willQuit = main.indexOf('app.on("will-quit"');
    expect(beforeQuit).toBeGreaterThan(0);
    expect(willQuit).toBeGreaterThan(0);
    expect(willQuit).toBeLessThan(beforeQuit);

    const beforeQuitBody = main.slice(beforeQuit, main.indexOf("\n  });", beforeQuit));
    const willQuitBody = main.slice(willQuit, main.indexOf("\n  });", willQuit));
    expect(beforeQuitBody).not.toContain("disposeIpc?.()");
    expect(willQuitBody).toContain("disposeIpc?.()");
    expect(willQuitBody).toContain("disposeIpc = undefined");
  });
});
