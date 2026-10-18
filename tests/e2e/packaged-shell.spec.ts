import { mkdtemp, rm, writeFile } from "node:fs/promises";
import type { ChildProcess } from "node:child_process";
import { tmpdir } from "node:os";
import { dirname, join, resolve } from "node:path";
import { _electron as electron, expect, test, type ElectronApplication } from "@playwright/test";
import { closeOwnedPackagedApplication } from "./packaged-shell-lifecycle";

test("packaged desktop opens the library and Build screen without training", async ({}, testInfo) => {
  test.setTimeout(120_000);
  const executable = process.env.OMNI_E2E_EXECUTABLE?.trim();
  test.skip(!executable, "Only packaged artifact smoke supplies OMNI_E2E_EXECUTABLE.");

  const dataDirectory = await mkdtemp(join(tmpdir(), "omni-packaged-shell-"));
  const environment = Object.fromEntries(
    Object.entries(process.env).filter(
      (entry): entry is [string, string] =>
        typeof entry[1] === "string" && entry[0] !== "ELECTRON_RUN_AS_NODE"
    )
  );
  let application: ElectronApplication | undefined;
  let child: ChildProcess | undefined;
  let failure: unknown;
  const started = performance.now(), phases: Array<{ phase: string; elapsedMs: number }> = [];
  let childOutput = "";
  const phase = (name: string) => {
    const value = { phase: name, elapsedMs: Math.round(performance.now() - started) };
    phases.push(value);
    console.info(`PACKAGED_SHELL_PHASE=${JSON.stringify(value)}`);
  };
  const retainOutput = (chunk: unknown) => { childOutput = `${childOutput}${String(chunk)}`.slice(-32_768); };
  try {
    phase("launch-start");
    application = await electron.launch({
      executablePath: resolve(executable!),
      args: [`--user-data-dir=${join(dataDirectory, "electron-profile")}`, "--disable-gpu"],
      cwd: dirname(resolve(executable!)),
      env: {
        ...environment,
        NODE_ENV: "test",
        OMNI_AGI_DATA_DIR: dataDirectory,
        OMNI_PACKAGED_ENGINE_REQUIRED: "1"
      },
      chromiumSandbox: false,
      timeout: 60_000
    });
    // SDK channel objects are disposed by normal close. Keep its exact public
    // child now for output, teardown and the later owned-profile exit check.
    child = application.process();
    phase("launch-complete");
    child.stdout?.on("data", retainOutput);
    child.stderr?.on("data", retainOutput);
    application.on("console", message => { if (message.type() === "error" || message.type() === "warning") retainOutput(`${message.type()}: ${message.text()}\n`); });
    await application.evaluate(({ app, BrowserWindow }) => {
      const observe = (contents: Electron.WebContents) => contents.on("preload-error", (_event, _path, error) => {
        console.error(`PACKAGED_PRELOAD_ERROR=${JSON.stringify({ name: error.name, message: error.message.slice(0, 4096) })}`);
      });
      for (const window of BrowserWindow.getAllWindows()) observe(window.webContents);
      app.on("web-contents-created", (_event, contents) => { observe(contents); });
    });
    phase("first-window-start");
    const page = await application.firstWindow({ timeout: 45_000 });
    phase("first-window-complete");
    page.on("pageerror", error => retainOutput(`${error.name}: ${error.message}\n`));
    page.on("console", message => { if (message.type() === "error" || message.type() === "warning") retainOutput(`${message.type()}: ${message.text()}\n`); });
    await page.waitForLoadState("domcontentloaded");
    // Capture errors that fired while Playwright was still attaching to the
    // already-created window; a missing sandbox preload must not pass as demo.
    for (const message of await page.consoleMessages()) {
      if (message.type() === "error" || /preload|module not found/i.test(message.text())) retainOutput(`${message.type()}: ${message.text()}\n`);
    }
    phase("native-bridge-start");
    expect(await page.evaluate(async () => {
      const api = (window as unknown as { omni?: { brain?: { list?: unknown }; window?: { platform?(): Promise<string> } } }).omni;
      if (!api || typeof api.brain?.list !== "function" || typeof api.window?.platform !== "function") {
        throw new Error("The actual sandbox preload bridge is absent; Design preview is not a packaged-runtime smoke.");
      }
      return api.window.platform();
    })).toBe(process.platform);
    phase("native-bridge-complete");
    phase("library-start");
    await expect(page.getByText("Brain Library").first()).toBeVisible({ timeout: 45_000 });
    await expect(page.getByText("Design preview", { exact: true })).toHaveCount(0);
    phase("library-complete");
    phase("build-navigation-start");
    await page.getByRole("button", { name: "Build a new brain" }).click();
    await expect(page.getByRole("heading", { name: "Build a brain" })).toBeVisible();
    phase("build-navigation-complete");
    // Deliberately stop before Create: CI verifies packaging and navigation,
    // while real neural training belongs to a separate acceptance run.
  } catch (error) { failure = error; }
  finally {
    try { if (application && child) await closeOwnedPackagedApplication(application, child, { observe: phase }); }
    catch (error) { failure = failure ? new AggregateError([failure, error], "Packaged shell navigation and normal shutdown failed.") : error; }
    if (!application || child && (child.exitCode !== null || child.signalCode !== null)) {
      try { await rm(dataDirectory, { recursive: true, force: true }); }
      catch (error) { failure = failure ? new AggregateError([failure, error], "Packaged shell failed and its owned profile could not be removed.") : error; }
    } else phase(child ? "owned-profile-retained-child-still-live" : "owned-profile-retained-no-child-handle");
    if (failure) {
      const diagnosticPath = testInfo.outputPath("packaged-shell-owned-child.log");
      await writeFile(diagnosticPath, `${JSON.stringify({ phases, normalShutdownRequired: true, neuralAcceptanceTested: false })}\n${childOutput}`, "utf8");
      await testInfo.attach("packaged-shell-owned-child", { path: diagnosticPath, contentType: "text/plain" });
    }
  }
  if (failure) throw failure;
});
