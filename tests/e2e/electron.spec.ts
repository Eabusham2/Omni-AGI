import { spawn, type ChildProcess } from "node:child_process";
import { createServer } from "node:net";
import { mkdtemp, rm } from "node:fs/promises";
import { tmpdir } from "node:os";
import { dirname, join, resolve } from "node:path";
import {
  _electron as electron,
  chromium,
  expect,
  test,
  type Browser,
  type ElectronApplication,
  type Page
} from "@playwright/test";

const repository = resolve(process.cwd());

interface RunningApplication {
  page: Page;
  close(): Promise<void>;
}

function environment(dataDirectory: string, installed: boolean): Record<string, string> {
  const inherited = Object.fromEntries(
    Object.entries(process.env).filter(
      (entry): entry is [string, string] => typeof entry[1] === "string"
    )
  );
  return {
    ...inherited,
    NODE_ENV: "test",
    OMNI_AGI_DATA_DIR: dataDirectory,
    OMNI_PACKAGED_ENGINE_REQUIRED: installed ? "1" : "",
    OMNI_PYTHON: installed
      ? ""
      : (process.env.OMNI_PYTHON ??
        (process.platform === "win32" ? "python" : "/usr/bin/python3"))
  };
}

async function reservePort(): Promise<number> {
  return new Promise((resolvePort, reject) => {
    const server = createServer();
    server.unref();
    server.once("error", reject);
    server.listen(0, "127.0.0.1", () => {
      const address = server.address();
      if (!address || typeof address === "string") {
        server.close();
        reject(new Error("Could not reserve a local CDP port."));
        return;
      }
      server.close((error) => {
        if (error) reject(error);
        else resolvePort(address.port);
      });
    });
  });
}

function waitForExit(child: ChildProcess, timeoutMs: number): Promise<boolean> {
  if (child.exitCode !== null || child.signalCode !== null) return Promise.resolve(true);
  return new Promise((resolveExit) => {
    const timer = setTimeout(() => {
      child.off("exit", onExit);
      resolveExit(false);
    }, timeoutMs);
    const onExit = (): void => {
      clearTimeout(timer);
      resolveExit(true);
    };
    child.once("exit", onExit);
  });
}

async function terminateProcessTree(child: ChildProcess): Promise<void> {
  if (child.exitCode !== null || child.signalCode !== null) return;
  if (process.platform === "win32" && child.pid) {
    await new Promise<void>((resolveTermination) => {
      const terminator = spawn(
        "taskkill",
        ["/PID", String(child.pid), "/T", "/F"],
        { stdio: "ignore", windowsHide: true }
      );
      terminator.once("error", () => resolveTermination());
      terminator.once("exit", () => resolveTermination());
    });
    return;
  }
  child.kill("SIGKILL");
  await waitForExit(child, 5_000);
}

async function waitForCdp(
  endpoint: string,
  child: ChildProcess,
  diagnostics: () => string
): Promise<void> {
  const deadline = Date.now() + 90_000;
  while (Date.now() < deadline) {
    if (child.exitCode !== null || child.signalCode !== null) {
      throw new Error(
        `Packaged app exited before CDP became ready (code ${child.exitCode}, signal ${child.signalCode}).\n${diagnostics()}`
      );
    }
    try {
      const response = await fetch(`${endpoint}/json/version`);
      if (response.ok) return;
    } catch {
      // The executable can take several seconds to unpack and start on CI.
    }
    await new Promise((resolveDelay) => setTimeout(resolveDelay, 250));
  }
  throw new Error(`Timed out waiting for packaged app CDP endpoint.\n${diagnostics()}`);
}

async function waitForPage(browser: Browser): Promise<Page> {
  const deadline = Date.now() + 60_000;
  while (Date.now() < deadline) {
    const context = browser.contexts()[0];
    const page = context
      ?.pages()
      .find((candidate) => !candidate.url().startsWith("devtools://"));
    if (page) return page;
    await new Promise((resolveDelay) => setTimeout(resolveDelay, 100));
  }
  throw new Error("Packaged app connected over CDP but did not create a renderer page.");
}

async function launchInstalled(
  executablePath: string,
  dataDirectory: string
): Promise<RunningApplication> {
  const port = await reservePort();
  const endpoint = `http://127.0.0.1:${port}`;
  const child = spawn(
    executablePath,
    [
      `--remote-debugging-port=${port}`,
      "--remote-allow-origins=*",
      `--user-data-dir=${join(dataDirectory, "electron-profile")}`,
      "--disable-gpu"
    ],
    {
      cwd: dirname(executablePath),
      env: environment(dataDirectory, true),
      stdio: ["ignore", "pipe", "pipe"],
      windowsHide: true
    }
  );
  let output = "";
  const capture = (chunk: Buffer): void => {
    output = `${output}${chunk.toString()}`.slice(-8_000);
  };
  child.stdout?.on("data", capture);
  child.stderr?.on("data", capture);

  let browser: Browser | undefined;
  try {
    await waitForCdp(endpoint, child, () => output);
    browser = await chromium.connectOverCDP(endpoint, { timeout: 120_000 });
    const page = await waitForPage(browser);
    let closed = false;
    return {
      page,
      close: async () => {
        if (closed) return;
        closed = true;
        if (!page.isClosed()) {
          await page
            .evaluate(() =>
              (
                window as unknown as {
                  omni: { window: { close(): Promise<void> } };
                }
              ).omni.window.close()
            )
            .catch(() => undefined);
        }
        await browser?.close().catch(() => undefined);
        const exitedCleanly = await waitForExit(child, 30_000);
        if (!exitedCleanly) await terminateProcessTree(child);
      }
    };
  } catch (error) {
    await browser?.close().catch(() => undefined);
    await terminateProcessTree(child);
    throw error;
  }
}

async function launch(dataDirectory: string): Promise<RunningApplication> {
  const installedExecutable = process.env.OMNI_E2E_EXECUTABLE?.trim();
  if (installedExecutable) {
    return launchInstalled(installedExecutable, dataDirectory);
  }
  const application: ElectronApplication = await electron.launch({
    args: [repository, "--disable-gpu"],
    cwd: repository,
    env: environment(dataDirectory, false)
  });
  return {
    page: await application.firstWindow(),
    close: () => application.close()
  };
}

test("build, run, learn, inspect, imagine, download, and restart one persistent brain", async () => {
  const dataDirectory = await mkdtemp(join(tmpdir(), "omni-electron-e2e-"));
  let application: RunningApplication | undefined;
  try {
    application = await launch(dataDirectory);
    let page = application.page;
    await page.waitForLoadState("domcontentloaded");

    await expect(page.getByText("Brain Library").first()).toBeVisible();
    await page.keyboard.press("Tab");
    expect(
      await page.evaluate(() => document.activeElement instanceof HTMLElement)
    ).toBe(true);

    await page.getByRole("button", { name: "Build a new brain" }).click();
    await expect(page.getByRole("heading", { name: "Build a brain" })).toBeVisible();
    await expect(page.locator(".recipe-card")).toHaveCount(6);
    await page
      .getByRole("button", { name: /Senses & tools Give it ways to perceive and act/ })
      .click();
    const audioCard = page.getByRole("button", {
      name: /Audio Hear, encode, and imagine sound/
    });
    const videoCard = page.getByRole("button", {
      name: /Video Learn temporal scenes/
    });
    await audioCard.click();
    await videoCard.click();
    await expect(audioCard).toHaveClass(/is-selected/);
    await expect(videoCard).toHaveClass(/is-selected/);
    await page.getByRole("button", { name: /Review Create the immutable origin/ }).click();
    await page.getByPlaceholder("Name this mind").fill("E2E Cortex");
    await page.getByRole("button", { name: "Create E2E Cortex" }).click();

    const composer = page.getByLabel("Message E2E Cortex");
    await expect(composer).toBeVisible({ timeout: 60_000 });
    await expect(page.getByText(/Python engine/)).toBeVisible({ timeout: 60_000 });
    await composer.fill("/help");
    await page.getByLabel("Send message").click();
    await expect(page.getByText(/Chat commands: \/tool/)).toBeVisible();

    await composer.fill("hello evolving cortex");
    await page.getByLabel("Send message").click();
    await expect(page.locator(".message--human")).toHaveCount(1, {
      timeout: 60_000
    });
    await expect(page.locator(".message--brain")).toHaveCount(1, {
      timeout: 60_000
    });

    await composer.fill("/imagine image direct chat memory");
    await page.getByLabel("Send message").click();
    await expect(
      page.getByText(/modality\.imagine\.generate completed and its visible result/)
    ).toBeVisible({ timeout: 60_000 });
    await expect
      .poll(() =>
        page.evaluate(async () => {
          const api = (
            window as unknown as {
              omni: {
                train: {
                  list(): Promise<Array<{ kind: string; state: string }>>;
                };
              };
            }
          ).omni;
          return (await api.train.list()).find((job) => job.kind === "image")
            ?.state;
        })
      )
      .toBe("complete");
    await expect(page.locator(".message--human").last()).toContainText(
      '"artifactPath"'
    );
    await expect(page.locator(".message--human").last()).toContainText(
      '"state": "complete"'
    );
    await expect(page.locator(".message--human").last()).not.toContainText(
      '"state": "running"'
    );

    await composer.fill("/agent Explore one isolated cobalt-memory association.");
    await page.getByLabel("Send message").click();
    await page.getByRole("button", { name: "Approve exact action" }).click();
    await expect(
      page.getByText(/agent\.fork\.start completed and its visible result/)
    ).toBeVisible({ timeout: 90_000 });

    const surfaces = [
      ["Data & training", "Data & training"],
      ["Brain map", "Brain map"],
      ["Trace & journal", "Trace & journal"],
      ["Tools & permissions", "Tools & permissions"],
      ["Forks & agents", "Forks & agents"]
    ] as const;
    for (const [navigation, heading] of surfaces) {
      await page.getByRole("button", { name: navigation }).click();
      await expect(page.getByRole("heading", { name: heading }).first()).toBeVisible();
    }

    await page.evaluate(() => {
      const state = window as unknown as {
        __omniDownloads: Array<{ href: string; download: string }>;
      };
      state.__omniDownloads = [];
      HTMLAnchorElement.prototype.click = function captureDownload() {
        state.__omniDownloads.push({
          href: this.href,
          download: this.download
        });
      };
    });
    await page.getByRole("button", { name: "Trace & journal" }).click();
    await page.getByRole("button", { name: "Export trace" }).click();
    await expect
      .poll(() =>
        page.evaluate(
          () =>
            (
              window as unknown as {
                __omniDownloads: Array<{ download: string }>;
              }
            ).__omniDownloads[0]?.download
        )
      )
      .toMatch(/\.trace\.json$/);

    await page.getByRole("button", { name: "Imagination" }).click();
    await expect(
      page.getByRole("heading", { name: "Imagination", exact: true })
    ).toBeVisible();
    await page.getByRole("button", { name: "Imagine image" }).click();
    await expect(page.getByAltText("Locally generated image artifact")).toBeVisible({
      timeout: 60_000
    });
    await expect(page.getByLabel("Download output")).toBeEnabled();
    await page.getByLabel("Download output").click();
    await expect
      .poll(() =>
        page.evaluate(
          () =>
            (
              window as unknown as {
                __omniDownloads: Array<{ download: string }>;
              }
            ).__omniDownloads[1]?.download
        )
      )
      .toMatch(/\.png$/);

    await page.getByRole("button", { name: "Audio", exact: true }).click();
    await page.getByRole("button", { name: "Imagine audio" }).click();
    await expect(page.locator("audio")).toBeVisible({ timeout: 60_000 });
    await expect(page.getByLabel("Download output")).toBeEnabled();
    await page.getByLabel("Download output").click();
    await expect
      .poll(() =>
        page.evaluate(
          () =>
            (
              window as unknown as {
                __omniDownloads: Array<{ download: string }>;
              }
            ).__omniDownloads[2]?.download
        )
      )
      .toMatch(/\.wav$/);

    await page.getByRole("button", { name: "Video", exact: true }).click();
    await page.getByRole("button", { name: "Imagine video" }).click();
    await expect(page.getByLabel("Locally generated video artifact")).toBeVisible({
      timeout: 60_000
    });
    await expect(page.getByLabel("Download output")).toBeEnabled();
    await page.getByLabel("Download output").click();
    await expect
      .poll(() =>
        page.evaluate(
          () =>
            (
              window as unknown as {
                __omniDownloads: Array<{ download: string }>;
              }
            ).__omniDownloads[3]?.download
        )
      )
      .toMatch(/\.mp4$/);

    await page.getByRole("button", { name: "Forks & agents" }).click();
    await page.getByRole("button", { name: "Review merge" }).first().click();
    await expect(page.getByText(/MERGE PREVIEW · NO WEIGHTS CHANGED YET/)).toBeVisible();
    await page.getByRole("button", { name: "Merge reviewed overlay" }).click();
    await expect(page.getByText(/Merged \d+ ideas/)).toBeVisible({
      timeout: 60_000
    });

    for (const label of [
      "Conversation",
      "Data & training",
      "Brain map",
      "Trace & journal",
      "Imagination",
      "Tools & permissions",
      "Forks & agents"
    ]) {
      await expect(page.getByRole("button", { name: label })).toHaveAttribute(
        "aria-label",
        label
      );
    }

    await application.close();
    application = await launch(dataDirectory);
    page = application.page;
    await page.waitForLoadState("domcontentloaded");
    await expect(page.getByText("E2E Cortex").first()).toBeVisible({
      timeout: 60_000
    });
    await page.getByText("E2E Cortex").first().click();
    await expect(page.getByLabel("Message E2E Cortex")).toBeVisible({
      timeout: 60_000
    });
    await expect(page.getByText("hello evolving cortex")).toBeVisible();
  } finally {
    await application?.close().catch(() => undefined);
    await rm(dataDirectory, {
      recursive: true,
      force: true,
      maxRetries: 20,
      retryDelay: 250
    });
  }
});
