import { spawn, type ChildProcess } from "node:child_process";
import { mkdtemp, rm } from "node:fs/promises";
import { createServer } from "node:net";
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
  return new Promise((resolvePort, rejectPort) => {
    const server = createServer();
    server.unref();
    server.once("error", rejectPort);
    server.listen(0, "127.0.0.1", () => {
      const address = server.address();
      if (!address || typeof address === "string") {
        server.close();
        rejectPort(new Error("Could not reserve a local CDP port."));
        return;
      }
      server.close((error) => {
        if (error) rejectPort(error);
        else resolvePort(address.port);
      });
    });
  });
}

function waitForExit(child: ChildProcess, timeoutMs: number): Promise<boolean> {
  if (child.exitCode !== null || child.signalCode !== null) {
    return Promise.resolve(true);
  }
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
  const deadline = Date.now() + 120_000;
  while (Date.now() < deadline) {
    if (child.exitCode !== null || child.signalCode !== null) {
      throw new Error(
        `Packaged app exited before CDP became ready (code ${String(child.exitCode)}, signal ${String(child.signalCode)}).\n${diagnostics()}`
      );
    }
    try {
      const response = await fetch(`${endpoint}/json/version`);
      if (response.ok) return;
    } catch {
      // Native and emulated packaged processes can take several seconds to
      // expose their debugging endpoint on hosted runners.
    }
    await new Promise((resolveDelay) => setTimeout(resolveDelay, 250));
  }
  throw new Error(`Timed out waiting for packaged app CDP endpoint.\n${diagnostics()}`);
}

async function waitForPage(browser: Browser): Promise<Page> {
  const deadline = Date.now() + 120_000;
  while (Date.now() < deadline) {
    const page = browser
      .contexts()
      .flatMap((context) => context.pages())
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
      "--disable-gpu",
      // Portable archives cannot preserve root ownership for chrome-sandbox.
      // This affects only the package test launch, not shipped defaults.
      "--no-sandbox"
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
    output = `${output}${chunk.toString("utf8")}`.slice(-16_000);
  };
  child.stdout?.on("data", capture);
  child.stderr?.on("data", capture);

  let browser: Browser | undefined;
  try {
    await waitForCdp(endpoint, child, () => output);
    browser = await chromium.connectOverCDP(endpoint, { timeout: 120_000 });
    const page = await waitForPage(browser);
    try {
      await page.waitForLoadState("domcontentloaded", { timeout: 120_000 });
    } catch (error) {
      throw new Error(
        `Packaged renderer closed before DOMContentLoaded (code ${String(child.exitCode)}, signal ${String(child.signalCode)}).\n${output}\n${error instanceof Error ? error.message : String(error)}`
      );
    }
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

async function sendNaturalMessage(
  page: Page,
  brainName: string,
  message: string
): Promise<void> {
  const composer = page.getByLabel(`Message ${brainName}`);
  const sendButton = page.getByLabel("Send message");
  const staleToastDismiss = page.getByRole("button", { name: "Dismiss" });
  if (await staleToastDismiss.isVisible().catch(() => false)) {
    await staleToastDismiss.click();
  }
  await expect(sendButton).toBeVisible({ timeout: 240_000 });
  await composer.fill(message);
  await expect(sendButton).toBeEnabled({ timeout: 240_000 });
  await sendButton.click();
  const persistedMessage = page
    .locator(".message--human")
    .getByText(message, { exact: true });
  const errorToast = page.locator(".toast");
  await expect(persistedMessage.or(errorToast).first()).toBeVisible({
    timeout: 240_000
  });
  if (await errorToast.isVisible().catch(() => false)) {
    throw new Error(
      `Chat failed before persistence: ${(await errorToast.textContent())?.trim() ?? "unknown error"}`
    );
  }
  await expect(persistedMessage).toBeVisible();
  await expect(page.getByLabel("Send message")).toBeVisible({
    timeout: 240_000
  });
}

test("stable v1 builds, runs, acts naturally, exposes every workspace, duplicates, and restarts", async () => {
  test.setTimeout(900_000);
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
    await expect(page.getByText("Step 1 of 4")).toBeVisible();
    await expect(
      page.getByRole("heading", { name: "Who are you creating?" })
    ).toBeVisible();
    await expect(
      page.getByRole("button", { name: /Omni Starter Recommended/ })
    ).toHaveClass(/is-selected/);
    await expect(page.locator('input[type="range"]')).toHaveCount(0);
    await page.getByPlaceholder("Name this mind").fill("E2E Cortex");
    await page.getByRole("button", { name: "Continue" }).click();

    await expect(page.getByText("Step 2 of 4")).toBeVisible();
    await expect(
      page.getByRole("heading", { name: "How should learning live?" })
    ).toBeVisible();
    for (const label of [
      "Learn continuously",
      "Retain exact sources",
      "Extended working memory",
      "Recursive improvement"
    ]) {
      await expect(page.getByText(label, { exact: true })).toBeVisible();
    }
    await expect(page.locator('input[type="range"]')).toHaveCount(0);
    await page.getByRole("button", { name: "Continue" }).click();

    await expect(page.getByText("Step 3 of 4")).toBeVisible();
    await expect(
      page.getByRole("heading", { name: "What can it experience first?" })
    ).toBeVisible();
    for (const label of [
      "Vision",
      "Image imagination",
      "Audio",
      "Video"
    ]) {
      await expect(
        page.locator("button.modality-card").filter({ hasText: label })
      ).toHaveClass(/is-selected/);
    }
    for (const label of [
      "Files & datasets",
      "Images",
      "Audio",
      "Video",
      "Whole folder"
    ]) {
      await expect(
        page.getByRole("button", { name: label, exact: true })
      ).toBeVisible();
    }
    await page.getByRole("button", { name: "Continue" }).click();

    await expect(page.getByText("Step 4 of 4")).toBeVisible();
    await expect(
      page.getByRole("heading", { name: "Choose action permissions." })
    ).toBeVisible();
    await expect(page.getByText("Initially trained Omni Starter")).toBeVisible();
    await expect(page.getByText("Behavioral prompt / RLHF")).toBeVisible();
    const evolutionPermission = page
      .locator(".tool-row")
      .filter({ hasText: "Recursive improvement" });
    await expect(
      evolutionPermission.getByRole("button", { name: "Ask", exact: true })
    ).toHaveClass(/is-active/);
    await page.getByRole("button", { name: "Create E2E Cortex" }).click();

    const composer = page.getByLabel("Message E2E Cortex");
    await expect(composer).toBeVisible({ timeout: 300_000 });
    await expect(page.getByText(/Python engine/)).toBeVisible({
      timeout: 120_000
    });

    for (const label of [
      "Upload files and datasets to learn",
      "Upload images to learn",
      "Upload audio to learn",
      "Upload video to learn",
      "Upload a folder to learn"
    ]) {
      await expect(page.getByLabel(label)).toBeVisible();
      await expect(page.getByLabel(label)).toBeEnabled();
    }

    await page.getByRole("button", { name: "Runtime card" }).click();
    const runtimeCard = page.locator(".runtime-card");
    await expect(runtimeCard.getByRole("heading", { name: "Transparent runtime" })).toBeVisible();
    await expect(runtimeCard.getByText("Behavioral system prompt", { exact: true })).toBeVisible();
    await expect(runtimeCard.getByText("Long-term source injection", { exact: true })).toBeVisible();
    await expect(runtimeCard.getByText("Reward model / RLHF", { exact: true })).toBeVisible();
    await expect(runtimeCard.getByText("Mandatory · −1 / 0 / +1", { exact: true })).toBeVisible();
    await expect(runtimeCard.getByText("Current context", { exact: true })).toBeVisible();
    await expect(
      runtimeCard.getByText("Latent assembly workspace", { exact: true })
    ).toBeVisible();

    await sendNaturalMessage(page, "E2E Cortex", "hello, tell me what you notice");
    await expect(
      page.locator(".message--human").filter({
        hasText: "hello, tell me what you notice"
      })
    ).toBeVisible();
    await expect(page.locator(".message--brain").last()).toBeVisible();

    await sendNaturalMessage(
      page,
      "E2E Cortex",
      "make an image from this internal scene"
    );
    const imaginationAction = page
      .locator(".chat-action-card")
      .filter({ hasText: "modality.imagine" })
      .last();
    await expect(imaginationAction).toContainText("complete", {
      timeout: 240_000
    });
    await expect(
      imaginationAction.locator("img, audio, video").first()
    ).toBeVisible({ timeout: 120_000 });

    await sendNaturalMessage(
      page,
      "E2E Cortex",
      "fork agents to investigate these independent parts"
    );
    await expect(
      page.getByRole("button", { name: "Approve exact action" })
    ).toBeVisible({ timeout: 120_000 });
    const agentActions = page
      .locator(".chat-action-card")
      .filter({ hasText: "agent.fork" });
    const approvedAgentAction = agentActions.nth((await agentActions.count()) - 1);
    await expect(approvedAgentAction).toContainText("approval required");
    await page.getByRole("button", { name: "Approve exact action" }).click();
    await expect(approvedAgentAction).toContainText("complete", {
      timeout: 180_000
    });
    await expect(
      page.getByText(
        /agent\.fork\.start completed and its visible result entered working experience/
      )
    ).toBeVisible({ timeout: 180_000 });

    await page.getByRole("button", { name: "Data & training", exact: true }).click();
    await expect(page.getByRole("heading", { name: "Data & training" })).toBeVisible();
    for (const label of ["Files & datasets", "Images", "Audio", "Video", "Whole folder"]) {
      await expect(
        page.getByRole("button", { name: label, exact: true })
      ).toBeVisible();
    }

    await page.getByRole("button", { name: "Brain map", exact: true }).click();
    await expect(page.getByRole("heading", { name: "Brain map" })).toBeVisible();
    await expect(page.getByText(/no display ceiling/i)).toBeVisible({
      timeout: 120_000
    });

    await page.getByRole("button", { name: "Trace & journal", exact: true }).click();
    await expect(page.getByRole("heading", { name: "Trace & journal" })).toBeVisible();
    await expect(
      page.getByRole("button", { name: "Operational trace", exact: true })
    ).toBeVisible();

    await page.getByRole("button", { name: "Tools & permissions", exact: true }).click();
    await expect(page.getByRole("heading", { name: "Tools & permissions" })).toBeVisible();
    await expect(page.getByText("Source evolution", { exact: true })).toBeVisible();

    await page.getByRole("button", { name: "Forks & agents", exact: true }).click();
    await expect(page.getByRole("heading", { name: "Forks & agents" })).toBeVisible();
    await expect(page.getByRole("button", { name: "Review merge" }).first()).toBeVisible({
      timeout: 120_000
    });

    await page.getByRole("button", { name: "Imagination", exact: true }).click();
    await expect(
      page.getByRole("heading", { name: "Imagination", exact: true })
    ).toBeVisible();
    await expect(page.getByRole("button", { name: "Imagine image" })).toBeVisible();
    await expect(page.getByRole("button", { name: "Audio", exact: true })).toBeVisible();
    await expect(page.getByRole("button", { name: "Video", exact: true })).toBeVisible();

    await page.getByRole("button", { name: "Evolution", exact: true }).click();
    await expect(page.getByRole("heading", { name: "Evolution lab" })).toBeVisible();
    await expect(page.getByText("New isolated candidate", { exact: true })).toBeVisible();
    await expect(page.getByText("ask permission", { exact: false })).toBeVisible();
    await expect(page.getByText("No improvement runs yet", { exact: true })).toBeVisible();

    for (const label of [
      "Conversation",
      "Data & training",
      "Brain map",
      "Trace & journal",
      "Imagination",
      "Tools & permissions",
      "Forks & agents",
      "Evolution"
    ]) {
      await expect(
        page.getByRole("button", { name: label, exact: true })
      ).toHaveAttribute("aria-label", label);
    }

    await page.getByRole("button", { name: "Conversation", exact: true }).click();
    await page.getByRole("button", { name: "Duplicate E2E Cortex" }).click();
    await expect(
      page.getByText(/E2E Cortex copy created with copy-on-write neural storage/)
    ).toBeVisible({ timeout: 180_000 });

    await page
      .getByRole("button", { name: "Brain library", exact: true })
      .click();
    await expect(
      page.getByRole("heading", { name: "E2E Cortex", exact: true })
    ).toBeVisible();
    await expect(
      page.getByRole("heading", { name: "E2E Cortex copy", exact: true })
    ).toBeVisible();

    await application.close();
    application = undefined;
    application = await launch(dataDirectory);
    page = application.page;
    await page.waitForLoadState("domcontentloaded");
    await expect(
      page.getByRole("heading", { name: "E2E Cortex", exact: true })
    ).toBeVisible({
      timeout: 120_000
    });
    await expect(
      page.getByRole("heading", { name: "E2E Cortex copy", exact: true })
    ).toBeVisible();
    await page
      .locator("article.brain-card")
      .filter({
        has: page.getByRole("heading", { name: "E2E Cortex", exact: true })
      })
      .click();
    await expect(page.getByLabel("Message E2E Cortex")).toBeVisible({
      timeout: 120_000
    });
    await expect(
      page.getByText("hello, tell me what you notice", { exact: true })
    ).toBeVisible();
    await expect(
      page.getByText("make an image from this internal scene", { exact: true })
    ).toBeVisible();
    await page.getByRole("button", { name: "Runtime card" }).click();
    await expect(page.getByText("Transparent runtime", { exact: true })).toBeVisible();
    await expect(page.getByText("Current context", { exact: true })).toBeVisible();
    await expect(
      page.getByText("Latent assembly workspace", { exact: true })
    ).toBeVisible();
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
