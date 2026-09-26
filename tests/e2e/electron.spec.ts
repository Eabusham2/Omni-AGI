import { execFile } from "node:child_process";
import { mkdtemp, rm } from "node:fs/promises";
import { tmpdir } from "node:os";
import { dirname, join, resolve } from "node:path";
import {
  _electron as electron,
  expect,
  test,
  type ElectronApplication,
  type Page
} from "@playwright/test";

const repository = resolve(process.cwd());

interface RunningApplication {
  page: Page;
  close(): Promise<void>;
}

function wait(milliseconds: number): Promise<void> {
  return new Promise((resolveWait) => setTimeout(resolveWait, milliseconds));
}

async function waitForProcessExit(
  child: ReturnType<ElectronApplication["process"]>,
  timeout: number
): Promise<boolean> {
  if (child.exitCode !== null || child.signalCode !== null) return true;
  return new Promise((resolveExit) => {
    let finished = false;
    const finish = (exited: boolean): void => {
      if (finished) return;
      finished = true;
      clearTimeout(timer);
      child.off("exit", onExit);
      resolveExit(exited);
    };
    const onExit = (): void => finish(true);
    const timer = setTimeout(() => finish(false), timeout);
    child.once("exit", onExit);
  });
}

async function forceCloseProcessTree(
  child: ReturnType<ElectronApplication["process"]>
): Promise<void> {
  if (child.exitCode !== null || child.signalCode !== null || !child.pid) return;
  if (process.platform === "win32") {
    await new Promise<void>((resolveKill) => {
      execFile(
        "taskkill.exe",
        ["/PID", String(child.pid), "/T", "/F"],
        { windowsHide: true },
        () => resolveKill()
      );
    });
  } else {
    child.kill("SIGKILL");
  }
  if (!(await waitForProcessExit(child, 15_000))) {
    throw new Error(`Electron process ${child.pid} remained alive after forced teardown.`);
  }
}

async function closeElectronApplication(
  application: ElectronApplication
): Promise<void> {
  const child = application.process();
  const gracefulClose = application.close().catch(() => undefined);
  if (!(await waitForProcessExit(child, 5_000))) {
    await forceCloseProcessTree(child);
  }
  await Promise.race([gracefulClose, wait(1_000)]);
  if (child.exitCode === null && child.signalCode === null) {
    await forceCloseProcessTree(child);
  }
}

function environment(dataDirectory: string, installed: boolean): Record<string, string> {
  const inherited = Object.fromEntries(
    Object.entries(process.env).filter(
      (entry): entry is [string, string] =>
        typeof entry[1] === "string" && entry[0] !== "ELECTRON_RUN_AS_NODE"
    )
  );
  return {
    ...inherited,
    NODE_ENV: "test",
    OMNI_E2E_CHAT_DELAY_MS: "3000",
    OMNI_AGI_DATA_DIR: dataDirectory,
    OMNI_PACKAGED_ENGINE_REQUIRED: installed ? "1" : "",
    OMNI_PYTHON: installed
      ? ""
      : (process.env.OMNI_PYTHON ??
        (process.platform === "win32" ? "python" : "/usr/bin/python3"))
  };
}

async function launchInstalled(
  executablePath: string,
  dataDirectory: string
): Promise<RunningApplication> {
  let application: ElectronApplication | undefined;
  try {
    application = await electron.launch({
      executablePath,
      args: [
        `--user-data-dir=${join(dataDirectory, "electron-profile")}`,
        "--disable-gpu"
      ],
      cwd: dirname(executablePath),
      env: environment(dataDirectory, true),
      // Portable tar archives cannot preserve root ownership for
      // chrome-sandbox. This affects only the test launch, not shipped defaults.
      chromiumSandbox: false,
      timeout: 180_000
    });
    const launchedApplication = application;
    const page = await launchedApplication.firstWindow({ timeout: 120_000 });
    let closed = false;
    let closing: Promise<void> | undefined;
    return {
      page,
      close: async () => {
        if (closed) return;
        closing ??= closeElectronApplication(launchedApplication).then(() => {
          closed = true;
        });
        try {
          await closing;
        } finally {
          if (!closed) closing = undefined;
        }
      }
    };
  } catch (error) {
    if (application) {
      await closeElectronApplication(application).catch(() => undefined);
    }
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
    close: () => closeElectronApplication(application)
  };
}

async function sendNaturalMessage(
  page: Page,
  brainName: string,
  message: string,
  navigateWhilePending = false
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
  const optimisticMessage = page
    .locator('.message--human[data-turn-state="pending"]')
    .getByText(message, { exact: true });
  const committedMessage = page
    .locator('.message--human[data-turn-state="committed"]')
    .getByText(message, { exact: true });
  const errorToast = page.locator(".toast");
  // Sending is optimistic: the human bubble must render immediately instead
  // of waiting for the neural worker and its persisted response.
  await expect(optimisticMessage).toBeVisible({ timeout: 1_500 });
  await expect(page.getByLabel("Stop current turn")).toBeVisible();
  if (navigateWhilePending) {
    await page.getByRole("button", { name: "Data & training", exact: true }).click();
    await expect(page.getByRole("heading", { name: "Data & training" })).toBeVisible();
    await page.getByRole("button", { name: "Conversation", exact: true }).click();
    await expect(optimisticMessage).toBeVisible();
    await expect(page.getByLabel("Stop current turn")).toBeVisible();
  }
  // Completion must be observed separately from the optimistic presentation.
  // Otherwise a late model failure can remove the bubble after this helper
  // returns and masquerade as a renderer/Steer state regression.
  await expect(sendButton.or(errorToast).first()).toBeVisible({ timeout: 240_000 });
  if (await errorToast.isVisible().catch(() => false)) {
    throw new Error(
      `Chat failed before persistence: ${(await errorToast.textContent())?.trim() ?? "unknown error"}`
    );
  }
  await expect(committedMessage).toBeVisible({ timeout: 30_000 });
  await expect(optimisticMessage).toHaveCount(0);
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
    await expect(page.getByText("Step 1 of 4", { exact: true })).toBeVisible();
    await expect(
      page.getByRole("heading", { name: "Who are you creating?" })
    ).toBeVisible();
    await expect(page.getByText("Active focus", { exact: true })).toHaveCount(0);
    await expect(page.getByText("Fading scratch trail", { exact: true })).toHaveCount(0);
    await expect(page.getByText("Blank Brain")).toHaveCount(0);
    await expect(page.getByText("Omni Starter")).toHaveCount(0);
    await expect(page.getByText("Retain exact sources")).toHaveCount(0);
    await expect(page.getByText("OmniCortex", { exact: true })).toBeVisible();
    await page.getByPlaceholder("Name this mind").fill("E2E Cortex");
    await page.getByRole("button", { name: "Continue" }).click();

    await expect(page.getByText("Step 2 of 4", { exact: true })).toBeVisible();
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
        page.locator(".modality-card").filter({ hasText: label })
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

    await expect(page.getByText("Step 3 of 4", { exact: true })).toBeVisible();
    await expect(page.getByRole("heading", { name: "Memory & storage" })).toBeVisible();
    await expect(
      page.getByText(/Each whole experience changes what is active now/)
    ).toBeVisible();
    await page.getByText("Research diagnostics", { exact: true }).click();
    await expect(page.getByText("Locally initialized native core")).toBeVisible();
    await page.getByRole("button", { name: "Continue" }).click();

    await expect(page.getByText("Step 4 of 4", { exact: true })).toBeVisible();
    await expect(
      page.getByRole("heading", { name: "Choose system access." })
    ).toBeVisible();
    await expect(page.getByText("Behavioral prompt / RLHF")).toBeVisible();
    await expect(page.getByRole("checkbox", { name: "System access" })).toBeChecked();
    await expect(page.getByRole("checkbox", { name: "Full Authority" })).not.toBeChecked();
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
    await expect(page.getByLabel("Start live voice")).toBeVisible();
    await expect(page.getByLabel("Start live voice")).toBeEnabled();
    await expect(page.getByLabel("Start live voice")).toHaveAttribute(
      "aria-pressed",
      "false"
    );
    await expect(page.getByLabel("Live voice settings")).toHaveCount(0);

    await page.getByRole("button", { name: "Runtime card" }).click();
    const runtimeCard = page.locator(".runtime-card");
    await expect(runtimeCard.getByRole("heading", { name: "Transparent runtime" })).toBeVisible();
    await expect(runtimeCard.getByText("Behavioral system prompt", { exact: true })).toBeVisible();
    await expect(runtimeCard.getByText("Long-term source injection", { exact: true })).toBeVisible();
    await expect(runtimeCard.getByText("Reward model / RLHF", { exact: true })).toBeVisible();
    await expect(runtimeCard.getByText("Mandatory · −1 / 0 / +1", { exact: true })).toBeVisible();
    await expect(runtimeCard.getByText("Active context", { exact: true })).toBeVisible();
    await expect(
      runtimeCard.getByText("Python neural memory", { exact: true })
    ).toBeVisible();
    await expect(runtimeCard.getByText("Made lasting", { exact: true })).toHaveCount(0);
    await expect(runtimeCard.getByText("Working thoughts", { exact: true })).toHaveCount(0);

    const firstDirection = "begin a broad response that I can redirect";
    const queuedDirection = "run this next without interrupting";
    const steeringDirection = "focus only on the memory evidence";
    await composer.fill(firstDirection);
    await page.getByLabel("Send message").click();
    await expect(
      page.locator('.message--human[data-turn-state="pending"]')
        .getByText(firstDirection, { exact: true })
    ).toBeVisible({ timeout: 1_500 });
    await composer.fill(queuedDirection);
    await expect(page.getByLabel("Queue message")).toBeVisible();
    await composer.press("Enter");
    const queuedReceipt = page
      .locator('.message--human[data-turn-state="queued"]')
      .getByText(queuedDirection, { exact: true });
    await expect(queuedReceipt).toBeVisible({ timeout: 2_500 });
    await composer.fill(steeringDirection);
    await expect(page.getByLabel("Steer current turn")).toBeVisible();
    await page.getByLabel("Steer current turn").click();
    const optimisticSteer = page
      .locator('.message--human[data-turn-state="pending"]')
      .getByText(steeringDirection, { exact: true });
    const committedSteer = page
      .locator('.message--human[data-turn-state="committed"]')
      .getByText(steeringDirection, { exact: true });
    await expect(optimisticSteer).toBeVisible({ timeout: 2_500 });
    await expect(committedSteer).toBeVisible({ timeout: 240_000 });
    await expect(optimisticSteer).toHaveCount(0);
    // Queue is a local delivery receipt until the active send settles. Do not
    // assume an ordering relative to Steer; require both actual persisted turns.
    await expect(
      page.locator('.message--human[data-turn-state="committed"]')
        .getByText(queuedDirection, { exact: true })
    ).toBeVisible({ timeout: 240_000 });
    await expect(queuedReceipt).toHaveCount(0);
    await expect(
      page.locator(".message--human").getByText(firstDirection, { exact: true })
    ).toBeVisible();
    await expect(page.getByLabel("Send message")).toBeVisible();

    await sendNaturalMessage(
      page,
      "E2E Cortex",
      "hello, tell me what you notice",
      true
    );
    await expect(
      page.locator(".message--human").filter({
        hasText: "hello, tell me what you notice"
      })
    ).toBeVisible();
    await expect(page.locator(".message--brain").last()).toBeVisible();

    // A newly random-initialized action head is not expected to route an
    // English request to a particular tool before it has learned that skill.
    // Exercise the explicit, deterministic modality path instead of treating
    // an untrained semantic choice as an image-decoder regression.
    await page.getByRole("button", { name: "Imagination", exact: true }).click();
    await expect(
      page.getByRole("heading", { name: "Imagination", exact: true })
    ).toBeVisible();
    await page.locator(".imagination-prompt textarea").fill(
      "An internal scene with luminous memory pathways"
    );
    await page.getByRole("button", { name: "Imagine image", exact: true }).click();
    await expect(page.getByText("FINAL ARTIFACT", { exact: true })).toBeVisible({
      timeout: 240_000
    });
    await expect(page.getByAltText("Locally generated image artifact")).toBeVisible();

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
    // Create a deterministic lineage fork here. Permissioned neural agent
    // proposals are covered at the controller/executor boundary; they must not
    // make this native shell gate depend on an untrained action-head choice.
    await page.getByRole("button", { name: "Fork this mind", exact: true }).click();
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
    await expect(
      page.locator(".evolution-policy").getByText("ask permission", { exact: false })
    ).toBeVisible();
    const evolutionArchive = page.locator(".evolution-archive");
    await expect(evolutionArchive).toBeVisible();
    await expect(
      evolutionArchive
        .getByText("No improvement runs yet", { exact: true })
        .or(evolutionArchive.locator(".evolution-run").first())
        .first()
    ).toBeVisible({ timeout: 120_000 });

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
    await page.getByRole("button", { name: "Duplicate instance E2E Cortex" }).click();
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
    await page.getByRole("button", { name: "Runtime card" }).click();
    await expect(page.getByText("Transparent runtime", { exact: true })).toBeVisible();
    await expect(page.getByText("Active context", { exact: true })).toBeVisible();
    await expect(
      page.getByText("Python neural memory", { exact: true })
    ).toBeVisible();
    await page.getByRole("button", { name: "Imagination", exact: true }).click();
    await expect(page.getByAltText("Saved image imagination").first()).toBeVisible({
      timeout: 120_000
    });
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
