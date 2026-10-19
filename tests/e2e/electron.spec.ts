import { execFile } from "node:child_process";
import { mkdtemp, rm } from "node:fs/promises";
import { tmpdir } from "node:os";
import { dirname, join, resolve } from "node:path";
import type { OmniApi } from "../../src/shared/types";
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
    await page.getByRole("button", { name: "Training & Evolution", exact: true }).click();
    await expect(page.getByRole("heading", { name: "Training & Evolution" })).toBeVisible();
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

async function loadReviewedInstance(page: Page, name: string): Promise<void> {
  const state = page.locator(".instance-lifecycle-controls > span");
  await expect(state).toBeVisible();
  await expect(state).not.toHaveText("Checking instance…");
  const dialog = page.getByRole("dialog", { name: `Load ${name}`, exact: true });
  if (await dialog.isVisible()) {
    const load = dialog.getByRole("button", { name: "Load instance", exact: true });
    await expect(load).toBeEnabled();
    await load.click();
    await expect(dialog).toBeHidden({ timeout: 240_000 });
  }
  await expect(page.getByLabel(`Message ${name}`)).toBeEditable({ timeout: 240_000 });
}

// Deferred full native qualification, not part of the fixed no-brain CI selection.
test("native desktop learns, retains chat across the five workspaces, preserves artifacts, and restarts", async () => {
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
      page.getByRole("heading", { name: "Initial learning" })
    ).toBeVisible();
    await expect(page.getByText(/Text, vision, audio, and video are included/)).toHaveCount(0);
    await expect(page.locator(".modality-card")).toHaveCount(0);
    await page.getByRole("button", { name: "Add source", exact: true }).click();
    await expect(page.getByRole("button", { name: "Files & media", exact: true })).toBeVisible();
    await expect(page.getByRole("button", { name: "Folder", exact: true })).toBeVisible();
    await page.keyboard.press("Escape");
    await page.getByRole("button", { name: "Continue" }).click();

    await expect(page.getByText("Step 3 of 4", { exact: true })).toBeVisible();
    await expect(page.getByRole("heading", { name: "Memory & storage" })).toBeVisible();
    await expect(
      page.getByText("Working memory holds current activity. Learning updates neural state over time.", { exact: true })
    ).toBeVisible();
    await expect(page.getByText("Research diagnostics", { exact: true })).toHaveCount(0);
    await page.getByRole("button", { name: "Continue" }).click();

    await expect(page.getByText("Step 4 of 4", { exact: true })).toBeVisible();
    await expect(
      page.getByRole("heading", { name: "Choose system access." })
    ).toBeVisible();
    await expect(page.getByText("Behavioral prompt / RLHF")).toHaveCount(0);
    await expect(page.getByRole("checkbox", { name: "System access" })).toBeChecked();
    await expect(page.getByRole("checkbox", { name: "Full Authority" })).not.toBeChecked();
    await page.getByRole("button", { name: "Create E2E Cortex" }).click();

    const composer = page.getByLabel("Message E2E Cortex");
    await expect(composer).toBeVisible({ timeout: 300_000 });
    await expect(page.getByText(/Python engine/)).toBeVisible({
      timeout: 120_000
    });

    await loadReviewedInstance(page, "E2E Cortex");
    for (const label of ["Attach", "Learn"]) {
      const picker = page.getByRole("button", { name: label, exact: true });
      await expect(picker).toBeVisible();
      await expect(picker).toBeEnabled();
      await picker.click();
      const choices = page.locator(".universal-source-picker__menu:popover-open");
      await expect(choices.getByRole("button", { name: "Files & media", exact: true })).toBeVisible();
      await expect(choices.getByRole("button", { name: "Folder", exact: true })).toBeVisible();
      await page.keyboard.press("Escape");
    }
    await page.getByRole("button", { name: "Call", exact: true }).click();
    await expect(page.getByLabel("Start live voice")).toBeVisible();
    await expect(page.getByLabel("Start live voice")).toBeEnabled();
    await page.getByRole("button", { name: "Call", exact: true }).click();
    await expect(page.getByLabel("Start live voice")).toBeHidden();

    await page.getByRole("button", { name: "Runtime card" }).click();
    const runtimeCard = page.locator(".runtime-card");
    await expect(runtimeCard.getByRole("heading", { name: "Transparent runtime" })).toBeVisible();
    await expect(runtimeCard.getByText("Behavioral system prompt", { exact: true })).toBeVisible();
    await expect(runtimeCard.getByText("Long-term source injection", { exact: true })).toBeVisible();
    await expect(runtimeCard.getByText("Reward model / RLHF", { exact: true })).toHaveCount(0);
    await expect(runtimeCard.getByText("Mandatory · −1 / 0 / +1", { exact: true })).toHaveCount(0);
    await expect(runtimeCard.getByText("Temporary context", { exact: true })).toBeVisible();
    await expect(
      runtimeCard.getByText("Recorded learning counters", { exact: true })
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
    const queuedReceipt = page.getByRole("region", { name: "Queued messages", exact: true })
      .getByRole("button", { name: "Edit queued message 1", exact: true });
    await expect(queuedReceipt).toBeVisible({ timeout: 2_500 });
    await expect(queuedReceipt).toHaveText(queuedDirection);
    await composer.fill(steeringDirection);
    await composer.press(process.platform === "darwin" ? "Meta+Enter" : "Control+Enter");
    const direction = page.getByRole("region", { name: "Steering directions", exact: true })
      .locator("li").filter({ hasText: steeringDirection });
    await expect(direction).toBeVisible({ timeout: 2_500 });
    await expect(direction.getByText("Completed", { exact: true })).toBeVisible({ timeout: 240_000 });
    await expect(page.locator(".message--human").getByText(steeringDirection, { exact: true })).toHaveCount(0);
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
    // Exercise the real explicit decoder API without resurrecting the removed
    // generator/gallery page. Inline ChatImaginationMedia visibility is covered
    // separately by its actual React fixture, not fabricated native events.
    const mediaJob = await page.evaluate(async () => {
      const api = (window as unknown as { omni: OmniApi }).omni;
      const owner = (await api.brain.list()).find(brain => brain.name === "E2E Cortex");
      if (!owner) throw new Error("The actual created brain is missing.");
      const before = await api.modality.artifacts(owner.id);
      const job = await api.modality.generate({ brainId: owner.id, modality: "image",
        prompt: "An internal scene with luminous memory pathways" });
      return { brainId: owner.id, jobId: job.id, priorArtifactIds: before.artifacts.map(artifact => artifact.id) };
    });
    await expect.poll(() => page.evaluate(async ({ brainId, jobId }) => {
      const api = (window as unknown as { omni: OmniApi }).omni;
      const job = (await api.train.list(brainId)).find(value => value.id === jobId);
      return job?.state === "failed" ? `failed: ${job.error}` : job?.state;
    }, mediaJob), { timeout: 240_000 }).toBe("complete");
    const savedArtifact = await page.evaluate(async ({ brainId, priorArtifactIds }) => {
      const api = (window as unknown as { omni: OmniApi }).omni;
      return (await api.modality.artifacts(brainId)).artifacts.find(artifact =>
        artifact.modality === "image" && !priorArtifactIds.includes(artifact.id));
    }, mediaJob);
    expect(savedArtifact).toMatchObject({ brainId: mediaJob.brainId, modality: "image", available: true });
    expect(savedArtifact?.sha256).toMatch(/^[a-f0-9]{64}$/);
    expect(savedArtifact?.bytes).toBeGreaterThan(0);

    await page.getByRole("button", { name: "Training & Evolution", exact: true }).click();
    await expect(page.getByRole("heading", { name: "Training & Evolution", exact: true })).toBeVisible();
    await page.getByRole("button", { name: "Add source", exact: true }).click();
    const learningChoices = page.locator(".universal-source-picker__menu:popover-open");
    await expect(learningChoices.getByRole("button", { name: "Files & media", exact: true })).toBeVisible();
    await expect(learningChoices.getByRole("button", { name: "Folder", exact: true })).toBeVisible();
    await page.keyboard.press("Escape");

    await page.getByRole("button", { name: "Brain map", exact: true }).click();
    const map = page.getByRole("region", { name: "Whole brain map", exact: true });
    await expect(map.getByRole("heading", { name: "Brain Map", exact: true })).toBeVisible();
    await expect(map.getByRole("button", { name: "Refresh brain map", exact: true })).toBeEnabled({ timeout: 120_000 });
    await expect(map.locator(".brain-map-graph")).toBeVisible({ timeout: 120_000 });

    await page.getByRole("button", { name: "Tools & permissions", exact: true }).click();
    await expect(page.getByRole("heading", { name: "Tools & permissions" })).toBeVisible();
    await expect(page.getByText("Source evolution", { exact: true })).toBeVisible();

    await page.getByRole("button", { name: "Training & Evolution", exact: true }).click();
    const evolution = page.getByRole("region", { name: "Evolution", exact: true });
    await expect(evolution.getByRole("heading", { name: "Evolution", exact: true })).toBeVisible();
    await expect(evolution.getByText("Improve from your goal", { exact: true })).toBeVisible();
    await expect(
      evolution.locator(".evolution-policy").getByText("ask permission", { exact: false })
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
      "Training & Evolution",
      "Brain map",
      "Tools & permissions",
      "Settings"
    ]) {
      await expect(
        page.getByRole("button", { name: label, exact: true })
      ).toHaveAttribute("aria-label", label);
    }
    await expect(page.getByRole("navigation", { name: "Workspace", exact: true }).getByRole("button")).toHaveCount(5);
    await page.getByRole("button", { name: "Settings", exact: true }).click();
    await expect(page.getByRole("heading", { name: "Memory & storage", exact: true })).toBeVisible();

    await page.getByRole("button", { name: "Conversation", exact: true }).click();
    // Copy-on-write remains a native operation; there is no header Duplicate
    // control in the requested five-section interface.
    const duplicateName = await page.evaluate(async brainId => {
      const api = (window as unknown as { omni: OmniApi }).omni;
      return (await api.brain.duplicate(brainId)).name;
    }, mediaJob.brainId);
    expect(duplicateName).toBe("E2E Cortex copy");

    await page
      .getByRole("button", { name: "Brain library", exact: true })
      .click();
    await page.reload();
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
    await loadReviewedInstance(page, "E2E Cortex");
    await expect(
      page.getByText("hello, tell me what you notice", { exact: true })
    ).toBeVisible();
    await page.getByRole("button", { name: "Runtime card" }).click();
    await expect(page.getByText("Transparent runtime", { exact: true })).toBeVisible();
    await expect(page.getByText("Temporary context", { exact: true })).toBeVisible();
    await expect(
      page.getByText("Recorded learning counters", { exact: true })
    ).toBeVisible();
    const retainedArtifact = await page.evaluate(async ({ brainId, artifactId }) => {
      const api = (window as unknown as { omni: OmniApi }).omni;
      return (await api.modality.artifacts(brainId)).artifacts.find(artifact => artifact.id === artifactId);
    }, { brainId: mediaJob.brainId, artifactId: savedArtifact!.id });
    expect(retainedArtifact).toMatchObject({ id: savedArtifact!.id, sha256: savedArtifact!.sha256, available: true });
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
