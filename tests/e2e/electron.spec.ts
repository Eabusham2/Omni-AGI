import { mkdtemp, rm } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join, resolve } from "node:path";
import { _electron as electron, expect, test, type ElectronApplication } from "@playwright/test";

const repository = resolve(process.cwd());

async function launch(dataDirectory: string): Promise<ElectronApplication> {
  const installedExecutable = process.env.OMNI_E2E_EXECUTABLE?.trim();
  return electron.launch({
    ...(installedExecutable
      ? {
          executablePath: installedExecutable,
          args: ["--disable-gpu"]
        }
      : {
          args: [repository, "--disable-gpu"]
        }),
    cwd: repository,
    env: {
      ...process.env,
      NODE_ENV: "test",
      OMNI_AGI_DATA_DIR: dataDirectory,
      OMNI_PACKAGED_ENGINE_REQUIRED: installedExecutable ? "1" : "",
      OMNI_PYTHON: installedExecutable
        ? ""
        : (process.env.OMNI_PYTHON ??
          (process.platform === "win32" ? "python" : "/usr/bin/python3"))
    }
  });
}

test("build, run, learn, inspect, imagine, download, and restart one persistent brain", async () => {
  const dataDirectory = await mkdtemp(join(tmpdir(), "omni-electron-e2e-"));
  let application: ElectronApplication | undefined;
  try {
    application = await launch(dataDirectory);
    let page = await application.firstWindow();
    await page.waitForLoadState("domcontentloaded");

    await expect(page.getByText("Brain Library").first()).toBeVisible();
    await page.keyboard.press("Tab");
    expect(
      await page.evaluate(() => document.activeElement instanceof HTMLElement)
    ).toBe(true);

    await page.getByRole("button", { name: "Build a new brain" }).click();
    await expect(page.getByRole("heading", { name: "Build a brain" })).toBeVisible();
    await expect(page.locator(".recipe-card")).toHaveCount(6);
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
    page = await application.firstWindow();
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
    await rm(dataDirectory, { recursive: true, force: true });
  }
});
