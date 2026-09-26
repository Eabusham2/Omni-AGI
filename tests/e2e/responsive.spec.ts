import { resolve } from "node:path";
import { _electron as electron, expect, test } from "@playwright/test";

const repository = resolve(process.cwd());
const viewports = [
  { name: "mobile", width: 390, height: 844 },
  { name: "tablet", width: 768, height: 1_024 },
  { name: "short desktop", width: 1_024, height: 600 },
  { name: "desktop", width: 1_366, height: 768 },
  { name: "design target", width: 1_480, height: 940 },
  { name: "ultrawide", width: 2_560, height: 1_080 }
] as const;

test("workspace remains reachable, overflow-safe, and keyboard-accessible at supported viewports", async () => {
  const inheritedEnvironment = Object.fromEntries(
    Object.entries(process.env).filter(
      (entry): entry is [string, string] =>
        typeof entry[1] === "string" && entry[0] !== "ELECTRON_RUN_AS_NODE"
    )
  );
  const application = await electron.launch({
    args: [resolve(repository, "tests/e2e/responsive-main.cjs")],
    env: {
      ...inheritedEnvironment,
      NODE_ENV: "test",
      OMNI_RESPONSIVE_REPOSITORY: repository
    }
  });

  try {
    const page = await application.firstWindow();
    await page.waitForLoadState("domcontentloaded");

    const resize = async (viewport: (typeof viewports)[number]): Promise<void> => {
      // Playwright's renderer viewport is deterministic even when the host OS
      // clamps a physical window to the current monitor's work area.
      await page.setViewportSize({ width: viewport.width, height: viewport.height });
      await expect
        .poll(() => page.evaluate(() => ({ width: innerWidth, height: innerHeight })))
        .toEqual({ width: viewport.width, height: viewport.height });
    };

    for (const viewport of viewports) {
      await test.step(`${viewport.name} library and builder`, async () => {
        await resize(viewport);
        await expect(page.getByRole("button", { name: "Build a new brain" })).toBeVisible();
        expect(
          await page.evaluate(
            () => document.documentElement.scrollWidth - document.documentElement.clientWidth
          )
        ).toBe(0);
        await page.getByRole("button", { name: "Build a new brain" }).click();
        await expect(page.getByText("Step 1 of 4", { exact: true })).toBeVisible();
        await expect(page.getByRole("button", { name: "Brain Library", exact: true })).toBeVisible();
        await expect(page.getByRole("button", { name: /Continue/ })).toBeVisible();
        expect(
          await page.evaluate(
            () => document.documentElement.scrollWidth - document.documentElement.clientWidth
          )
        ).toBe(0);
        await page.getByRole("button", { name: "Brain Library", exact: true }).click();
      });
    }

    await resize(viewports[4]);
    await page
      .locator("article.brain-card")
      .filter({ has: page.getByRole("heading", { name: "Aster", exact: true }) })
      .click();
    await expect(page.getByLabel("Message Aster")).toBeVisible();

    for (const viewport of viewports) {
      await test.step(`${viewport.name}: ${viewport.width}x${viewport.height}`, async () => {
        await resize(viewport);

        await expect(page.getByLabel("Workspace")).toBeVisible();
        await expect(page.getByLabel("Message Aster")).toBeVisible();
        await expect(page.getByLabel("Send message")).toBeVisible();
        const liveVoice = page.getByLabel("Start live voice");
        await expect(liveVoice).toBeVisible();
        await expect(liveVoice).toHaveAttribute("aria-pressed", "false");
        await expect(page.getByLabel("Live voice settings")).toHaveCount(0);
        await expect(page.getByRole("group", { name: "Delivery" })).toHaveCount(0);

        const overflow = await page.evaluate(() => ({
          document: document.documentElement.scrollWidth - document.documentElement.clientWidth,
          body: document.body.scrollWidth - document.body.clientWidth,
          shell: Math.ceil(
            (document.querySelector(".app-shell")?.scrollWidth ?? 0) -
              (document.querySelector(".app-shell")?.clientWidth ?? 0)
          )
        }));
        expect(overflow, `${viewport.name} must not create root horizontal overflow`).toEqual({
          document: 0,
          body: 0,
          shell: 0
        });

        const liveVoiceBounds = await liveVoice.boundingBox();
        expect(liveVoiceBounds, `${viewport.name} live voice must stay reachable`).not.toBeNull();
        expect(liveVoiceBounds!.x).toBeGreaterThanOrEqual(0);
        expect(liveVoiceBounds!.x + liveVoiceBounds!.width).toBeLessThanOrEqual(
          viewport.width + 1
        );

        const workspaceButtons = page.getByLabel("Workspace").getByRole("button");
        await expect(workspaceButtons).toHaveCount(9);
        for (let index = 0; index < 9; index += 1) {
          const button = workspaceButtons.nth(index);
          await button.evaluate((element) =>
            element.scrollIntoView({ block: "nearest", inline: "nearest" })
          );
          const bounds = await button.boundingBox();
          expect(bounds, `workspace action ${index + 1} must be reachable`).not.toBeNull();
          expect(bounds!.x).toBeGreaterThanOrEqual(0);
          expect(bounds!.x + bounds!.width).toBeLessThanOrEqual(viewport.width + 1);
          expect(bounds!.y).toBeGreaterThanOrEqual(0);
          expect(bounds!.y + bounds!.height).toBeLessThanOrEqual(viewport.height + 1);
        }

        await page.getByLabel("Device & runtime").click();
        await expect(
          page.getByRole("heading", { name: "Memory & shared storage", level: 1 })
        ).toBeVisible();
        const ramPolicy = page.getByRole("group", { name: "Omni-wide RAM policy" });
        await expect(ramPolicy.getByRole("button", { name: /^Auto/ })).toBeVisible();
        const advancedCap = ramPolicy.getByRole("button", { name: /^Advanced cap/ });
        await expect(advancedCap).toBeVisible();
        await advancedCap.click();
        await expect(page.getByLabel(/Omni share of safe pool/)).toBeVisible();
        expect(
          await page.evaluate(
            () => document.documentElement.scrollWidth - document.documentElement.clientWidth
          ),
          `${viewport.name} device settings must not create root horizontal overflow`
        ).toBe(0);
        await page.getByRole("button", { name: "Conversation", exact: true }).click();
        await expect(page.getByLabel("Message Aster")).toBeVisible();

        const unlabeledVisibleButtons = await page.locator("button").evaluateAll((buttons) =>
          buttons
            .filter((button) => {
              const bounds = button.getBoundingClientRect();
              const style = getComputedStyle(button);
              return bounds.width > 0 && bounds.height > 0 && style.visibility !== "hidden";
            })
            .filter(
              (button) =>
                !button.getAttribute("aria-label")?.trim() &&
                !button.textContent?.trim() &&
                !button.getAttribute("title")?.trim()
            )
            .map((button) => button.outerHTML.slice(0, 180))
        );
        expect(unlabeledVisibleButtons).toEqual([]);

        if (viewport.width <= 870) {
          const inspectorToggle = page.getByLabel("Open cortex inspector");
          await expect(inspectorToggle).toBeVisible();
          await expect(inspectorToggle).toHaveAttribute("aria-expanded", "false");
          await inspectorToggle.click();
          await expect(page.getByLabel("Cortex and runtime inspector")).toBeVisible();
          await expect(page.getByLabel("Close cortex inspector").first()).toHaveAttribute(
            "aria-expanded",
            "true"
          );
          await page.getByRole("button", { name: "Runtime card" }).click();
          await expect(page.getByText("Transparent runtime", { exact: true })).toBeVisible();
          await page.keyboard.press("Escape");
          await expect(page.getByLabel("Cortex and runtime inspector")).toBeHidden();
        } else {
          await expect(page.getByLabel("Open cortex inspector")).toBeHidden();
          await expect(page.getByLabel("Cortex and runtime inspector")).toBeVisible();
          await page.getByRole("button", { name: "Runtime card" }).click();
          await expect(page.getByText("Transparent runtime", { exact: true })).toBeVisible();
        }

        await page.keyboard.press("Tab");
        expect(
          await page.evaluate(
            () => document.activeElement !== document.body && document.activeElement !== null
          )
        ).toBe(true);
      });
    }
  } finally {
    await application.close();
  }
});
