import { resolve } from "node:path";
import { _electron as electron, expect, test } from "@playwright/test";

const repository = resolve(process.cwd());
const inheritedEnvironment = Object.fromEntries(
  Object.entries(process.env).filter(
    (entry): entry is [string, string] =>
      typeof entry[1] === "string" && entry[0] !== "ELECTRON_RUN_AS_NODE"
  )
);

test("Open gallery reaches the selected brain's persisted imagination gallery", async () => {
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
    await page.setViewportSize({ width: 390, height: 844 });
    await page
      .locator("article.brain-card")
      .filter({ has: page.getByRole("heading", { name: "Aster", exact: true }) })
      .click();
    await page.getByRole("button", { name: "Imagination", exact: true }).click();

    await page.getByRole("button", { name: "Open gallery", exact: true }).click();
    const gallery = page.getByRole("dialog", { name: "Aster imagination gallery" });
    await expect(gallery).toBeVisible();
    await expect(gallery.getByText("PERSISTED LOCAL OUTPUTS", { exact: true })).toBeVisible();
    await expect(gallery.getByText("No persisted artifacts for Aster", { exact: true }))
      .toBeVisible();
    await expect(gallery.getByText("Design preview", { exact: true })).toBeVisible();

    await page.keyboard.press("Escape");
    await expect(gallery).toBeHidden();
    await page.getByRole("button", { name: "Open gallery", exact: true }).click();
    await gallery.getByRole("button", { name: "Close imagination gallery" }).click();
    await expect(gallery).toBeHidden();

    await page.getByRole("button", { name: "Conversation", exact: true }).click();
    await expect(page.getByLabel("Message Aster")).toBeVisible();
  } finally {
    await application.close();
  }
});
