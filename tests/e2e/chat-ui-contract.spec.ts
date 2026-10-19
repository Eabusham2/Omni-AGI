import { resolve } from "node:path";
import { _electron as electron, expect, test } from "@playwright/test";

const repository = resolve(process.cwd());
const inheritedEnvironment = Object.fromEntries(
  Object.entries(process.env).filter(
    (entry): entry is [string, string] =>
      typeof entry[1] === "string" && entry[0] !== "ELECTRON_RUN_AS_NODE"
  )
);

test("offline Electron design preview preserves drafts and navigation without pretending to chat", async () => {
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

    const composer = page.getByLabel("Message Aster");
    const sendButton = page.getByLabel("Send message");
    await expect(composer).toBeVisible();
    await expect(page.getByRole("group", { name: "Message controls", exact: true }).getByRole("button")).toHaveCount(1);
    await expect(page.getByRole("button", { name: "Attach", exact: true })).toBeVisible();
    await expect(page.getByRole("button", { name: "Learn", exact: true })).toBeVisible();
    await expect(page.getByText(
      "Preview only — no brain is connected. Sending and learning are unavailable."
    )).toBeVisible();

    const humanCount = await page.locator(".message--human").count();
    const brainCount = await page.locator(".message--brain").count();
    const draft = "This draft must not become a simulated neural reply";
    await composer.fill(draft);
    await expect(sendButton).toBeDisabled();
    await expect(page.locator(".composer__input-count")).toHaveAttribute("aria-label", /input tokens/);
    await expect(page.locator(".composer__suggestions")).toHaveCount(0);

    // The keyboard path may be invoked even though the Send control is disabled.
    // It must keep the draft and refuse to create a fake pending or settled turn.
    await composer.press("Enter");
    await expect(page.getByText(
      "Neural engine unavailable in this design preview. Your draft was not sent."
    )).toBeVisible();
    await expect(composer).toHaveValue(draft);
    await expect(page.locator(".message--human")).toHaveCount(humanCount);
    await expect(page.locator(".message--brain")).toHaveCount(brainCount);
    await expect(page.getByLabel("Queue message")).toHaveCount(0);
    await expect(page.getByLabel("Steer current turn")).toHaveCount(0);
    await expect(page.getByLabel("Stop current turn")).toHaveCount(0);

    await page.getByRole("button", { name: "Training & Evolution", exact: true }).click();
    await expect(page.getByRole("heading", { name: "Training & Evolution" })).toBeVisible();
    await expect(page.locator(".workspace-header__chat-status")).toHaveCount(0);
    await page.getByRole("button", { name: "Back to conversation", exact: true }).click();
    await expect(composer).toBeVisible();

    await page.getByRole("button", { name: "Brain map", exact: true }).click();
    const wholeMap = page.getByRole("region", { name: "Whole brain map", exact: true });
    await expect(wholeMap.getByRole("heading", { name: "Brain Map", exact: true })).toBeVisible();
    await expect(wholeMap.getByText("Brain Map is unavailable in this UI-only preview.", { exact: true })).toBeVisible();
    await expect(wholeMap.getByRole("checkbox")).toHaveCount(0);
    await expect(wholeMap.getByRole("button", { name: "Refresh brain map", exact: true })).toBeDisabled();
    await expect(wholeMap.locator(".brain-map-graph")).toHaveCount(0);
    await expect(page.locator(".workspace-header__chat-status")).toHaveCount(0);
    await wholeMap.getByRole("button", { name: "Back to chat", exact: true }).click();
    await expect(composer).toBeVisible();
    await expect(composer).toHaveValue(draft);
    await expect(sendButton).toBeDisabled();
    await expect(page.locator(".message--human")).toHaveCount(humanCount);
    await expect(page.locator(".message--brain")).toHaveCount(brainCount);
    await expect(page.locator(".pondering-label")).toHaveCount(0);
    await expect(page.locator(".response-token-counter")).toHaveCount(0);
  } finally {
    await application.close();
  }
});
