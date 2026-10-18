import { expect, it } from "vitest";
import { libraryRuntimePresentation } from "../src/renderer/src/libraryRuntimePresentation";

it("labels the unconnected interface honestly rather than claiming compute is ready", () => {
  expect(libraryRuntimePresentation(true)).toEqual({ title: "Interface preview", detail: "No neural engine is connected.", badge: "No trained brain" });
});

it("does not infer an online worker, hardware measurement or loaded model from the desktop bridge", () => {
  const notice = libraryRuntimePresentation(false);
  expect(notice).toEqual({ title: "Local workspace", detail: "Instances are loaded and checked separately." });
  expect(notice.badge).toBeUndefined();
  expect(JSON.stringify(notice)).not.toMatch(/online|ready|scaling enabled/i);
});
