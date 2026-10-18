import { expect, it } from "vitest";
import { PluginInteractivePanel } from "../src/renderer/src/PluginInteractivePanel";
it("gives a changed brain/package/panel fresh Open state while late prior handles retain their own cleanup", () => {
  const props = { brainId: "brain-a", pluginId: "notes", contentSha256: "a".repeat(64), panelId: "inspect", title: "Inspect" };
  const original = PluginInteractivePanel(props);
  expect(PluginInteractivePanel({ ...props, title: "New display title" }).key).toBe(original.key);
  for (const changed of [{ brainId: "brain-b" }, { pluginId: "other" }, { contentSha256: "b".repeat(64) }, { panelId: "other" }]) {
    expect(PluginInteractivePanel({ ...props, ...changed }).key).not.toBe(original.key);
  }
});
