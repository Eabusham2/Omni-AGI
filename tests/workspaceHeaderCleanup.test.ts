import { readFileSync } from "node:fs";
import { describe, expect, it } from "vitest";

const source = readFileSync(new URL("../src/renderer/src/App.tsx", import.meta.url), "utf8");
const shell = source.slice(source.indexOf("function WorkspaceShell("), source.indexOf("function ChatWorkspace("));

describe("requested workspace header cleanup", () => {
  it("removes ambiguous Pause/Delete and fabricated learning activity without removing scoped actions", () => {
    expect(shell).not.toContain('className="workspace-delete"');
    expect(shell).not.toContain("Pause background training");
    expect(shell).not.toContain("Fast conversation learning active");
    expect(shell).not.toContain("toggleLearning");
    expect(shell).toContain('useState("Checking runtime…")');
    expect(shell).toContain("<InstanceLifecycleControls");
    expect(shell).toContain('className="workspace-duplicate"');
    expect(source.includes('aria-label={`Delete instance ${brain.name}`}')).toBe(true);
    expect(source.includes('id="instance-delete-title"')).toBe(true);
    expect(source.includes('stopDatasetActivity("pause",')).toBe(true);
  });
  it("gives attachment choices an opaque bounded surface and a keyboard exit", () => {
    const styles = readFileSync(new URL("../src/renderer/src/styles.css", import.meta.url), "utf8");
    const menu = styles.match(/\.composer__attach-options \{([^}]+)\}/)?.[1];
    expect(menu).toContain("background: var(--surface-solid)");
    expect(menu).toContain("max-height: min(60dvh, 24rem)");
    expect(menu).toContain("overflow-y: auto");
    const component = readFileSync(new URL("../src/renderer/src/ChatAttachmentMenu.tsx", import.meta.url), "utf8");
    expect(component).toContain('popover="auto"');
    expect(component).toContain('<section className="composer__attach-menu">');
    expect(component).toContain("popoverTarget={id}");
    expect(component).toContain("hidePopover()");
    expect(component).toContain("if (!active");
    expect(source.includes('<ChatAttachmentMenu key={brain.id} active={!hidden}>')).toBe(true);
  });
});
