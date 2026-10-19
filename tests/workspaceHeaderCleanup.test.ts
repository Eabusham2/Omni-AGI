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
    expect(shell).not.toContain('className="workspace-duplicate"');
    expect(source).toMatch(/onDelete:\s*\(summaries: BrainSummary\[\]\)\s*=>\s*void/);
    expect(source).toContain("pendingDeleteTargets.current.shift() ?? null");
    expect(source).toContain("pendingDeleteTargets.current = []");
    expect(source.includes('id="instance-delete-title"')).toBe(true);
    expect(source.includes('stopDatasetActivity("pause",')).toBe(true);
  });
  it("gives attachment choices an opaque bounded surface and a keyboard exit", () => {
    const styles = readFileSync(new URL("../src/renderer/src/styles.css", import.meta.url), "utf8");
    const menu = styles.match(/\.composer__attach-options \{([^}]+)\}/)?.[1];
    expect(menu).toContain("background: var(--surface-solid)");
    expect(menu).toContain("max-height: min(60dvh, 24rem)");
    expect(menu).toContain("overflow-y: auto");
    const component = readFileSync(new URL("../src/renderer/src/UniversalSourcePicker.tsx", import.meta.url), "utf8");
    expect(component).toContain('popover="auto"');
    expect(component).toContain('<div className="universal-source-picker">');
    expect(component).toContain("popoverTarget={id}");
    expect(component).toContain("hidePopover()");
    expect(component).toContain("(!active || disabled)");
    expect(source).toContain('label="Attach" active={!hidden}');
    expect(source).toContain('label="Learn" icon="brain" active={!hidden}');
  });
  it("uses two universal chat source actions without treating Attach as learning", () => {
    const chat = source.slice(source.indexOf("function ChatWorkspace("), source.indexOf("function MessageBubble("));
    expect(chat).toContain("taskAttachmentDraft.stage(selection)");
    expect(chat).toContain("attachExperience(selection)");
    expect(chat).not.toContain("Attach images");
    expect(chat).not.toContain("Learn from video");
  });
  it("moves actual chat preference APIs into chat and retains resource admission outside advanced sections", () => {
    const chat = source.slice(source.indexOf("function ChatWorkspace("), source.indexOf("function MessageBubble("));
    const settings = source.slice(source.indexOf("function DeviceRuntimeWorkspace("), source.indexOf("function TrainingTelemetryMetrics("));
    const build = source.slice(source.indexOf("function SimpleBuildWizard("), source.indexOf("function WorkspaceShell("));
    expect(chat).toContain("window.omni.brain.setActiveMode(brain.id, enabled)");
    expect(chat).toContain("await onActiveModeChange(result)");
    expect(chat).toContain("window.omni.brain.update(brain.id, { ...brain.config, bestWorkEnabled: enabled })");
    expect(chat).toContain("!owner.active || owner.pending || turnActive");
    expect(chat).not.toContain("bestWorkExplorationBranches:");
    expect(settings).not.toContain("setActiveModeEnabled");
    expect(settings).toContain("configWithResourceEnvelope");
    expect(settings).toContain("!resourcePlan?.allowed");
    expect(settings).toContain("contextMaximumTokens");
    expect(settings).toContain("sharedStoragePoolBytes");
    expect(settings).toContain('role={planBlocked || resourceError ? "alert" : "status"}');
    expect(settings).toContain("resourcePlan?.warnings.length");
    expect(build).toContain("modalities: { vision: true, image: true, audio: true, video: true }");
    expect(build).not.toContain("What can it experience first?");
    expect(build).not.toContain("modality-card");
  });
  it("omits fixed architecture rows and captions while retaining measured runtime and access state", () => {
    const chat = source.slice(source.indexOf("function ChatWorkspace("), source.indexOf("function MessageBubble("));
    const runtime = source.slice(source.indexOf("function RuntimeCard("), source.indexOf("function DeviceRuntimeWorkspace("));
    expect(runtime).not.toContain("Reward model / RLHF");
    expect(runtime).not.toContain("Ternary forward paths");
    expect(runtime).not.toContain('["Runtime", brain.config.runtime]');
    expect(runtime).not.toContain("runtime-card__note");
    expect(chat).not.toContain("authoritative count</em>");
    expect(chat).not.toContain("measured, not configured");
    expect(runtime).toContain("workspace.hiddenBehavioralPrompt");
    expect(runtime).toContain("workspace.rawLongTermTextInjected");
    expect(runtime).toContain("telemetry?.corticalWeightsText");
    expect(runtime).toContain("telemetry?.backgroundErrorText");
    expect(runtime).toContain("neuralResourceRuntimeRows(workspace?.runtimeCard)");
    expect(runtime).toContain("workspace.learning.fastNeuralMemory.connectionUpdatesTotal");
    expect(runtime).toContain("evolutionPermission");
    expect(runtime).toContain("enabledTools.length");
    expect(chat).toContain("<NeuralParameterCount accounting={chatParameterAccounting}");
    expect(chat).toContain("recentTrace.driveScores.curiosity");
    expect(chat).toContain("<VoiceRoutingControls");
    expect(chat).toContain("<LivePerceptionPanel");
  });
});
