import { mkdtemp, readFile, rm } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { afterEach, describe, expect, it, vi } from "vitest";
import {
  BackgroundRuntimeController,
  BackgroundRuntimePreferenceStore,
  type BackgroundRuntimeCommands,
  type BackgroundRuntimeScheduler,
  type BackgroundRuntimeView
} from "../src/main/backgroundRuntimeController";
import type { IdleCognitionSchedulerStatus } from "../src/main/idleCognitionScheduler";

const temporaryDirectories: string[] = [];

afterEach(async () => {
  await Promise.all(
    temporaryDirectories.splice(0).map((path) => rm(path, { recursive: true, force: true }))
  );
});

async function preferenceStore(): Promise<{
  store: BackgroundRuntimePreferenceStore;
  path: string;
}> {
  const directory = await mkdtemp(join(tmpdir(), "omni-background-runtime-"));
  temporaryDirectories.push(directory);
  const path = join(directory, "runtime.json");
  return { store: new BackgroundRuntimePreferenceStore(path), path };
}

function status(
  phase: IdleCognitionSchedulerStatus["phase"] = "waiting",
  cancellable = false
): IdleCognitionSchedulerStatus {
  return {
    phase,
    detail: phase === "running" ? "An organic cycle is running." : "Waiting.",
    cancellable,
    maxDutyCycle: 0.12,
    updatedAt: 1
  };
}

function scheduler(initial = status()): BackgroundRuntimeScheduler & {
  emit(next: IdleCognitionSchedulerStatus): void;
  start: ReturnType<typeof vi.fn>;
  stop: ReturnType<typeof vi.fn>;
  cancelActive: ReturnType<typeof vi.fn>;
} {
  let current = initial;
  const listeners = new Set<(value: IdleCognitionSchedulerStatus) => void>();
  return {
    start: vi.fn(),
    stop: vi.fn(),
    status: () => ({ ...current }),
    onStatus: (listener) => {
      listeners.add(listener);
      listener({ ...current });
      return () => listeners.delete(listener);
    },
    cancelActive: vi.fn().mockResolvedValue(false),
    emit: (next) => {
      current = next;
      for (const listener of listeners) listener({ ...next });
    }
  };
}

function tray() {
  let latestView: BackgroundRuntimeView | undefined;
  let latestCommands: BackgroundRuntimeCommands | undefined;
  return {
    render: vi.fn((view: BackgroundRuntimeView, commands: BackgroundRuntimeCommands) => {
      latestView = view;
      latestCommands = commands;
    }),
    destroy: vi.fn(),
    view: () => latestView,
    commands: () => latestCommands
  };
}

describe("BackgroundRuntimeController", () => {
  it("keeps the process resident after the last window closes and exposes bounded policy", async () => {
    let visible = true;
    const preferences = await preferenceStore();
    await preferences.store.initialize();
    await preferences.store.setKeepActiveAfterWindowClose(true);
    const neural = scheduler();
    const surface = tray();
    const quit = vi.fn();
    const controller = new BackgroundRuntimeController({
      store: preferences.store,
      scheduler: neural,
      tray: surface,
      hasVisibleWindows: () => visible,
      openStudio: vi.fn(),
      requestAppQuit: quit
    });

    await controller.initialize();
    expect(neural.start).toHaveBeenCalledOnce();
    visible = false;
    expect(controller.handleLastWindowClosed()).toBe(true);
    expect(controller.snapshot()).toMatchObject({
      mode: "background-resident",
      keepActiveAfterWindowClose: true,
      cancellable: false
    });
    expect(controller.snapshot().resourceStatus).toContain("12% duty");
    expect(controller.snapshot().permissionStatus).toContain("saved Off, Ask, Auto");
    expect(quit).not.toHaveBeenCalled();
  });

  it("updates the tray with running state and cancels only that optional cycle", async () => {
    const preferences = await preferenceStore();
    const neural = scheduler();
    neural.cancelActive.mockResolvedValue(true);
    const surface = tray();
    const controller = new BackgroundRuntimeController({
      store: preferences.store,
      scheduler: neural,
      tray: surface,
      hasVisibleWindows: () => false,
      openStudio: vi.fn(),
      requestAppQuit: vi.fn(),
      resourceStatus: () => "12% cap · foreground and training preempt",
      permissionStatus: () => "Ask permission · 30 second approval window"
    });
    await controller.initialize();

    neural.emit({
      ...status("running", true),
      activeBrainId: "brain-1",
      activeSince: 2
    });
    expect(surface.view()).toMatchObject({
      title: "Background cognition is running",
      cancellable: true,
      resourceStatus: "12% cap · foreground and training preempt",
      permissionStatus: "Ask permission · 30 second approval window"
    });

    surface.commands()?.cancelBackground();
    await vi.waitFor(() => expect(neural.cancelActive).toHaveBeenCalledOnce());
  });

  it("uses explicit Quit to stop, preempt, destroy the tray, and request app cleanup", async () => {
    const preferences = await preferenceStore();
    const neural = scheduler(status("running", true));
    neural.cancelActive.mockResolvedValue(true);
    const surface = tray();
    const requestAppQuit = vi.fn();
    const controller = new BackgroundRuntimeController({
      store: preferences.store,
      scheduler: neural,
      tray: surface,
      hasVisibleWindows: () => false,
      openStudio: vi.fn(),
      requestAppQuit
    });
    await controller.initialize();

    await controller.quitExplicitly();
    expect(neural.stop).toHaveBeenCalledOnce();
    expect(neural.cancelActive).toHaveBeenCalledOnce();
    expect(surface.destroy).toHaveBeenCalledOnce();
    expect(requestAppQuit).toHaveBeenCalledOnce();
    expect(controller.handleLastWindowClosed()).toBe(false);
  });

  it("persists opt-out atomically and quits a windowless resident process", async () => {
    const preferences = await preferenceStore();
    const store = preferences.store;
    const neural = scheduler();
    const surface = tray();
    const requestAppQuit = vi.fn();
    const controller = new BackgroundRuntimeController({
      store,
      scheduler: neural,
      tray: surface,
      hasVisibleWindows: () => false,
      openStudio: vi.fn(),
      requestAppQuit
    });
    await controller.initialize();

    await controller.setKeepActive(false);
    expect(requestAppQuit).toHaveBeenCalledOnce();
    expect(JSON.parse(await readFile(preferences.path, "utf8")))
      .toEqual({ schemaVersion: 1, keepActiveAfterWindowClose: false });
  });

  it("defaults OS background residence off until the user explicitly enables it", async () => {
    const neural = scheduler();
    const controller = new BackgroundRuntimeController({
      store: (await preferenceStore()).store,
      scheduler: neural,
      tray: tray(),
      hasVisibleWindows: () => false,
      openStudio: vi.fn(),
      requestAppQuit: vi.fn()
    });

    await controller.initialize();
    expect(controller.snapshot().keepActiveAfterWindowClose).toBe(false);
    expect(controller.handleLastWindowClosed()).toBe(false);
  });
});
