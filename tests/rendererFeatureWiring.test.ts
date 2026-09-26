import { readFileSync } from "node:fs";
import { resolve } from "node:path";
import { describe, expect, it } from "vitest";

const source = (path: string): string =>
  readFileSync(resolve(process.cwd(), path), "utf8");

const app = source("src/renderer/src/App.tsx");
const preload = source("src/preload/index.ts");
const main = source("src/main/index.ts");
const ipc = source("src/main/ipc.ts");
const actionController = source("src/main/chatActionController.ts");
const actionPresentation = source("src/renderer/src/chatActionPresentation.ts");

describe("renderer release feature wiring", () => {
  it("provides visible create, list, and confirmed restore recovery controls", () => {
    expect(app).toContain("const listSnapshots = window.omni?.brain?.listSnapshots;");
    expect(app).toContain("await listSnapshots(brain.id)");
    expect(app).toMatch(
      /window\.omni\.brain\.snapshot\(\s*brain\.id,\s*label\.trim\(\) \|\| undefined,\s*operationId\s*\)/
    );
    expect(app).toMatch(
      /window\.omni\.brain\.restoreSnapshot\(\s*brain\.id,\s*snapshot\.id,\s*operationId\s*\)/
    );
    expect(app).toContain("const confirmed = window.confirm(");
    expect(app).toContain('role="status" aria-live="polite">{status}</p>');
    expect(app).toContain("setStatus(recoveryPointCreationErrorMessage(error));");
    expect(app).toContain(
      "storage-operation-card storage-operation-card--${storageOperation.state}"
    );
    expect(app).toContain(
      '!["complete", "failed", "cancelled"].includes(storageOperation.state)'
    );
  });

  it("refreshes the Library and reports OS bundle import success and failure", () => {
    expect(app).toContain("window.omni.brain.onImported((imported) =>");
    expect(app).toContain("window.omni.brain.onImportFailed((failure) =>");
    expect(app).toContain("setSummaries(next)");
    expect(preload).toContain("ipcRenderer.on(IPC.brain.importFailed, wrapped)");
    expect(main).toContain("mainWindow.webContents.send(IPC.brain.importFailed");
  });

  it("keeps catalog inspection without letting a recipe replace the native build", () => {
    expect(app).toContain('createPresetConfig("whole-brain", "Nova")');
    expect(app).toContain("New minds always initialize the native OmniCortex");
    expect(app).toContain("Inspect source");
    expect(app).not.toContain("initialRecipe={pendingBuildRecipe}");
    expect(app).not.toContain("onBuildRecipe(recipe)");
    expect(app).not.toContain("window.omni!.catalog.loadRecipeEntry(entry.id)");
    expect(app).not.toContain("window.omni!.catalog.loadRecipeFile()");
    expect(app).not.toContain("window.omni!.catalog.loadRecipeUrl({");
  });

  it("exposes credential revocation and MCP schema refresh", () => {
    expect(app).toContain("window.omni.teacher.removeCredential(teacherProvider)");
    expect(app).toContain("window.omni.mcp.refresh(brain.id, serverId)");
    expect(app).toContain("Remove key");
    expect(app).toContain("Refresh</Button>");
  });

  it("ships a valid history test query and truthful map salience wording", () => {
    expect(app).toContain('args: { query: "recent conversation", limit: 50');
    expect(app).not.toContain('args: { query: "", limit: 50');
    expect(app).toContain('useState<"all" | "active" | "salient">');
    expect(app).toContain("High importance in detailed views; high peak activation in clustered views");
    expect(app).not.toContain('filter === "important" && cluster.maxActivation');
  });

  it("uses the atomic one-brain Active Mode contract and surfaces switches", () => {
    expect(app).toContain("window.omni.brain.setActiveMode(brain.id, enabled)");
    expect(app).toContain("result.deactivated.map((item) => item.name)");
    expect(app).toContain("setSummaries(next)");
    expect(app).toContain("Enabling this identity atomically switches Active Mode off");
  });

  it("keeps initial-learning Pause and Cancel controls plus paused Resume", () => {
    const runningStart = app.indexOf("initializationRunning ? (");
    const failedStart = app.indexOf('activeBrain?.readiness.state === "failed"', runningStart);
    const runningSurface = app.slice(runningStart, failedStart);
    expect(runningSurface).toContain("Pause training");
    expect(runningSurface).toContain("Cancel training");
    expect(source("src/renderer/src/initializationRecoveryPresentation.ts")).toContain(
      'actionLabel: "Resume training"'
    );
  });

  it("resolves approved chat actions through typed main-owned learning only", () => {
    const approvalStart = app.indexOf("const approvePendingTool = async");
    const approvalEnd = app.indexOf("const steerCurrentTurn", approvalStart);
    const approvalFlow = app.slice(approvalStart, approvalEnd);
    expect(approvalStart).toBeGreaterThan(-1);
    expect(approvalFlow).toContain("window.omni.chat.approveAction({");
    expect(approvalFlow).not.toContain("window.omni.chat.send(");
    expect(approvalFlow).not.toContain("window.omni.tool.execute(");
    expect(app).not.toContain("visibleToolResultInput");
    expect(actionPresentation).not.toContain("VisibleToolResultInput");
    expect(preload).toContain("invoke(IPC.chat.approveAction, request)");
    expect(ipc).toContain("handle(IPC.chat.approveAction");
    expect(ipc).toContain("return actions.approveAction({");
    expect(actionController).toContain("async approveAction(");
    expect(actionController).toContain("this.service.learnStructuredExperience(request.brainId");
    const controllerApprovalStart = actionController.indexOf("async approveAction(");
    const controllerApprovalEnd = actionController.indexOf("async send(", controllerApprovalStart);
    expect(actionController.slice(controllerApprovalStart, controllerApprovalEnd))
      .not.toContain("this.service.chat(");
  });
});
