import { readFileSync } from "node:fs";
import { resolve } from "node:path";
import { runInNewContext } from "node:vm";
import ts from "typescript";
import { describe, expect, it, vi } from "vitest";

const main = readFileSync(resolve(process.cwd(), "src/main/index.ts"), "utf8");

describe("desktop shutdown IPC lifecycle", () => {
  it.each([true, false])("retires a destroyed window from its captured identity without reading destroyed webContents (current=%s)", current => {
    const source = ts.createSourceFile("index.ts", main, ts.ScriptTarget.Latest, true, ts.ScriptKind.TS);
    let listener: ts.ArrowFunction | undefined;
    let captured: ts.VariableDeclaration | undefined;
    const visit = (node: ts.Node): void => {
      if (ts.isVariableDeclaration(node) && node.name.getText(source) === "studioWebContentsId") captured = node;
      if (ts.isCallExpression(node) && node.expression.getText(source) === "window.on"
        && ts.isStringLiteral(node.arguments[0]!) && node.arguments[0]!.text === "closed"
        && ts.isArrowFunction(node.arguments[1]!)) listener = node.arguments[1] as ts.ArrowFunction;
      ts.forEachChild(node, visit);
    };
    visit(source);
    expect(captured?.initializer?.getText(source)).toBe("window.webContents.id");
    expect(listener).toBeDefined();
    const resetOwner = vi.fn(); let destroyedReads = 0;
    const window = { get webContents(): never { destroyedReads += 1; throw new TypeError("Object has been destroyed"); } };
    const scope = { window, mainWindow: window as unknown, studioWebContentsId: 41,
      trustedStudioWebContentsId: current ? 41 : 99, touchBarController: { resetOwner } };
    const compiled = ts.transpileModule(`(${listener!.getText(source)})`, {
      compilerOptions: { target: ts.ScriptTarget.ES2022, module: ts.ModuleKind.None }
    }).outputText;
    expect(() => runInNewContext(compiled, scope)()).not.toThrow();
    expect(destroyedReads).toBe(0);
    expect(scope.trustedStudioWebContentsId).toBe(current ? undefined : 99);
    expect(scope.mainWindow).toBeUndefined();
    expect(resetOwner).toHaveBeenCalledTimes(1);
  });

  it("keeps renderer IPC handlers alive through asynchronous before-quit cleanup", () => {
    const beforeQuit = main.indexOf('app.on("before-quit"');
    const willQuit = main.indexOf('app.on("will-quit"');
    expect(beforeQuit).toBeGreaterThan(0);
    expect(willQuit).toBeGreaterThan(0);
    expect(willQuit).toBeLessThan(beforeQuit);

    const beforeQuitBody = main.slice(beforeQuit, main.indexOf("\n  });", beforeQuit));
    const willQuitBody = main.slice(willQuit, main.indexOf("\n  });", willQuit));
    expect(beforeQuitBody).not.toContain("disposeIpc?.()");
    expect(willQuitBody).toContain("disposeIpc?.()");
    expect(willQuitBody).toContain("disposeIpc = undefined");
  });
});
