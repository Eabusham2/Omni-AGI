import { readFileSync } from "node:fs";
import { describe, expect, it } from "vitest";
import ts from "typescript";

describe("intrinsic improvement availability source contract", () => {
  it("normalizes availability to true without using the old user switch", () => {
    const text = readFileSync("src/main/brainRepository.ts", "utf8");
    const source = ts.createSourceFile("brainRepository.ts", text, ts.ScriptTarget.Latest, true);
    const normalizer = source.statements.find(node => ts.isFunctionDeclaration(node) && node.name?.text === "normalizeConfig");
    expect(normalizer).toBeDefined();
    const assignments: ts.BinaryExpression[] = [];
    const visit = (node: ts.Node): void => {
      if (ts.isBinaryExpression(node) && node.operatorToken.kind === ts.SyntaxKind.EqualsToken &&
        ts.isPropertyAccessExpression(node.left) && node.left.getText(source) === "merged.recursiveImprovement") assignments.push(node);
      ts.forEachChild(node, visit);
    };
    visit(normalizer!);
    expect(assignments).toHaveLength(1);
    expect(assignments[0]!.right.kind).toBe(ts.SyntaxKind.TrueKeyword);
    const engine = readFileSync("engine/omni_core/config.py", "utf8");
    expect(engine).toContain('"recursive_improvement": True,');
    expect(engine).not.toContain('raw.get("recursiveImprovement"');
  });

  it("does not derive host authority from intrinsic availability", () => {
    const evolution = readFileSync("src/renderer/src/EvolutionWorkspace.tsx", "utf8");
    expect(evolution).not.toContain("brain.config.recursiveImprovement !== false");
    expect(evolution).toContain('const enabled = permission !== "off";');
    expect(evolution).not.toContain("promotionPolicy: permission");
    const controller = readFileSync("src/main/evolutionController.ts", "utf8");
    expect(controller).toContain('toolId: "source.self-modify"');
    expect(controller).toMatch(/brain\.toolPermissions\?\.\s*find/);
  });
});
