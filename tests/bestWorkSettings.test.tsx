import React from "react";
import { renderToStaticMarkup } from "react-dom/server";
import { describe, expect, it } from "vitest";
import { BestWorkSettings } from "../src/renderer/src/BestWorkSettings";

describe("Best Work settings during foreground work", () => {
  it("keeps editing controls available while the safe-boundary apply is disabled", () => {
    const html = renderToStaticMarkup(<BestWorkSettings
      preferences={{ bestWorkEnabled: false, bestWorkExplorationBranches: 16 }}
      applyBlockedReason="Apply after the current reply."
      onSave={async () => { throw new Error("Static presentation must not call the backend."); }}
    />);
    expect(html.match(/<input[^>]+type="checkbox"[^>]*>/)?.[0]).not.toContain("disabled");
    expect(html.match(/<input[^>]+type="number"[^>]*>/)?.[0]).not.toContain("disabled");
    expect(html.match(/<button[^>]+type="submit"[^>]*>/)?.[0]).toContain("disabled");
    expect(html).not.toContain("fieldset disabled");
  });
});
