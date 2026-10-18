import React from "react";
import { renderToStaticMarkup } from "react-dom/server";
import { describe, expect, it } from "vitest";
import { CallLayout } from "../src/renderer/src/CallLayout";

describe("call and conversation placement", () => {
  it("keeps the capture owner mounted when its controls are hidden", () => {
    const html = renderToStaticMarkup(<CallLayout open={false}
      controls={<section hidden aria-label="Call controls">Real owner stays mounted</section>}>
      <div role="log">Conversation stays visible</div>
    </CallLayout>);
    expect(html).toContain("Real owner stays mounted"); expect(html).toContain("hidden");
    expect(html).toContain("call-layout__conversation"); expect(html).toContain("Conversation stays visible");
    expect(html).not.toContain("is-open");
  });
  it("orders the call controls beside a separate readable conversation surface", () => {
    const html = renderToStaticMarkup(<CallLayout open controls={<section>Call controls</section>}><form>Message input</form></CallLayout>);
    expect(html).toContain("call-layout is-open");
    expect(html.indexOf("Call controls")).toBeLessThan(html.indexOf("Message input"));
    expect(html).not.toContain("iframe"); expect(html).not.toContain("dialog");
  });
});
