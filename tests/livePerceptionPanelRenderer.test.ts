import { readFileSync } from "node:fs";
import { resolve } from "node:path";
import { describe, expect, it } from "vitest";

const panel = readFileSync(
  resolve(process.cwd(), "src/renderer/src/LivePerceptionPanel.tsx"),
  "utf8"
);
const app = readFileSync(
  resolve(process.cwd(), "src/renderer/src/App.tsx"),
  "utf8"
);
const styles = readFileSync(
  resolve(process.cwd(), "src/renderer/src/styles.css"),
  "utf8"
);

describe("Live Perception renderer surface", () => {
  it("stays reachable without live voice and binds only to typed observation controls", () => {
    expect(app).toContain(
      '</>\n          ) : null}\n          <LivePerceptionPanel brain={brain} />'
    );
    expect(panel).toContain("controller.requestConfigure");
    expect(panel).toContain("controller.requestSnapshot");
    expect(panel).toContain("controllerRef.current?.cancelSnapshot()");
    expect(panel).toContain("controller.start(source, plan, \"neural\")");
    expect(panel).toContain("livePerceptionErrorMessage(error)");
    expect(panel).not.toContain("fetch(");
  });

  it("shows negotiated limits, backpressure, control attribution, and truthful retention copy", () => {
    expect(panel).toContain("state.negotiatedCapture");
    expect(panel).toContain("sourceNativeWidth");
    expect(panel).toContain("packetsDroppedBackpressure");
    expect(panel).toContain('control.source === "brain"');
    expect(panel).toContain('(["native", "current", "custom"] as const)');
    expect(panel).toContain("Raw packets are not stored.");
    expect(panel).toContain("does not commit dataset coverage");
    expect(panel).not.toContain("720p");
  });

  it("removes collapsed perception and snapshot controls from layout", () => {
    expect(styles).toContain(
      ".live-perception-panel:not([open]) > .live-perception-panel__body"
    );
    expect(styles).toContain(
      ".live-perception-advanced:not([open]) > div"
    );
  });
});
