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
).replace(/\r\n?/g, "\n");
const styles = readFileSync(
  resolve(process.cwd(), "src/renderer/src/styles.css"),
  "utf8"
);
const callLayout = readFileSync(resolve(process.cwd(), "src/renderer/src/CallLayout.tsx"), "utf8");
const preview = readFileSync(resolve(process.cwd(), "src/renderer/src/LiveCapturePreview.tsx"), "utf8");
const captureStyles = readFileSync(resolve(process.cwd(), "src/renderer/src/cameraScreenPanel.css"), "utf8");

describe("Live Perception renderer surface", () => {
  it("stays reachable without live voice and binds only to typed observation controls", () => {
    expect(app).toContain('<LivePerceptionPanel brain={brain} runtimeLoaded={runtimeLoaded} />');
    expect(app).toContain('hidden={!callOpen}');
    expect(callLayout).toContain('{controls}<div');
    expect(callLayout).not.toContain('open ? controls');
    expect(panel).toContain("controller.requestConfigure");
    expect(panel).toContain("controller.requestSnapshot");
    expect(panel).toContain("controllerRef.current?.cancelSnapshot()");
    expect(panel).toContain("controller.start(source, plan, \"neural\", delivery)");
    expect(panel).toContain("livePerceptionErrorMessage(error)");
    expect(panel).toContain('key={brain.id}');
    expect(panel).toContain('controller.subscribePreview(setPreview)');
    expect(panel).toContain('if (runtimeLoaded) return;');
    expect(panel).toContain('void controller?.cancel()');
    expect(panel).not.toContain('<strong>Live Perception</strong>');
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

  it("keeps real previews accessible inside Call and collapses advanced/snapshot layout only", () => {
    expect(preview).toContain('attachOwnedCapturePreview(video.current, stream, setPlayError)');
    expect(preview).toContain('return owned.release');
    expect(preview).toContain('Latest accepted snapshot');
    expect(preview).toContain('{snapshot ? <details'); // Keep the disclosure mounted while each new frame URL replaces the last.
    expect(preview).not.toContain('{still ? <details');
    expect(styles).toContain(
      ".live-perception-advanced:not([open]) > div"
    );
    expect(captureStyles).toContain('.camera-screen-preview__snapshot:not([open]) > img');
    expect(captureStyles).toContain(':focus-visible');
  });
});
