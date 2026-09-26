import { describe, expect, it } from "vitest";
import {
  allowTrustedAudioMedia,
  allowTrustedDisplayMedia,
  allowTrustedStudioMedia
} from "../src/main/mediaPermissionPolicy";

const valid = {
  trustedWebContentsId: 41,
  requestingWebContentsId: 41,
  permission: "media",
  isMainFrame: true,
  requestingUrl: "file:///opt/Omni/out/renderer/index.html",
  currentRendererUrl: "file:///opt/Omni/out/renderer/index.html",
  mediaTypes: ["audio"]
} as const;

describe("trusted studio audio permission", () => {
  it("allows only the top-level studio renderer's audio-only request", () => {
    expect(allowTrustedAudioMedia(valid)).toBe(true);
    expect(
      allowTrustedAudioMedia({
        ...valid,
        requestingUrl: `${valid.requestingUrl}#conversation`,
        currentRendererUrl: `${valid.currentRendererUrl}#library`
      })
    ).toBe(true);
  });

  it("allows camera-only or camera-plus-microphone only for the same trusted renderer", () => {
    expect(allowTrustedStudioMedia({ ...valid, mediaTypes: ["video"] })).toBe(true);
    expect(
      allowTrustedStudioMedia({ ...valid, mediaTypes: ["audio", "video"] })
    ).toBe(true);
    expect(
      allowTrustedAudioMedia({ ...valid, mediaTypes: ["audio", "video"] })
    ).toBe(false);
  });

  it("requires a top-level user gesture from the loaded renderer for display capture", () => {
    const display = {
      userGesture: true,
      isMainFrame: true,
      requestingUrl: valid.requestingUrl,
      currentRendererUrl: valid.currentRendererUrl
    };
    expect(allowTrustedDisplayMedia(display)).toBe(true);
    expect(allowTrustedDisplayMedia({ ...display, userGesture: false })).toBe(false);
    expect(allowTrustedDisplayMedia({ ...display, isMainFrame: false })).toBe(false);
    expect(
      allowTrustedDisplayMedia({
        ...display,
        requestingUrl: "https://attacker.example/"
      })
    ).toBe(false);
  });

  it.each([
    ["other permission", { permission: "geolocation" }],
    ["empty media types", { mediaTypes: [] }],
    ["unknown device", { mediaTypes: ["display"] }],
    ["duplicate device", { mediaTypes: ["audio", "audio"] }],
    ["subframe", { isMainFrame: false }],
    ["other web contents", { requestingWebContentsId: 99 }],
    ["navigated renderer", { requestingUrl: "https://attacker.example/" }],
    ["missing URL", { requestingUrl: undefined }]
  ])("denies %s", (_label, patch) => {
    expect(allowTrustedStudioMedia({ ...valid, ...patch })).toBe(false);
  });
});
