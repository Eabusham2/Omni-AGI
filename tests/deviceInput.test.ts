import { createHash } from "node:crypto";
import { describe, expect, it, vi } from "vitest";
import {
  DEVICE_INPUT_LIMITS,
  HostDeviceInputBackend,
  type DeviceInputAuthorization,
  type DeviceProcessInvocation,
  type DeviceProcessOutcome,
  type DeviceProcessRunner
} from "../src/main/deviceInput";

const authorization: DeviceInputAuthorization = {
  granted: true,
  policy: "ask",
  decisionId: "tool-decision:42"
};

function outcomes(
  ...values: DeviceProcessOutcome[]
): { runner: DeviceProcessRunner; calls: DeviceProcessInvocation[] } {
  const calls: DeviceProcessInvocation[] = [];
  const runner: DeviceProcessRunner = vi.fn(async (invocation) => {
    calls.push(invocation);
    const value = values.shift();
    if (!value) throw new Error("Unexpected native process invocation.");
    return value;
  });
  return { runner, calls };
}

const complete = (stdout = ""): DeviceProcessOutcome => ({
  exitCode: 0,
  stdout,
  stderr: ""
});

function decodePowerShell(invocation: DeviceProcessInvocation): string {
  const encodedAt = invocation.args.indexOf("-EncodedCommand");
  expect(encodedAt).toBeGreaterThanOrEqual(0);
  return Buffer.from(invocation.args[encodedAt + 1]!, "base64").toString("utf16le");
}

describe("HostDeviceInputBackend", () => {
  it("requires a current audited policy decision before probing or controlling the host", async () => {
    const fake = outcomes();
    const backend = new HostDeviceInputBackend({ platform: "win32", runner: fake.runner });

    const result = await backend.execute({ action: "click" });

    expect(result).toMatchObject({
      ok: false,
      state: "denied",
      errorCode: "authorization-required",
      audit: {
        action: "click",
        state: "denied",
        request: { button: "left" }
      }
    });
    expect(fake.calls).toHaveLength(0);
  });

  it("rejects malformed authorization rather than accepting an unauditable bypass", async () => {
    const fake = outcomes();
    const backend = new HostDeviceInputBackend({ platform: "linux", runner: fake.runner });

    const result = await backend.execute(
      { action: "key-press", key: "enter" },
      {
        authorization: {
          granted: true,
          policy: "full-authority",
          decisionId: "contains whitespace"
        }
      }
    );

    expect(result.errorCode).toBe("authorization-required");
    expect(fake.calls).toHaveLength(0);
  });

  it("detects an interactive Windows backend without displaying or enabling a shell", async () => {
    const fake = outcomes(complete("OMNI_INPUT_READY"));
    const backend = new HostDeviceInputBackend({ platform: "win32", runner: fake.runner });

    const capabilities = await backend.detectCapabilities();

    expect(capabilities).toMatchObject({
      platform: "win32",
      backend: "windows-user32",
      available: true
    });
    expect(Object.values(capabilities.actions).every((entry) => entry.available)).toBe(true);
    expect(fake.calls[0]).toMatchObject({
      executable: "powershell.exe",
      shell: false,
      windowsHide: true,
      timeoutMs: 2_000
    });
    expect(fake.calls[0]?.args).toEqual(expect.arrayContaining([
      "-NoProfile",
      "-NonInteractive",
      "Hidden",
      "-EncodedCommand"
    ]));
  });

  it("builds a bounded Windows User32 move and audits the native result", async () => {
    const fake = outcomes(complete("OMNI_INPUT_READY"), complete("OMNI_INPUT_OK"));
    const backend = new HostDeviceInputBackend({
      platform: "win32",
      runner: fake.runner,
      now: (() => {
        let value = Date.parse("2026-08-22T10:00:00.000Z");
        return () => value += 5;
      })(),
      createId: () => "input-audit-1"
    });

    const result = await backend.execute(
      { action: "move-pointer", x: -1200, y: 840, timeoutMs: 750 },
      { authorization }
    );

    expect(result).toMatchObject({
      ok: true,
      state: "complete",
      audit: {
        id: "input-audit-1",
        action: "move-pointer",
        backend: "windows-user32",
        durationMs: 10,
        policy: "ask",
        policyDecisionId: "tool-decision:42",
        request: { x: -1200, y: 840 },
        process: {
          executable: "powershell.exe",
          exitCode: 0,
          cancelled: false,
          timedOut: false
        }
      }
    });
    expect(fake.calls[1]).toMatchObject({ shell: false, windowsHide: true, timeoutMs: 750 });
    const script = decodePowerShell(fake.calls[1]!);
    expect(script).toContain("DllImport(\"user32.dll\"");
    expect(script).toContain("SendInput");
    expect(script).toContain("[OmniDeviceInput]::Move(-1200, 840)");
  });

  it("passes Windows Unicode text through SendInput while redacting text from the audit", async () => {
    const secret = "Hello 🧠; Remove-Item C:\\\\never";
    const fake = outcomes(complete("OMNI_INPUT_READY"), complete("OMNI_INPUT_OK"));
    const backend = new HostDeviceInputBackend({ platform: "win32", runner: fake.runner });

    const result = await backend.execute(
      { action: "text", text: secret },
      { authorization }
    );

    expect(result.state).toBe("complete");
    expect(result.audit.request).toEqual({
      utf8Bytes: Buffer.byteLength(secret, "utf8"),
      utf16Units: secret.length,
      sha256: createHash("sha256").update(secret).digest("hex")
    });
    expect(JSON.stringify(result)).not.toContain(secret);
    const script = decodePowerShell(fake.calls[1]!);
    expect(script).not.toContain(secret);
    expect(script).toContain("[OmniDeviceInput]::Text($value)");
    expect(fake.calls[1]?.stdin).toBe(secret);
    expect(fake.calls[1]?.args.join(" ").length).toBeLessThan(32_000);
  });

  it("exposes missing macOS Accessibility permission instead of pretending actions work", async () => {
    const fake = outcomes(complete("OMNI_ACCESS_REQUIRED"));
    const backend = new HostDeviceInputBackend({ platform: "darwin", runner: fake.runner });

    const capabilities = await backend.detectCapabilities();

    expect(capabilities).toMatchObject({
      backend: "macos-accessibility",
      available: false
    });
    expect(capabilities.actions.text).toMatchObject({
      supported: true,
      available: false,
      permission: "required",
      reason: expect.stringMatching(/Accessibility/)
    });
  });

  it("uses macOS System Events for keys and never invokes a shell", async () => {
    const fake = outcomes(complete("OMNI_ACCESS_GRANTED"), complete("true"), complete());
    const backend = new HostDeviceInputBackend({ platform: "darwin", runner: fake.runner });

    const result = await backend.execute(
      { action: "key-press", key: "enter", modifiers: ["meta", "shift"] },
      { authorization }
    );

    expect(result.state).toBe("complete");
    expect(fake.calls[2]).toMatchObject({
      executable: "/usr/bin/osascript",
      shell: false,
      windowsHide: true
    });
    expect(fake.calls[2]?.args).toEqual([
      "-e",
      "tell application \"System Events\" to key code 36 using {command down, shift down}"
    ]);
  });

  it("reports System Events separately while retaining permitted pointer actions", async () => {
    const fake = outcomes(
      complete("OMNI_ACCESS_GRANTED"),
      { exitCode: 1, stdout: "", stderr: "Not authorized to send Apple events (-1743)" }
    );
    const backend = new HostDeviceInputBackend({ platform: "darwin", runner: fake.runner });

    const capabilities = await backend.detectCapabilities();

    expect(capabilities.available).toBe(true);
    expect(capabilities.actions.click).toMatchObject({ available: true, permission: "granted" });
    expect(capabilities.actions.text).toMatchObject({
      available: false,
      permission: "required",
      reason: expect.stringMatching(/System Events/)
    });
  });

  it.each([
    [
      { action: "move-pointer" as const, x: 100, y: 200 },
      "CGWarpMouseCursorPosition"
    ],
    [
      { action: "click" as const, button: "right" as const },
      "CGEventCreateMouseEvent"
    ],
    [
      { action: "scroll" as const, deltaX: 120, deltaY: -240 },
      "CGEventCreateScrollWheelEvent"
    ]
  ])("uses permission-checked macOS Core Graphics for %s", async (command, marker) => {
    const fake = outcomes(complete("OMNI_ACCESS_GRANTED"), complete("true"), complete());
    const backend = new HostDeviceInputBackend({ platform: "darwin", runner: fake.runner });

    const result = await backend.execute(command, { authorization });

    expect(result.state).toBe("complete");
    expect(fake.calls[2]?.args.slice(0, 3)).toEqual(["-l", "JavaScript", "-e"]);
    expect(fake.calls[2]?.args[3]).toContain("AXIsProcessTrusted");
    expect(fake.calls[2]?.args[3]).toContain(marker);
  });

  it("turns a macOS System Events denial into an explicit permission result", async () => {
    const fake = outcomes(
      complete("OMNI_ACCESS_GRANTED"),
      complete("true"),
      { exitCode: 1, stdout: "", stderr: "System Events got an error: Not authorized (-1743)" }
    );
    const backend = new HostDeviceInputBackend({ platform: "darwin", runner: fake.runner });

    const result = await backend.execute(
      { action: "text", text: "hello" },
      { authorization }
    );

    expect(result).toMatchObject({
      state: "denied",
      errorCode: "permission-required",
      error: expect.stringMatching(/macOS denied/)
    });
  });

  it("does not claim Linux host control without an X11 display", async () => {
    const fake = outcomes();
    const backend = new HostDeviceInputBackend({
      platform: "linux",
      runner: fake.runner,
      environment: { XDG_SESSION_TYPE: "wayland", DISPLAY: ":0" }
    });

    const capabilities = await backend.detectCapabilities();
    const result = await backend.execute({ action: "click" }, { authorization });

    expect(capabilities.available).toBe(false);
    expect(capabilities.actions.click.reason).toMatch(/X11-only|Wayland/);
    expect(result).toMatchObject({ state: "unavailable", errorCode: "capability-unavailable" });
    expect(fake.calls).toHaveLength(0);
  });

  it("reports a missing xdotool installation honestly", async () => {
    const fake = outcomes({
      exitCode: null,
      stdout: "",
      stderr: "",
      errorCode: "ENOENT",
      errorMessage: "spawn xdotool ENOENT"
    });
    const backend = new HostDeviceInputBackend({
      platform: "linux",
      runner: fake.runner,
      environment: { XDG_SESSION_TYPE: "x11", DISPLAY: ":0" }
    });

    const capabilities = await backend.detectCapabilities();

    expect(capabilities).toMatchObject({ backend: "unavailable", available: false });
    expect(capabilities.actions.text.reason).toBe("xdotool is not installed.");
  });

  it.each([
    [
      { action: "move-pointer" as const, x: 12, y: 34 },
      ["mousemove", "--sync", "12", "34"]
    ],
    [
      { action: "click" as const, button: "middle" as const },
      ["click", "--repeat", "1", "2"]
    ],
    [
      { action: "scroll" as const, deltaX: -240, deltaY: 120 },
      ["click", "--repeat", "1", "4", "click", "--repeat", "2", "6"]
    ],
    [
      { action: "key-press" as const, key: "f2", modifiers: ["ctrl", "meta"] as const },
      ["key", "--clearmodifiers", "ctrl+super+F2"]
    ],
    [
      { action: "text" as const, text: "hello --window 0" },
      ["type", "--clearmodifiers", "--delay", "0", "--", "hello --window 0"]
    ]
  ])("uses a separate, bounded xdotool argument vector for %s", async (command, expected) => {
    const fake = outcomes(complete("X=1\nY=2"), complete());
    const backend = new HostDeviceInputBackend({
      platform: "linux",
      runner: fake.runner,
      environment: { XDG_SESSION_TYPE: "x11", DISPLAY: ":0" }
    });

    const result = await backend.execute(command, { authorization });

    expect(result.state).toBe("complete");
    expect(fake.calls[1]).toMatchObject({
      executable: "xdotool",
      args: expected,
      shell: false,
      windowsHide: true
    });
  });

  it.each([
    [{ action: "move-pointer", x: 100_001, y: 0 }, /between/],
    [{ action: "move-pointer", x: 1.5, y: 0 }, /finite integer/],
    [{ action: "scroll", deltaX: 0, deltaY: 0 }, /non-zero/],
    [{ action: "scroll", deltaY: 10_001 }, /between/],
    [{ action: "key-press", key: "launch-nukes" }, /portable key set/],
    [{ action: "key-press", key: "a", modifiers: ["ctrl", "ctrl"] }, /duplicates/],
    [{ action: "text", text: "\0" }, /without NUL/],
    [{ action: "click", timeoutMs: 11_000 }, /between/]
  ])("rejects an out-of-contract command before capability detection: %o", async (command, error) => {
    const fake = outcomes();
    const backend = new HostDeviceInputBackend({ platform: "win32", runner: fake.runner });

    const result = await backend.execute(command as never, { authorization });

    expect(result).toMatchObject({ state: "failed", errorCode: "invalid-command" });
    expect(result.error).toMatch(error);
    expect(fake.calls).toHaveLength(0);
  });

  it("bounds text by UTF-8 bytes rather than only JavaScript character count", async () => {
    const fake = outcomes();
    const backend = new HostDeviceInputBackend({ platform: "win32", runner: fake.runner });
    const oversized = "🧠".repeat((DEVICE_INPUT_LIMITS.textBytes / 4) + 1);

    const result = await backend.execute(
      { action: "text", text: oversized },
      { authorization }
    );

    expect(result.errorCode).toBe("invalid-command");
    expect(result.error).toMatch(/UTF-8 bytes/);
    expect(fake.calls).toHaveLength(0);
  });

  it("cancels before probing and returns a complete audit record", async () => {
    const fake = outcomes();
    const backend = new HostDeviceInputBackend({ platform: "win32", runner: fake.runner });
    const controller = new AbortController();
    controller.abort();

    const result = await backend.execute(
      { action: "click" },
      { authorization, signal: controller.signal }
    );

    expect(result).toMatchObject({
      state: "cancelled",
      errorCode: "cancelled",
      audit: { action: "click", state: "cancelled" }
    });
    expect(fake.calls).toHaveLength(0);
  });

  it("reports cancellation and timeout from an in-flight native action", async () => {
    const cancelledFake = outcomes(
      complete("OMNI_INPUT_READY"),
      { exitCode: null, stdout: "", stderr: "", cancelled: true }
    );
    const timedOutFake = outcomes(
      complete("OMNI_INPUT_READY"),
      { exitCode: null, stdout: "", stderr: "", timedOut: true }
    );
    const cancelledBackend = new HostDeviceInputBackend({
      platform: "win32",
      runner: cancelledFake.runner
    });
    const timedOutBackend = new HostDeviceInputBackend({
      platform: "win32",
      runner: timedOutFake.runner
    });

    const cancelled = await cancelledBackend.execute({ action: "click" }, { authorization });
    const timedOut = await timedOutBackend.execute(
      { action: "click", timeoutMs: 500 },
      { authorization }
    );

    expect(cancelled).toMatchObject({ state: "cancelled", errorCode: "cancelled" });
    expect(timedOut).toMatchObject({ state: "failed", errorCode: "timed-out" });
    expect(timedOut.audit.process).toMatchObject({ timedOut: true });
  });

  it("does not run a process or advertise actions on unsupported hosts", async () => {
    const fake = outcomes();
    const backend = new HostDeviceInputBackend({ platform: "freebsd", runner: fake.runner });

    const capabilities = await backend.detectCapabilities();
    const result = await backend.execute({ action: "click" }, { authorization });

    expect(capabilities.backend).toBe("unavailable");
    expect(capabilities.actions.click.supported).toBe(false);
    expect(result.state).toBe("unavailable");
    expect(fake.calls).toHaveLength(0);
  });
});
