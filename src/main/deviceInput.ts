import { createHash, randomUUID } from "node:crypto";
import { spawn, type ChildProcess } from "node:child_process";

export const DEVICE_INPUT_LIMITS = Object.freeze({
  coordinate: 100_000,
  scrollDelta: 10_000,
  textBytes: 16 * 1024,
  timeoutMinimumMs: 250,
  timeoutMaximumMs: 10_000,
  timeoutDefaultMs: 4_000,
  processOutputBytes: 32 * 1024
});

export type DeviceInputAction =
  | "move-pointer"
  | "click"
  | "scroll"
  | "key-press"
  | "text";

export type DeviceInputBackend =
  | "windows-user32"
  | "macos-accessibility"
  | "linux-xdotool"
  | "unavailable";

export type DeviceInputPermission =
  | "granted"
  | "required"
  | "not-required"
  | "unknown";

export type DeviceInputModifier = "ctrl" | "alt" | "shift" | "meta";
export type DeviceInputButton = "left" | "middle" | "right";

export interface MovePointerCommand {
  action: "move-pointer";
  x: number;
  y: number;
  timeoutMs?: number;
}

export interface ClickCommand {
  action: "click";
  button?: DeviceInputButton;
  timeoutMs?: number;
}

export interface ScrollCommand {
  action: "scroll";
  deltaX?: number;
  deltaY?: number;
  timeoutMs?: number;
}

export interface KeyPressCommand {
  action: "key-press";
  key: string;
  modifiers?: readonly DeviceInputModifier[];
  timeoutMs?: number;
}

export interface TypeTextCommand {
  action: "text";
  text: string;
  timeoutMs?: number;
}

export type DeviceInputCommand =
  | MovePointerCommand
  | ClickCommand
  | ScrollCommand
  | KeyPressCommand
  | TypeTextCommand;

export interface DeviceInputAuthorization {
  /** A positive decision made by ToolExecutor or an equally trusted policy owner. */
  granted: true;
  policy: "ask" | "auto" | "full-authority";
  decisionId: string;
}

export interface DeviceInputExecutionContext {
  authorization?: DeviceInputAuthorization;
  signal?: AbortSignal;
}

export interface DeviceInputActionCapability {
  supported: boolean;
  available: boolean;
  permission: DeviceInputPermission;
  reason?: string;
}

export interface DeviceInputCapabilities {
  platform: NodeJS.Platform;
  backend: DeviceInputBackend;
  available: boolean;
  actions: Record<DeviceInputAction, DeviceInputActionCapability>;
  detectedAt: string;
  notes: string[];
}

export interface DeviceProcessInvocation {
  executable: string;
  args: readonly string[];
  /** Bounded process input. This is never copied into the audit record. */
  stdin?: string;
  timeoutMs: number;
  signal?: AbortSignal;
  environment?: Readonly<Record<string, string | undefined>>;
  shell: false;
  windowsHide: true;
}

export interface DeviceProcessOutcome {
  exitCode: number | null;
  stdout: string;
  stderr: string;
  cancelled?: boolean;
  timedOut?: boolean;
  errorCode?: string;
  errorMessage?: string;
  outputTruncated?: boolean;
}

export type DeviceProcessRunner = (
  invocation: DeviceProcessInvocation
) => Promise<DeviceProcessOutcome>;

export type DeviceInputResultState =
  | "complete"
  | "denied"
  | "unavailable"
  | "cancelled"
  | "failed";

export interface DeviceInputAuditRecord {
  id: string;
  action: DeviceInputAction | "invalid";
  state: DeviceInputResultState;
  platform: NodeJS.Platform;
  backend: DeviceInputBackend;
  startedAt: string;
  completedAt: string;
  durationMs: number;
  policyDecisionId?: string;
  policy?: DeviceInputAuthorization["policy"];
  request: Record<string, string | number | string[]>;
  process?: {
    executable: string;
    exitCode: number | null;
    cancelled: boolean;
    timedOut: boolean;
    outputTruncated: boolean;
  };
}

export interface DeviceInputResult {
  ok: boolean;
  state: DeviceInputResultState;
  action: DeviceInputAction | "invalid";
  errorCode?:
    | "authorization-required"
    | "invalid-command"
    | "capability-unavailable"
    | "permission-required"
    | "cancelled"
    | "timed-out"
    | "backend-failed";
  error?: string;
  audit: DeviceInputAuditRecord;
}

export interface DeviceInputBackendOptions {
  platform?: NodeJS.Platform;
  runner?: DeviceProcessRunner;
  environment?: Readonly<Record<string, string | undefined>>;
  now?: () => number;
  createId?: () => string;
}

interface NormalizedCommand {
  action: DeviceInputAction;
  timeoutMs: number;
  auditRequest: Record<string, string | number | string[]>;
  x?: number;
  y?: number;
  deltaX?: number;
  deltaY?: number;
  button?: DeviceInputButton;
  key?: string;
  modifiers?: DeviceInputModifier[];
  text?: string;
}

const ACTIONS: readonly DeviceInputAction[] = [
  "move-pointer",
  "click",
  "scroll",
  "key-press",
  "text"
];

const PRESSABLE_NAMED_KEYS = new Set([
  "backspace",
  "tab",
  "enter",
  "escape",
  "space",
  "pageup",
  "pagedown",
  "end",
  "home",
  "left",
  "up",
  "right",
  "down",
  "insert",
  "delete"
]);

const WINDOWS_VIRTUAL_KEYS: Readonly<Record<string, number>> = Object.freeze({
  backspace: 0x08,
  tab: 0x09,
  enter: 0x0d,
  shift: 0x10,
  ctrl: 0x11,
  alt: 0x12,
  escape: 0x1b,
  space: 0x20,
  pageup: 0x21,
  pagedown: 0x22,
  end: 0x23,
  home: 0x24,
  left: 0x25,
  up: 0x26,
  right: 0x27,
  down: 0x28,
  insert: 0x2d,
  delete: 0x2e,
  meta: 0x5b
});

const MAC_KEY_CODES: Readonly<Record<string, number>> = Object.freeze({
  a: 0,
  s: 1,
  d: 2,
  f: 3,
  h: 4,
  g: 5,
  z: 6,
  x: 7,
  c: 8,
  v: 9,
  b: 11,
  q: 12,
  w: 13,
  e: 14,
  r: 15,
  y: 16,
  t: 17,
  "1": 18,
  "2": 19,
  "3": 20,
  "4": 21,
  "6": 22,
  "5": 23,
  "9": 25,
  "7": 26,
  "8": 28,
  "0": 29,
  o: 31,
  u: 32,
  i: 34,
  p: 35,
  enter: 36,
  l: 37,
  j: 38,
  k: 40,
  n: 45,
  m: 46,
  tab: 48,
  space: 49,
  backspace: 51,
  escape: 53,
  f5: 96,
  f6: 97,
  f7: 98,
  f3: 99,
  f8: 100,
  f9: 101,
  f11: 103,
  f10: 109,
  f12: 111,
  home: 115,
  pageup: 116,
  delete: 117,
  f4: 118,
  end: 119,
  f2: 120,
  pagedown: 121,
  f1: 122,
  left: 123,
  right: 124,
  down: 125,
  up: 126
});

const LINUX_KEYS: Readonly<Record<string, string>> = Object.freeze({
  backspace: "BackSpace",
  tab: "Tab",
  enter: "Return",
  escape: "Escape",
  space: "space",
  pageup: "Page_Up",
  pagedown: "Page_Down",
  end: "End",
  home: "Home",
  left: "Left",
  up: "Up",
  right: "Right",
  down: "Down",
  insert: "Insert",
  delete: "Delete"
});

const WINDOWS_INPUT_TYPE = String.raw`
using System;
using System.ComponentModel;
using System.Runtime.InteropServices;

public static class OmniDeviceInput {
  private const uint INPUT_MOUSE = 0;
  private const uint INPUT_KEYBOARD = 1;
  private const uint KEYEVENTF_KEYUP = 0x0002;
  private const uint KEYEVENTF_UNICODE = 0x0004;
  private const uint MOUSEEVENTF_LEFTDOWN = 0x0002;
  private const uint MOUSEEVENTF_LEFTUP = 0x0004;
  private const uint MOUSEEVENTF_RIGHTDOWN = 0x0008;
  private const uint MOUSEEVENTF_RIGHTUP = 0x0010;
  private const uint MOUSEEVENTF_MIDDLEDOWN = 0x0020;
  private const uint MOUSEEVENTF_MIDDLEUP = 0x0040;
  private const uint MOUSEEVENTF_WHEEL = 0x0800;
  private const uint MOUSEEVENTF_HWHEEL = 0x1000;

  [StructLayout(LayoutKind.Sequential)]
  private struct MOUSEINPUT {
    public int dx;
    public int dy;
    public uint mouseData;
    public uint dwFlags;
    public uint time;
    public UIntPtr dwExtraInfo;
  }

  [StructLayout(LayoutKind.Sequential)]
  private struct KEYBDINPUT {
    public ushort wVk;
    public ushort wScan;
    public uint dwFlags;
    public uint time;
    public UIntPtr dwExtraInfo;
  }

  [StructLayout(LayoutKind.Sequential)]
  private struct HARDWAREINPUT {
    public uint uMsg;
    public ushort wParamL;
    public ushort wParamH;
  }

  [StructLayout(LayoutKind.Explicit)]
  private struct INPUTUNION {
    [FieldOffset(0)] public MOUSEINPUT mouse;
    [FieldOffset(0)] public KEYBDINPUT keyboard;
    [FieldOffset(0)] public HARDWAREINPUT hardware;
  }

  [StructLayout(LayoutKind.Sequential)]
  private struct INPUT {
    public uint type;
    public INPUTUNION value;
  }

  [DllImport("user32.dll", SetLastError = true)]
  private static extern bool SetCursorPos(int x, int y);

  [DllImport("user32.dll", SetLastError = true)]
  private static extern uint SendInput(uint count, INPUT[] values, int size);

  private static void Send(INPUT[] values) {
    uint sent = SendInput((uint)values.Length, values, Marshal.SizeOf(typeof(INPUT)));
    if (sent != values.Length) {
      throw new Win32Exception(Marshal.GetLastWin32Error(),
        "OMNI_INPUT_DENIED: Windows blocked synthetic input. A secure or higher-integrity desktop may be active.");
    }
  }

  private static INPUT Mouse(uint flags, int data) {
    return new INPUT {
      type = INPUT_MOUSE,
      value = new INPUTUNION {
        mouse = new MOUSEINPUT { mouseData = unchecked((uint)data), dwFlags = flags }
      }
    };
  }

  private static INPUT Keyboard(ushort key, ushort scan, uint flags) {
    return new INPUT {
      type = INPUT_KEYBOARD,
      value = new INPUTUNION {
        keyboard = new KEYBDINPUT { wVk = key, wScan = scan, dwFlags = flags }
      }
    };
  }

  public static void Move(int x, int y) {
    if (!SetCursorPos(x, y)) throw new Win32Exception(Marshal.GetLastWin32Error());
  }

  public static void Click(string button) {
    uint down;
    uint up;
    switch (button) {
      case "left": down = MOUSEEVENTF_LEFTDOWN; up = MOUSEEVENTF_LEFTUP; break;
      case "middle": down = MOUSEEVENTF_MIDDLEDOWN; up = MOUSEEVENTF_MIDDLEUP; break;
      case "right": down = MOUSEEVENTF_RIGHTDOWN; up = MOUSEEVENTF_RIGHTUP; break;
      default: throw new ArgumentException("Unsupported button.");
    }
    Send(new [] { Mouse(down, 0), Mouse(up, 0) });
  }

  public static void Scroll(int deltaX, int deltaY) {
    if (deltaY != 0 && deltaX != 0) {
      Send(new [] { Mouse(MOUSEEVENTF_WHEEL, deltaY), Mouse(MOUSEEVENTF_HWHEEL, deltaX) });
    } else if (deltaY != 0) {
      Send(new [] { Mouse(MOUSEEVENTF_WHEEL, deltaY) });
    } else {
      Send(new [] { Mouse(MOUSEEVENTF_HWHEEL, deltaX) });
    }
  }

  public static void Key(ushort key, ushort[] modifiers) {
    INPUT[] values = new INPUT[(modifiers.Length * 2) + 2];
    int index = 0;
    foreach (ushort modifier in modifiers) values[index++] = Keyboard(modifier, 0, 0);
    values[index++] = Keyboard(key, 0, 0);
    values[index++] = Keyboard(key, 0, KEYEVENTF_KEYUP);
    for (int i = modifiers.Length - 1; i >= 0; i--) {
      values[index++] = Keyboard(modifiers[i], 0, KEYEVENTF_KEYUP);
    }
    Send(values);
  }

  public static void Text(string text) {
    INPUT[] values = new INPUT[text.Length * 2];
    int index = 0;
    foreach (char value in text) {
      values[index++] = Keyboard(0, value, KEYEVENTF_UNICODE);
      values[index++] = Keyboard(0, value, KEYEVENTF_UNICODE | KEYEVENTF_KEYUP);
    }
    Send(values);
  }
}
`;

function boundedInteger(
  value: unknown,
  label: string,
  minimum: number,
  maximum: number
): number {
  if (typeof value !== "number" || !Number.isFinite(value) || !Number.isInteger(value)) {
    throw new Error(`${label} must be a finite integer.`);
  }
  if (value < minimum || value > maximum) {
    throw new Error(`${label} must be between ${minimum} and ${maximum}.`);
  }
  return value;
}

function boundedTimeout(value: unknown): number {
  if (value === undefined) return DEVICE_INPUT_LIMITS.timeoutDefaultMs;
  return boundedInteger(
    value,
    "timeoutMs",
    DEVICE_INPUT_LIMITS.timeoutMinimumMs,
    DEVICE_INPUT_LIMITS.timeoutMaximumMs
  );
}

function normalizeKey(value: unknown): string {
  if (typeof value !== "string") throw new Error("key must be a string.");
  const key = value.trim().toLocaleLowerCase();
  if (/^[a-z0-9]$/.test(key)) return key;
  if (/^f(?:[1-9]|1[0-2])$/.test(key)) return key;
  if (PRESSABLE_NAMED_KEYS.has(key)) return key;
  throw new Error("key is not in the bounded portable key set.");
}

function normalizeModifiers(value: unknown): DeviceInputModifier[] {
  if (value === undefined) return [];
  if (!Array.isArray(value) || value.length > 4) {
    throw new Error("modifiers must be an array with at most four entries.");
  }
  const result: DeviceInputModifier[] = [];
  for (const candidate of value) {
    if (!["ctrl", "alt", "shift", "meta"].includes(String(candidate))) {
      throw new Error("modifiers may contain only ctrl, alt, shift, or meta.");
    }
    const modifier = candidate as DeviceInputModifier;
    if (result.includes(modifier)) throw new Error("modifiers may not contain duplicates.");
    result.push(modifier);
  }
  return result;
}

function normalizeCommand(command: unknown): NormalizedCommand {
  if (!command || typeof command !== "object" || Array.isArray(command)) {
    throw new Error("Device input command must be an object.");
  }
  const candidate = command as Record<string, unknown>;
  const actionValue = candidate.action;
  if (!ACTIONS.includes(actionValue as DeviceInputAction)) {
    throw new Error("Device input action is unsupported.");
  }
  const action = actionValue as DeviceInputAction;
  const timeoutMs = boundedTimeout(candidate.timeoutMs);
  if (action === "move-pointer") {
    const x = boundedInteger(
      candidate.x,
      "x",
      -DEVICE_INPUT_LIMITS.coordinate,
      DEVICE_INPUT_LIMITS.coordinate
    );
    const y = boundedInteger(
      candidate.y,
      "y",
      -DEVICE_INPUT_LIMITS.coordinate,
      DEVICE_INPUT_LIMITS.coordinate
    );
    return { action, x, y, timeoutMs, auditRequest: { x, y } };
  }
  if (action === "click") {
    const button = candidate.button ?? "left";
    if (!["left", "middle", "right"].includes(String(button))) {
      throw new Error("button must be left, middle, or right.");
    }
    return {
      action,
      button: button as DeviceInputButton,
      timeoutMs,
      auditRequest: { button: String(button) }
    };
  }
  if (action === "scroll") {
    const deltaX = boundedInteger(
      candidate.deltaX ?? 0,
      "deltaX",
      -DEVICE_INPUT_LIMITS.scrollDelta,
      DEVICE_INPUT_LIMITS.scrollDelta
    );
    const deltaY = boundedInteger(
      candidate.deltaY ?? 0,
      "deltaY",
      -DEVICE_INPUT_LIMITS.scrollDelta,
      DEVICE_INPUT_LIMITS.scrollDelta
    );
    if (deltaX === 0 && deltaY === 0) {
      throw new Error("At least one scroll delta must be non-zero.");
    }
    return { action, deltaX, deltaY, timeoutMs, auditRequest: { deltaX, deltaY } };
  }
  if (action === "key-press") {
    const key = normalizeKey(candidate.key);
    const modifiers = normalizeModifiers(candidate.modifiers);
    return { action, key, modifiers, timeoutMs, auditRequest: { key, modifiers } };
  }
  const text = candidate.text;
  if (typeof text !== "string" || text.length === 0 || text.includes("\0")) {
    throw new Error("text must be a non-empty string without NUL bytes.");
  }
  const bytes = Buffer.byteLength(text, "utf8");
  if (bytes > DEVICE_INPUT_LIMITS.textBytes) {
    throw new Error(`text must not exceed ${DEVICE_INPUT_LIMITS.textBytes} UTF-8 bytes.`);
  }
  return {
    action,
    text,
    timeoutMs,
    auditRequest: {
      utf8Bytes: bytes,
      utf16Units: text.length,
      sha256: createHash("sha256").update(text).digest("hex")
    }
  };
}

function allCapabilities(
  supported: boolean,
  available: boolean,
  permission: DeviceInputPermission,
  reason?: string
): Record<DeviceInputAction, DeviceInputActionCapability> {
  return Object.fromEntries(
    ACTIONS.map((action) => [action, { supported, available, permission, reason }])
  ) as Record<DeviceInputAction, DeviceInputActionCapability>;
}

function boundedOutput(
  current: Buffer<ArrayBufferLike>,
  chunk: Buffer<ArrayBufferLike>
): { value: Buffer<ArrayBufferLike>; truncated: boolean } {
  const remaining = DEVICE_INPUT_LIMITS.processOutputBytes - current.byteLength;
  if (remaining <= 0) return { value: current, truncated: true };
  return {
    value: Buffer.concat([current, chunk.subarray(0, remaining)]),
    truncated: chunk.byteLength > remaining
  };
}

function stopChild(child: ChildProcess): void {
  if (child.exitCode !== null) return;
  child.kill("SIGTERM");
}

/** Default runner. Tests inject a fake runner and never reach this process boundary. */
export const runDeviceInputProcess: DeviceProcessRunner = async (
  invocation
): Promise<DeviceProcessOutcome> =>
  new Promise((resolve) => {
    if (invocation.signal?.aborted) {
      resolve({ exitCode: null, stdout: "", stderr: "", cancelled: true });
      return;
    }
    const child = spawn(invocation.executable, [...invocation.args], {
      env: { ...process.env, ...invocation.environment },
      shell: false,
      windowsHide: true,
      stdio: [invocation.stdin === undefined ? "ignore" : "pipe", "pipe", "pipe"]
    });
    let stdout: Buffer<ArrayBufferLike> = Buffer.alloc(0);
    let stderr: Buffer<ArrayBufferLike> = Buffer.alloc(0);
    let outputTruncated = false;
    let settled = false;
    let termination: "cancelled" | "timed-out" | undefined;
    const finish = (outcome: DeviceProcessOutcome): void => {
      if (settled) return;
      settled = true;
      clearTimeout(timer);
      invocation.signal?.removeEventListener("abort", abort);
      resolve({ ...outcome, outputTruncated });
    };
    child.stdout?.on("data", (chunk: Buffer<ArrayBufferLike>) => {
      const next = boundedOutput(stdout, chunk);
      stdout = next.value;
      outputTruncated ||= next.truncated;
    });
    child.stderr?.on("data", (chunk: Buffer<ArrayBufferLike>) => {
      const next = boundedOutput(stderr, chunk);
      stderr = next.value;
      outputTruncated ||= next.truncated;
    });
    if (invocation.stdin !== undefined) {
      // The command has already validated the byte bound. Ignore a benign
      // EPIPE when a cancelled or failed native process closes first.
      child.stdin?.on("error", () => undefined);
      child.stdin?.end(invocation.stdin, "utf8");
    }
    const terminate = (reason: "cancelled" | "timed-out"): void => {
      if (termination) return;
      termination = reason;
      stopChild(child);
    };
    const timer = setTimeout(() => terminate("timed-out"), invocation.timeoutMs);
    const abort = (): void => terminate("cancelled");
    invocation.signal?.addEventListener("abort", abort, { once: true });
    if (invocation.signal?.aborted) abort();
    child.once("error", (error: NodeJS.ErrnoException) => {
      finish({
        exitCode: null,
        stdout: stdout.toString("utf8"),
        stderr: stderr.toString("utf8"),
        cancelled: termination === "cancelled",
        timedOut: termination === "timed-out",
        errorCode: error.code,
        errorMessage: error.message
      });
    });
    child.once("close", (exitCode) => {
      finish({
        exitCode,
        stdout: stdout.toString("utf8"),
        stderr: stderr.toString("utf8"),
        cancelled: termination === "cancelled",
        timedOut: termination === "timed-out"
      });
    });
  });

function powershellArgs(script: string): string[] {
  return [
    "-NoLogo",
    "-NoProfile",
    "-NonInteractive",
    "-WindowStyle",
    "Hidden",
    "-EncodedCommand",
    Buffer.from(script, "utf16le").toString("base64")
  ];
}

function processInvocation(
  executable: string,
  args: readonly string[],
  timeoutMs: number,
  signal: AbortSignal | undefined,
  environment: Readonly<Record<string, string | undefined>>,
  stdin?: string
): DeviceProcessInvocation {
  return {
    executable,
    args,
    timeoutMs,
    signal,
    environment,
    stdin,
    shell: false,
    windowsHide: true
  };
}

function windowsKeyCode(key: string): number {
  if (/^[a-z]$/.test(key)) return key.toUpperCase().charCodeAt(0);
  if (/^[0-9]$/.test(key)) return key.charCodeAt(0);
  if (/^f(?:[1-9]|1[0-2])$/.test(key)) return 0x6f + Number(key.slice(1));
  return WINDOWS_VIRTUAL_KEYS[key]!;
}

function windowsScript(command: NormalizedCommand): string {
  let operation: string;
  if (command.action === "move-pointer") {
    operation = `[OmniDeviceInput]::Move(${command.x}, ${command.y})`;
  } else if (command.action === "click") {
    operation = `[OmniDeviceInput]::Click('${command.button}')`;
  } else if (command.action === "scroll") {
    operation = `[OmniDeviceInput]::Scroll(${command.deltaX}, ${command.deltaY})`;
  } else if (command.action === "key-press") {
    const modifierKeys = command.modifiers!.map((modifier) => windowsKeyCode(modifier));
    operation = `[OmniDeviceInput]::Key(${windowsKeyCode(command.key!)}, [ushort[]]@(${modifierKeys.join(",")}))`;
  } else {
    operation = "$value = [Console]::In.ReadToEnd()\n[OmniDeviceInput]::Text($value)";
  }
  return `$ErrorActionPreference = 'Stop'\nAdd-Type -TypeDefinition @'${WINDOWS_INPUT_TYPE}'@\n${operation}\n[Console]::Out.Write('OMNI_INPUT_OK')`;
}

function macModifierTerms(modifiers: readonly DeviceInputModifier[]): string {
  const terms: Record<DeviceInputModifier, string> = {
    ctrl: "control down",
    alt: "option down",
    shift: "shift down",
    meta: "command down"
  };
  return modifiers.length > 0
    ? ` using {${modifiers.map((modifier) => terms[modifier]).join(", ")}}`
    : "";
}

function macCoreGraphicsScript(command: NormalizedCommand): string {
  const prelude = `ObjC.import('ApplicationServices');\nif (!$.AXIsProcessTrusted()) { throw new Error('OMNI_ACCESSIBILITY_REQUIRED'); }`;
  if (command.action === "move-pointer") {
    return `${prelude}\n$.CGWarpMouseCursorPosition($.CGPointMake(${command.x}, ${command.y}));\n'OMNI_INPUT_OK';`;
  }
  if (command.action === "click") {
    const button = { left: 0, right: 1, middle: 2 }[command.button!];
    const eventTypes = {
      left: [1, 2],
      right: [3, 4],
      middle: [25, 26]
    } as const;
    const [down, up] = eventTypes[command.button!];
    return `${prelude}
const current = $.CGEventCreate(null);
const point = $.CGEventGetLocation(current);
$.CFRelease(current);
const down = $.CGEventCreateMouseEvent(null, ${down}, point, ${button});
const up = $.CGEventCreateMouseEvent(null, ${up}, point, ${button});
$.CGEventPost(0, down);
$.CGEventPost(0, up);
$.CFRelease(down);
$.CFRelease(up);
'OMNI_INPUT_OK';`;
  }
  const horizontal = Math.max(-100, Math.min(100, Math.round(command.deltaX! / 120)));
  const vertical = Math.max(-100, Math.min(100, Math.round(command.deltaY! / 120)));
  const adjustedHorizontal = horizontal === 0 && command.deltaX !== 0 ? Math.sign(command.deltaX!) : horizontal;
  const adjustedVertical = vertical === 0 && command.deltaY !== 0 ? Math.sign(command.deltaY!) : vertical;
  return `${prelude}
const event = $.CGEventCreateScrollWheelEvent2(null, 1, 2, ${adjustedVertical}, ${adjustedHorizontal}, 0);
$.CGEventPost(0, event);
$.CFRelease(event);
'OMNI_INPUT_OK';`;
}

function macSystemEventsArgs(command: NormalizedCommand): string[] {
  if (command.action === "key-press") {
    const code = MAC_KEY_CODES[command.key!];
    if (code === undefined) throw new Error("key is unavailable through macOS System Events.");
    const script = `tell application "System Events" to key code ${code}${macModifierTerms(command.modifiers!)}`;
    return ["-e", script];
  }
  const script = `on run argv
  if (count of argv) is not 1 then error "OMNI_INVALID_TEXT"
  tell application "System Events" to keystroke (item 1 of argv)
end run`;
  // `--` keeps text beginning with a dash out of osascript's option parser.
  return ["-e", script, "--", command.text!];
}

function linuxKeyName(key: string): string {
  if (/^[a-z0-9]$/.test(key)) return key;
  if (/^f(?:[1-9]|1[0-2])$/.test(key)) return key.toUpperCase();
  return LINUX_KEYS[key]!;
}

function linuxArgs(command: NormalizedCommand): string[] {
  if (command.action === "move-pointer") {
    return ["mousemove", "--sync", String(command.x), String(command.y)];
  }
  if (command.action === "click") {
    const button = { left: "1", middle: "2", right: "3" }[command.button!];
    return ["click", "--repeat", "1", button];
  }
  if (command.action === "scroll") {
    const args: string[] = [];
    const addAxis = (delta: number, negativeButton: string, positiveButton: string): void => {
      if (delta === 0) return;
      const repeat = Math.max(1, Math.min(100, Math.ceil(Math.abs(delta) / 120)));
      args.push("click", "--repeat", String(repeat), delta < 0 ? negativeButton : positiveButton);
    };
    addAxis(command.deltaY!, "5", "4");
    addAxis(command.deltaX!, "6", "7");
    return args;
  }
  if (command.action === "key-press") {
    const modifiers = command.modifiers!.map((modifier) => ({
      ctrl: "ctrl",
      alt: "alt",
      shift: "shift",
      meta: "super"
    })[modifier]);
    return ["key", "--clearmodifiers", [...modifiers, linuxKeyName(command.key!)].join("+")];
  }
  // xdotool uses getopt; the terminator prevents text from becoming an option.
  return ["type", "--clearmodifiers", "--delay", "0", "--", command.text!];
}

function permissionFailure(platform: NodeJS.Platform, outcome: DeviceProcessOutcome): boolean {
  const detail = `${outcome.stderr}\n${outcome.errorMessage ?? ""}`.toLocaleLowerCase();
  if (platform === "darwin") {
    return /not authorized|not permitted|-1743|accessibility|assistive|omni_accessibility_required/.test(detail);
  }
  if (platform === "win32") {
    return /access is denied|omni_input_denied|higher-integrity|secure desktop/.test(detail);
  }
  return false;
}

function actionFromUnknown(command: unknown): DeviceInputAction | "invalid" {
  if (command && typeof command === "object" && !Array.isArray(command)) {
    const action = (command as Record<string, unknown>).action;
    if (ACTIONS.includes(action as DeviceInputAction)) return action as DeviceInputAction;
  }
  return "invalid";
}

export class HostDeviceInputBackend {
  private readonly platform: NodeJS.Platform;
  private readonly runner: DeviceProcessRunner;
  private readonly environment: Readonly<Record<string, string | undefined>>;
  private readonly now: () => number;
  private readonly createId: () => string;

  constructor(options: DeviceInputBackendOptions = {}) {
    this.platform = options.platform ?? process.platform;
    this.runner = options.runner ?? runDeviceInputProcess;
    this.environment = options.environment ?? process.env;
    this.now = options.now ?? Date.now;
    this.createId = options.createId ?? randomUUID;
  }

  async detectCapabilities(signal?: AbortSignal): Promise<DeviceInputCapabilities> {
    const detectedAt = new Date(this.now()).toISOString();
    if (signal?.aborted) {
      return {
        platform: this.platform,
        backend: "unavailable",
        available: false,
        actions: allCapabilities(false, false, "unknown", "Capability detection was cancelled."),
        detectedAt,
        notes: ["No input command was run."]
      };
    }
    if (this.platform === "win32") {
      const probe = `$ErrorActionPreference = 'Stop'\nif (-not [Environment]::UserInteractive) { throw 'OMNI_NO_INTERACTIVE_DESKTOP' }\n[Console]::Out.Write('OMNI_INPUT_READY')`;
      const outcome = await this.runner(
        processInvocation(
          "powershell.exe",
          powershellArgs(probe),
          2_000,
          signal,
          this.environment
        )
      );
      const available = outcome.exitCode === 0 && outcome.stdout.includes("OMNI_INPUT_READY");
      const reason = available
        ? undefined
        : outcome.cancelled
          ? "Capability detection was cancelled."
          : "A noninteractive PowerShell/User32 desktop is unavailable.";
      return {
        platform: this.platform,
        backend: available ? "windows-user32" : "unavailable",
        available,
        actions: allCapabilities(true, available, available ? "not-required" : "unknown", reason),
        detectedAt,
        notes: [
          "Input is sent only to the current interactive desktop.",
          "Windows may block input to secure or higher-integrity applications."
        ]
      };
    }
    if (this.platform === "darwin") {
      const probe = "ObjC.import('ApplicationServices'); $.AXIsProcessTrusted() ? 'OMNI_ACCESS_GRANTED' : 'OMNI_ACCESS_REQUIRED';";
      const outcome = await this.runner(
        processInvocation(
          "/usr/bin/osascript",
          ["-l", "JavaScript", "-e", probe],
          2_000,
          signal,
          this.environment
        )
      );
      const trusted = outcome.exitCode === 0 && outcome.stdout.includes("OMNI_ACCESS_GRANTED");
      if (outcome.cancelled || signal?.aborted) {
        return {
          platform: this.platform,
          backend: "unavailable",
          available: false,
          actions: allCapabilities(true, false, "unknown", "Capability detection was cancelled."),
          detectedAt,
          notes: ["No input command was run."]
        };
      }
      if (!trusted) {
        const reason = "Enable Accessibility for Omni AGI Studio before allowing host input.";
        return {
          platform: this.platform,
          backend: "macos-accessibility",
          available: false,
          actions: allCapabilities(true, false, "required", reason),
          detectedAt,
          notes: [
            "Pointer actions use Core Graphics; key and text actions use System Events.",
            "Permission failures are returned explicitly and never treated as completed input."
          ]
        };
      }
      const systemEventsProbe = await this.runner(
        processInvocation(
          "/usr/bin/osascript",
          ["-e", "tell application \"System Events\" to get UI elements enabled"],
          2_000,
          signal,
          this.environment
        )
      );
      if (systemEventsProbe.cancelled || signal?.aborted) {
        return {
          platform: this.platform,
          backend: "unavailable",
          available: false,
          actions: allCapabilities(true, false, "unknown", "Capability detection was cancelled."),
          detectedAt,
          notes: ["No input command was run."]
        };
      }
      const systemEventsAvailable =
        systemEventsProbe.exitCode === 0 && /\btrue\b/i.test(systemEventsProbe.stdout);
      const pointerCapability: DeviceInputActionCapability = {
        supported: true,
        available: true,
        permission: "granted"
      };
      const systemEventsCapability: DeviceInputActionCapability = {
        supported: true,
        available: systemEventsAvailable,
        permission: systemEventsAvailable ? "granted" : "required",
        reason: systemEventsAvailable
          ? undefined
          : "Allow Omni AGI Studio to control System Events for keyboard and text input."
      };
      return {
        platform: this.platform,
        backend: "macos-accessibility",
        available: true,
        actions: {
          "move-pointer": { ...pointerCapability },
          click: { ...pointerCapability },
          scroll: { ...pointerCapability },
          "key-press": { ...systemEventsCapability },
          text: { ...systemEventsCapability }
        },
        detectedAt,
        notes: [
          "Pointer actions use Core Graphics; key and text actions use System Events.",
          "Permission failures are returned explicitly and never treated as completed input."
        ]
      };
    }
    if (this.platform === "linux") {
      const display = this.environment.DISPLAY?.trim();
      const sessionType = this.environment.XDG_SESSION_TYPE?.trim().toLocaleLowerCase();
      if (!display || sessionType === "wayland") {
        const reason = !display
          ? "No X11 display is available."
          : "xdotool is X11-only; host-wide input is not exposed for a Wayland session.";
        return {
          platform: this.platform,
          backend: "unavailable",
          available: false,
          actions: allCapabilities(true, false, "not-required", reason),
          detectedAt,
          notes: ["Install and run xdotool inside an X11 session to enable host input."]
        };
      }
      const outcome = await this.runner(
        processInvocation(
          "xdotool",
          ["getmouselocation", "--shell"],
          2_000,
          signal,
          this.environment
        )
      );
      const available = outcome.exitCode === 0;
      const reason = available
        ? undefined
        : outcome.errorCode === "ENOENT"
          ? "xdotool is not installed."
          : outcome.cancelled
            ? "Capability detection was cancelled."
            : "xdotool could not connect to the active X11/XTEST session.";
      return {
        platform: this.platform,
        backend: available ? "linux-xdotool" : "unavailable",
        available,
        actions: allCapabilities(true, available, "not-required", reason),
        detectedAt,
        notes: ["Linux host input is intentionally limited to xdotool on X11."]
      };
    }
    return {
      platform: this.platform,
      backend: "unavailable",
      available: false,
      actions: allCapabilities(false, false, "unknown", `Host input is unsupported on ${this.platform}.`),
      detectedAt,
      notes: ["No native process was started."]
    };
  }

  async execute(
    command: DeviceInputCommand,
    context: DeviceInputExecutionContext = {}
  ): Promise<DeviceInputResult> {
    const started = this.now();
    const startedAt = new Date(started).toISOString();
    const id = this.createId();
    let normalized: NormalizedCommand;
    try {
      normalized = normalizeCommand(command);
    } catch (error) {
      return this.result({
        id,
        action: actionFromUnknown(command),
        state: "failed",
        started,
        startedAt,
        backend: "unavailable",
        request: {},
        errorCode: "invalid-command",
        error: error instanceof Error ? error.message : "Device input command is invalid."
      });
    }
    const authorization = context.authorization;
    if (
      !authorization ||
      authorization.granted !== true ||
      !["ask", "auto", "full-authority"].includes(authorization.policy) ||
      !/^[a-z0-9][a-z0-9._:-]{0,127}$/i.test(authorization.decisionId)
    ) {
      return this.result({
        id,
        action: normalized.action,
        state: "denied",
        started,
        startedAt,
        backend: "unavailable",
        request: normalized.auditRequest,
        errorCode: "authorization-required",
        error: "A current audited tool-policy decision is required before host input."
      });
    }
    if (context.signal?.aborted) {
      return this.result({
        id,
        action: normalized.action,
        state: "cancelled",
        started,
        startedAt,
        backend: "unavailable",
        request: normalized.auditRequest,
        authorization,
        errorCode: "cancelled",
        error: "Device input was cancelled before execution."
      });
    }
    const capabilities = await this.detectCapabilities(context.signal);
    const capability = capabilities.actions[normalized.action];
    if (!capability.available) {
      const permissionRequired = capability.permission === "required";
      const cancelled = context.signal?.aborted;
      return this.result({
        id,
        action: normalized.action,
        state: cancelled ? "cancelled" : permissionRequired ? "denied" : "unavailable",
        started,
        startedAt,
        backend: capabilities.backend,
        request: normalized.auditRequest,
        authorization,
        errorCode: cancelled
          ? "cancelled"
          : permissionRequired
            ? "permission-required"
            : "capability-unavailable",
        error: capability.reason ?? "This host input action is unavailable."
      });
    }
    let invocation: DeviceProcessInvocation;
    try {
      invocation = this.invocation(normalized, context.signal);
    } catch (error) {
      return this.result({
        id,
        action: normalized.action,
        state: "unavailable",
        started,
        startedAt,
        backend: capabilities.backend,
        request: normalized.auditRequest,
        authorization,
        errorCode: "capability-unavailable",
        error: error instanceof Error ? error.message : "This action is unavailable."
      });
    }
    const outcome = await this.runner(invocation);
    const processAudit = {
      executable: invocation.executable,
      exitCode: outcome.exitCode,
      cancelled: Boolean(outcome.cancelled),
      timedOut: Boolean(outcome.timedOut),
      outputTruncated: Boolean(outcome.outputTruncated)
    };
    if (outcome.cancelled || context.signal?.aborted) {
      return this.result({
        id,
        action: normalized.action,
        state: "cancelled",
        started,
        startedAt,
        backend: capabilities.backend,
        request: normalized.auditRequest,
        authorization,
        process: processAudit,
        errorCode: "cancelled",
        error: "Device input was cancelled."
      });
    }
    if (outcome.timedOut) {
      return this.result({
        id,
        action: normalized.action,
        state: "failed",
        started,
        startedAt,
        backend: capabilities.backend,
        request: normalized.auditRequest,
        authorization,
        process: processAudit,
        errorCode: "timed-out",
        error: `Device input timed out after ${normalized.timeoutMs} ms.`
      });
    }
    if (permissionFailure(this.platform, outcome)) {
      return this.result({
        id,
        action: normalized.action,
        state: "denied",
        started,
        startedAt,
        backend: capabilities.backend,
        request: normalized.auditRequest,
        authorization,
        process: processAudit,
        errorCode: "permission-required",
        error: this.platform === "darwin"
          ? "macOS denied Accessibility or System Events automation access."
          : "Windows blocked input to the current secure or higher-integrity desktop."
      });
    }
    if (outcome.exitCode !== 0) {
      return this.result({
        id,
        action: normalized.action,
        state: "failed",
        started,
        startedAt,
        backend: capabilities.backend,
        request: normalized.auditRequest,
        authorization,
        process: processAudit,
        errorCode: "backend-failed",
        error: "The native input backend did not complete the requested action."
      });
    }
    return this.result({
      id,
      action: normalized.action,
      state: "complete",
      started,
      startedAt,
      backend: capabilities.backend,
      request: normalized.auditRequest,
      authorization,
      process: processAudit
    });
  }

  private invocation(
    command: NormalizedCommand,
    signal?: AbortSignal
  ): DeviceProcessInvocation {
    if (this.platform === "win32") {
      return processInvocation(
        "powershell.exe",
        powershellArgs(windowsScript(command)),
        command.timeoutMs,
        signal,
        this.environment,
        command.action === "text" ? command.text : undefined
      );
    }
    if (this.platform === "darwin") {
      const args = command.action === "key-press" || command.action === "text"
        ? macSystemEventsArgs(command)
        : ["-l", "JavaScript", "-e", macCoreGraphicsScript(command)];
      return processInvocation(
        "/usr/bin/osascript",
        args,
        command.timeoutMs,
        signal,
        this.environment
      );
    }
    if (this.platform === "linux") {
      return processInvocation(
        "xdotool",
        linuxArgs(command),
        command.timeoutMs,
        signal,
        this.environment
      );
    }
    throw new Error(`Host input is unsupported on ${this.platform}.`);
  }

  private result(input: {
    id: string;
    action: DeviceInputAction | "invalid";
    state: DeviceInputResultState;
    started: number;
    startedAt: string;
    backend: DeviceInputBackend;
    request: Record<string, string | number | string[]>;
    authorization?: DeviceInputAuthorization;
    process?: DeviceInputAuditRecord["process"];
    errorCode?: DeviceInputResult["errorCode"];
    error?: string;
  }): DeviceInputResult {
    const completed = this.now();
    const audit: DeviceInputAuditRecord = {
      id: input.id,
      action: input.action,
      state: input.state,
      platform: this.platform,
      backend: input.backend,
      startedAt: input.startedAt,
      completedAt: new Date(completed).toISOString(),
      durationMs: Math.max(0, completed - input.started),
      policyDecisionId: input.authorization?.decisionId,
      policy: input.authorization?.policy,
      request: input.request,
      process: input.process
    };
    return {
      ok: input.state === "complete",
      state: input.state,
      action: input.action,
      errorCode: input.errorCode,
      error: input.error,
      audit
    };
  }
}
