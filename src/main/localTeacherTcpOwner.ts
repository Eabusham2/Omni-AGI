import { execFile } from "node:child_process";
import type { Socket } from "node:net";
import type { LocalTeacherChild } from "./localTeacherRunner";
import { cleanLocalTeacherEnvironment } from "./localTeacherRunner";
import { tmpdir } from "node:os";

export function validateLocalTeacherTcpOwner(value: unknown, pid: number, port: number, clientPort: number): void {
  const rows = Array.isArray(value) ? value : [value];
  const listener = rows.some(row => row && row.OwningProcess === pid && row.LocalPort === port && row.LocalAddress === "127.0.0.1" && row.State === "Listen");
  const peer = rows.some(row => row && row.OwningProcess === pid && row.LocalPort === port && row.LocalAddress === "127.0.0.1" &&
    row.RemoteAddress === "127.0.0.1" && row.RemotePort === clientPort && row.State === "Established");
  if (!listener || !peer) throw new Error("The connected teacher socket cannot be authenticated to the exact live child; no prompt/auth was sent.");
}

/** Fixed read-only OS query. No elevation, arbitrary shell, process scan or remote bind. */
export async function verifyLocalTeacherTcpOwner(child: LocalTeacherChild, port: number, socket: Socket, signal: AbortSignal): Promise<void> {
  if (process.platform !== "win32") throw new Error("Owned local TCP verification is Windows-only.");
  const pid = child.child.pid, clientPort = socket.localPort;
  if (!child.alive() || !pid || !clientPort || ![pid, port, clientPort].every(value => Number.isSafeInteger(value) && value > 0) ||
    socket.remoteAddress !== "127.0.0.1" || socket.remotePort !== port) throw new Error("Selected teacher TCP identity is unavailable.");
  signal.throwIfAborted();
  const script = `$ErrorActionPreference='Stop'; @(Get-NetTCPConnection -LocalAddress '127.0.0.1' -LocalPort ${port} | ` +
    `Where-Object { $_.OwningProcess -eq ${pid} } | Select-Object OwningProcess,LocalAddress,LocalPort,RemoteAddress,RemotePort,` +
    `@{Name='State';Expression={$_.State.ToString()}}) | ConvertTo-Json -Compress`;
  const stdout = await new Promise<string>((resolve, reject) => execFile("powershell.exe", ["-NoLogo", "-NoProfile", "-NonInteractive", "-Command", script],
    { signal, timeout: 3000, maxBuffer: 65536, windowsHide: true, env: cleanLocalTeacherEnvironment(tmpdir()) }, (error, output) => error ? reject(error) : resolve(output)));
  signal.throwIfAborted(); if (!child.alive()) throw new Error("Selected teacher exited during socket-owner verification.");
  validateLocalTeacherTcpOwner(JSON.parse(stdout), pid, port, clientPort);
}
