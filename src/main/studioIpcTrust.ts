/** No iframe, including a same-process one, may invoke Studio's preload API. */
export function requireStudioIpcSender(input: {
  senderId: number;
  trustedSenderId?: number;
  senderFrame: unknown;
  mainFrame: unknown;
  frameUrl?: string;
  rendererUrl: string;
}): void {
  if (input.trustedSenderId === undefined || input.senderId !== input.trustedSenderId ||
    !input.senderFrame || input.senderFrame !== input.mainFrame || !input.frameUrl) throw new Error("Studio IPC requires its trusted main frame.");
  try {
    const frame = new URL(input.frameUrl), expected = new URL(input.rendererUrl);
    frame.hash = ""; expected.hash = "";
    if (frame.href !== expected.href) throw new Error("different document");
  } catch { throw new Error("Studio IPC is unavailable to this document."); }
}
