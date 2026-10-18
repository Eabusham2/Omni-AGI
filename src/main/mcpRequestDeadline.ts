/** Network waiting budget excludes explicit human/native host-request review. */
export class McpRequestDeadline {
  private readonly controller = new AbortController();
  private timer?: NodeJS.Timeout;
  private began = 0;
  private remaining: number;
  private holds = 0;
  private closed = false;
  readonly signal = this.controller.signal;
  constructor(milliseconds: number, private readonly clock = (): number => performance.now()) {
    this.remaining = milliseconds; this.start();
  }
  private start(): void {
    if (this.closed || this.holds || this.signal.aborted) return;
    this.began = this.clock();
    this.timer = setTimeout(() => { this.timer = undefined; this.controller.abort(new Error("MCP network request timed out.")); }, Math.max(0, this.remaining));
    this.timer.unref?.();
  }
  pause(): () => void {
    if (this.closed || this.signal.aborted) return () => undefined;
    if (this.holds++ === 0 && this.timer) { clearTimeout(this.timer); this.timer = undefined; this.remaining = Math.max(0, this.remaining - (this.clock() - this.began)); }
    let released = false;
    return () => { if (released) return; released = true; this.holds = Math.max(0, this.holds - 1); this.start(); };
  }
  close(): void { this.closed = true; if (this.timer) clearTimeout(this.timer); this.timer = undefined; }
}
