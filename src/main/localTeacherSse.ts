/** Raw llama-server completion deltas, not a chat/template or console parser. */
export class LocalTeacherSse {
  private readonly decoder = new TextDecoder("utf-8", { fatal: true });
  private pending = "";
  private terminal?: Record<string, unknown>;
  private bytes = 0;
  constructor(private readonly maximumContent: number, private readonly maximumFrame: number,
    private readonly append: (bytes: Buffer) => Promise<void>) {}
  get contentBytes(): number { return this.bytes; }
  async push(chunk: Uint8Array): Promise<void> {
    this.pending += this.decoder.decode(chunk, { stream: true });
    for (;;) {
      const match = /\r?\n\r?\n/.exec(this.pending); if (!match) break;
      const event = this.pending.slice(0, match.index); this.pending = this.pending.slice(match.index + match[0].length);
      if (Buffer.byteLength(event) > this.maximumFrame) throw new Error("Teacher SSE frame exceeds the admitted parse envelope.");
      const parts = event.split(/\r?\n/).filter(line => line.startsWith("data:")).map(line => line.slice(5).replace(/^ /, ""));
      if (!parts.length) continue;
      if (this.terminal) throw new Error("Teacher emitted data after its terminal completion frame.");
      const data = JSON.parse(parts.join("\n")) as Record<string, unknown>;
      if (!data || Array.isArray(data) || data.error || typeof data.content !== "string" || typeof data.stop !== "boolean")
        throw new Error("Teacher SSE content/terminal schema is invalid.");
      if (data.truncated === true) throw new Error("Teacher reported a truncated prompt/context; no clipped example is eligible.");
      // Current verified llama-server streaming final frames have no content.
      // Refuse an aggregate/unknown variant instead of duplicating its answer.
      if (data.stop && data.content !== "") throw new Error("Teacher terminal-content semantics are unsupported; no content was guessed or stripped.");
      for (const point of data.content) if (point.length === 1 && point.charCodeAt(0) >= 0xd800 && point.charCodeAt(0) <= 0xdfff)
        throw new Error("Teacher emitted invalid Unicode; no replacement bytes were learned.");
      const bytes = Buffer.from(data.content);
      if (this.bytes + bytes.length > this.maximumContent) throw new Error("Teacher output exceeded its explicit byte envelope; retained prefix is not a complete source.");
      await this.append(bytes); this.bytes += bytes.length;
      if (data.stop) this.terminal = data;
    }
    if (Buffer.byteLength(this.pending) > this.maximumFrame) throw new Error("Teacher SSE frame exceeds the admitted parse envelope.");
  }
  finish(): Record<string, unknown> {
    this.pending += this.decoder.decode();
    if (this.pending.trim()) throw new Error("Teacher SSE ended with an incomplete frame.");
    if (!this.terminal || !this.bytes) throw new Error("Teacher did not provide complete nonempty generated content.");
    return this.terminal;
  }
}
