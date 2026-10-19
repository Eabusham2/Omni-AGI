import { createHash } from "node:crypto";
import { describe, expect, it, vi } from "vitest";
import { splitPublicReplyAtActions, verifyReplyActionLocations, type PublicReplySource } from "../src/renderer/src/chatReplyProjection";
import type { ActionEvent } from "../src/shared/types";

const digest = (text: string): string => createHash("sha256").update(text).digest("hex");
const hash = async (text: string): Promise<string> => digest(text);
const source = (content = "Opening. Continuation.", live = false): PublicReplySource => ({
  key: live ? "live-turn" : "message-reply", turnId: "turn", ...(live ? {} : { messageId: "reply" }), content, live
});
function action(id: string, prefix = "Opening. ", live = false): ActionEvent {
  return { id, brainId: "owner", chatTurnId: "turn", neuralActionId: "a".repeat(32),
    action: { kind: "ponder", source: "brain", arguments: {} }, state: "running",
    createdAt: "2026-10-18T12:00:00.000Z", updatedAt: "2026-10-18T12:00:00.000Z",
    replyAnchor: { turnId: "turn", ...(live ? {} : { messageId: "reply" }), utf16Offset: prefix.length, prefixSha256: digest(prefix) } };
}

describe("exact public-reply action projection", () => {
  it("verifies a canonical reply identity/prefix and preserves every character while interleaving", async () => {
    const value = source(), ponder = action("ponder"), before = JSON.stringify([value, ponder]);
    const verified = await verifyReplyActionLocations("owner", [value], [ponder], hash);
    expect(verified.get(ponder.id)).toEqual({ sourceKey: value.key, utf16Offset: 9 });
    const result = splitPublicReplyAtActions(value.content, [ponder], new Map([[ponder.id, verified.get(ponder.id)!.utf16Offset]]));
    expect(result.pieces.map(piece => piece.kind === "text" ? piece.text : piece.event.id)).toEqual(["Opening. ", "ponder", "Continuation."]);
    expect(result.pieces.flatMap(piece => piece.kind === "text" ? [piece.text] : []).join("")).toBe(value.content);
    expect(JSON.stringify([value, ponder])).toBe(before);
  });

  it("allows an unbound anchor only on the exact live turn, never on saved history", async () => {
    const value = source(undefined, true), ponder = action("live", undefined, true);
    expect((await verifyReplyActionLocations("owner", [value], [ponder], hash)).size).toBe(1);
    expect((await verifyReplyActionLocations("owner", [source()], [ponder], hash)).size).toBe(0);
    expect((await verifyReplyActionLocations("owner", [{ ...value, turnId: "other" }], [ponder], hash)).size).toBe(0);
    expect((await verifyReplyActionLocations("owner", [value], [action("bound")], hash)).size).toBe(0);
  });

  it("rejects wrong brain/turn/message identities and unchanged-looking but different text", async () => {
    const original = action("p");
    for (const event of [
      { ...original, brainId: "other" }, { ...original, chatTurnId: "other" },
      { ...original, replyAnchor: { ...original.replyAnchor!, messageId: "other" } },
      { ...original, replyAnchor: { ...original.replyAnchor!, prefixSha256: "0".repeat(64) } }
    ]) expect((await verifyReplyActionLocations("owner", [source()], [event], hash)).size).toBe(0);
    expect((await verifyReplyActionLocations("owner", [source("Changed. Continuation.")], [original], hash)).size).toBe(0);
    expect((await verifyReplyActionLocations("owner", [source()], [{ ...original, replyAnchor: undefined }], hash)).size).toBe(0);
  });

  it("does not guess trimming or substrings in the renderer", async () => {
    const raw = action("raw", " Opening. ");
    expect((await verifyReplyActionLocations("owner", [source()], [raw], hash)).size).toBe(0);
    const canonical = action("mapped-by-host");
    expect((await verifyReplyActionLocations("owner", [source()], [canonical], hash)).size).toBe(1);
  });

  it("uses UTF16 character boundaries without splitting a surrogate pair", async () => {
    const content = "A🧠 B", good = action("good", "A🧠"), bad = action("bad", content.slice(0, 2));
    expect((await verifyReplyActionLocations("owner", [source(content)], [good], hash)).size).toBe(1);
    expect((await verifyReplyActionLocations("owner", [source(content)], [bad], hash)).size).toBe(0);
    for (const offset of [-1, 1.5, 100, Number.MAX_SAFE_INTEGER + 1]) {
      expect((await verifyReplyActionLocations("owner", [source(content)], [{ ...good, replyAnchor: { ...good.replyAnchor!, utf16Offset: offset } }], hash)).size).toBe(0);
    }
  });

  it("refuses duplicate action or reply identity instead of choosing an arbitrary match", async () => {
    const event = action("duplicate"), value = source();
    expect((await verifyReplyActionLocations("owner", [value], [event, { ...event }], hash)).size).toBe(0);
    expect((await verifyReplyActionLocations("owner", [value, { ...value }], [event], hash)).size).toBe(0);
  });

  it("hashes each exact shared prefix once and keeps stable order at an identical offset", async () => {
    const events = [action("first"), action("second")], sha = vi.fn(hash), value = source();
    const verified = await verifyReplyActionLocations("owner", [value], events, sha);
    expect(verified.size).toBe(2); expect(sha).toHaveBeenCalledTimes(1);
    const split = splitPublicReplyAtActions(value.content, events, new Map(events.map(event => [event.id, 9])));
    expect(split.pieces.filter(piece => piece.kind === "action").map(piece => piece.event.id)).toEqual(["first", "second"]);
  });

  it("refuses reused source keys across different turns and empty bound identities", async () => {
    const value = source(), event = action("one");
    const other = { ...value, turnId: "other", messageId: "other", content: "Different content" };
    expect((await verifyReplyActionLocations("owner", [value, other], [event], hash)).size).toBe(0);
    expect((await verifyReplyActionLocations("owner", [{ ...value, messageId: "" }],
      [{ ...event, replyAnchor: { ...event.replyAnchor!, messageId: "" } }], hash)).size).toBe(0);
  });

  it("handles start/end and invalid offsets without inventing or discarding public text", () => {
    const events = [action("start", ""), action("end", "Text"), action("invalid")];
    const result = splitPublicReplyAtActions("Text", events, new Map([["start", 0], ["end", 4], ["invalid", 99]]));
    expect(result.pieces.map(piece => piece.kind === "text" ? piece.text : piece.event.id)).toEqual(["start", "Text", "end"]);
    expect([...result.includedActionIds]).toEqual(["start", "end"]);
  });

  it("does not convert a hashing failure into an accepted anchor", async () => {
    await expect(verifyReplyActionLocations("owner", [source()], [action("p")], async () => { throw new Error("Hash unavailable"); })).rejects.toThrow("Hash unavailable");
  });
});
