import type { ActionEvent } from "../../shared/types";

export interface PublicReplySource {
  key: string;
  turnId: string;
  messageId?: string;
  content: string;
  live: boolean;
}
export interface VerifiedReplyLocation { sourceKey: string; utf16Offset: number }
export type ReplyPiece = { kind: "text"; start: number; end: number; text: string } |
  { kind: "action"; offset: number; event: ActionEvent };

export async function publicReplyPrefixSha256(text: string): Promise<string> {
  const bytes = new Uint8Array(await globalThis.crypto.subtle.digest("SHA-256", new TextEncoder().encode(text)));
  return Array.from(bytes, byte => byte.toString(16).padStart(2, "0")).join("");
}

function characterBoundary(text: string, offset: number): boolean {
  if (!Number.isSafeInteger(offset) || offset < 0 || offset > text.length) return false;
  const before = text.charCodeAt(offset - 1), after = text.charCodeAt(offset);
  return !(before >= 0xd800 && before <= 0xdbff && after >= 0xdc00 && after <= 0xdfff);
}

/** Only exact host-bound public prefixes authorize splitting visible prose. */
export async function verifyReplyActionLocations(brainId: string, sources: readonly PublicReplySource[],
  events: readonly ActionEvent[], sha256 = publicReplyPrefixSha256): Promise<Map<string, VerifiedReplyLocation>> {
  const verified = new Map<string, VerifiedReplyLocation>();
  const counts = new Map<string, number>(), sourceCounts = new Map<string, number>(), digests = new Map<string, Promise<string>>();
  for (const source of sources) sourceCounts.set(source.key, (sourceCounts.get(source.key) ?? 0) + 1);
  for (const event of events) counts.set(event.id, (counts.get(event.id) ?? 0) + 1);
  for (const event of events) {
    const anchor = event.replyAnchor;
    if (!anchor || counts.get(event.id) !== 1 || event.brainId !== brainId || !anchor.turnId ||
        event.chatTurnId !== anchor.turnId || (anchor.messageId !== undefined && !anchor.messageId) ||
        !/^[a-f0-9]{64}$/.test(anchor.prefixSha256)) continue;
    const matches = sources.filter(source => source.turnId === anchor.turnId &&
      (anchor.messageId !== undefined ? source.messageId === anchor.messageId && !source.live : source.live));
    if (matches.length !== 1) continue;
    const source = matches[0]!;
    if (!source.key || sourceCounts.get(source.key) !== 1 || !characterBoundary(source.content, anchor.utf16Offset)) continue;
    const key = `${source.key}:${anchor.utf16Offset}`;
    let digest = digests.get(key);
    if (!digest) { digest = sha256(source.content.slice(0, anchor.utf16Offset)); digests.set(key, digest); }
    if (await digest === anchor.prefixSha256) verified.set(event.id, { sourceKey: source.key, utf16Offset: anchor.utf16Offset });
  }
  return verified;
}

/** The canonical message is unchanged; this is a reversible presentation only. */
export function splitPublicReplyAtActions(content: string, events: readonly ActionEvent[],
  offsets: ReadonlyMap<string, number>): { pieces: ReplyPiece[]; includedActionIds: Set<string> } {
  const positioned = events.map((event, order) => ({ event, order, offset: offsets.get(event.id) }))
    .filter((row): row is { event: ActionEvent; order: number; offset: number } => row.offset !== undefined && characterBoundary(content, row.offset))
    .sort((a, b) => a.offset - b.offset || a.order - b.order);
  const pieces: ReplyPiece[] = [], includedActionIds = new Set<string>();
  let start = 0;
  for (const { event, offset } of positioned) {
    if (includedActionIds.has(event.id)) continue;
    if (offset > start) pieces.push({ kind: "text", start, end: offset, text: content.slice(start, offset) });
    pieces.push({ kind: "action", offset, event }); includedActionIds.add(event.id); start = offset;
  }
  if (start < content.length) pieces.push({ kind: "text", start, end: content.length, text: content.slice(start) });
  return { pieces, includedActionIds };
}
