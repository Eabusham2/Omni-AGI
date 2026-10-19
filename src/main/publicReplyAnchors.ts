import { createHash } from "node:crypto";
import type { ActionEvent, ActionReplyAnchor, ChatMessage, PublicReplyTextBinding } from "../shared/types";

const sha256 = (value: string): string => createHash("sha256").update(value).digest("hex");

export function normalizePublicReplyTextBinding(value: unknown): PublicReplyTextBinding | undefined {
  if (!value || typeof value !== "object" || Array.isArray(value)) return undefined;
  const binding = value as PublicReplyTextBinding;
  if (binding.format !== "omni-public-reply-text-binding" || binding.version !== 1 ||
      !["identity", "strip"].includes(binding.normalization) ||
      ![binding.rawSha256, binding.canonicalSha256].every(hash => typeof hash === "string" && /^[a-f0-9]{64}$/.test(hash)) ||
      ![binding.rawUtf16Length, binding.canonicalUtf16Length, binding.removedPrefixUtf16, binding.removedSuffixUtf16]
        .every(count => Number.isSafeInteger(count) && count >= 0) ||
      binding.canonicalUtf16Length + binding.removedPrefixUtf16 + binding.removedSuffixUtf16 !== binding.rawUtf16Length ||
      binding.normalization === "identity" && (binding.removedPrefixUtf16 !== 0 || binding.removedSuffixUtf16 !== 0)) return undefined;
  return { format: binding.format, version: 1, normalization: binding.normalization,
    rawSha256: binding.rawSha256, canonicalSha256: binding.canonicalSha256,
    rawUtf16Length: binding.rawUtf16Length, canonicalUtf16Length: binding.canonicalUtf16Length,
    removedPrefixUtf16: binding.removedPrefixUtf16, removedSuffixUtf16: binding.removedSuffixUtf16 };
}

/** Main-only measurements of actual public delta order, never thought text. */
export class PublicReplyAnchors {
  private readonly chunks: string[];
  private readonly digest;
  private length: number;
  private readonly measured = new WeakMap<ActionEvent, ActionReplyAnchor>();

  constructor(private readonly turnId: string, retainedPublishedText = "") {
    this.chunks = [retainedPublishedText]; this.length = retainedPublishedText.length;
    this.digest = createHash("sha256").update(retainedPublishedText);
  }

  append(delta: string): void {
    this.chunks.push(delta); this.length += delta.length; this.digest.update(delta);
  }

  measure(event: ActionEvent): void {
    const anchor = { turnId: this.turnId, utf16Offset: this.length, prefixSha256: this.digest.copy().digest("hex") };
    this.measured.set(event, anchor); event.replyAnchor = { ...anchor };
  }

  bind(events: ActionEvent[], message: ChatMessage, candidate?: PublicReplyTextBinding): void {
    const raw = this.chunks.join("");
    const binding = normalizePublicReplyTextBinding(candidate);
    const start = binding?.removedPrefixUtf16 ?? 0;
    const end = raw.length - (binding?.removedSuffixUtf16 ?? 0);
    const sameOwner = message.role === "brain" && message.turnId === this.turnId && typeof message.id === "string" && message.id.length > 0;
    const exact = sameOwner && (binding
      ? binding.rawUtf16Length === raw.length && binding.canonicalUtf16Length === message.content.length &&
        binding.rawSha256 === sha256(raw) && binding.canonicalSha256 === sha256(message.content) && raw.slice(start, end) === message.content
      : candidate === undefined && raw === message.content);
    if (!exact) {
      for (const event of events) delete event.replyAnchor;
      return;
    }
    const rows = events.flatMap(event => {
      const mark = this.measured.get(event);
      if (!mark || mark.turnId !== this.turnId) { delete event.replyAnchor; return []; }
      return [{ event, offset: Math.max(0, Math.min(message.content.length, mark.utf16Offset - start)) }];
    }).sort((left, right) => left.offset - right.offset);
    const prefix = createHash("sha256");
    let prior = 0;
    for (const { event, offset } of rows) {
      prefix.update(message.content.slice(prior, offset)); prior = offset;
      const bound = { turnId: this.turnId, messageId: message.id, utf16Offset: offset, prefixSha256: prefix.copy().digest("hex") };
      if (JSON.stringify(event.replyAnchor) !== JSON.stringify(bound)) {
        // An earlier immutable approval/decline revision may already exist
        // without its unfinished anchor. Binding is a distinct append-only
        // presentation revision, not permission to alter that stored row.
        const priorUpdate = Date.parse(event.updatedAt);
        event.updatedAt = new Date(Math.max(Date.now(), Number.isFinite(priorUpdate) ? priorUpdate + 1 : 0)).toISOString();
      }
      event.replyAnchor = bound;
    }
  }
}
