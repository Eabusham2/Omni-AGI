/** Normalize a real user input; capacity is admitted by the runtime, not a character cap. */
export function cleanChatInput(value: string): string {
  const clean = value.replace(/\0/g, "").trim();
  if (!clean) throw new Error("A chat message cannot be empty.");
  return clean;
}

/** UTF-8 byte-token count without allocating a second whole encoded message. */
export function utf8InputTokenCount(value: string): number {
  let count = 0;
  for (let index = 0; index < value.length; index += 1) {
    const unit = value.charCodeAt(index);
    if (unit < 0x80) count += 1;
    else if (unit < 0x800) count += 2;
    else if (unit >= 0xd800 && unit <= 0xdbff && index + 1 < value.length &&
      value.charCodeAt(index + 1) >= 0xdc00 && value.charCodeAt(index + 1) <= 0xdfff) {
      count += 4;
      index += 1;
    } else count += 3;
  }
  return count;
}

export function chatInputCapacity(value: string, selectedContextTokens: number) {
  const clean = value.replace(/\0/g, "").trim();
  const payloadTokens = utf8InputTokenCount(clean);
  // Native byte-token prompt: BOS + actual HUMAN input + BRAIN boundary.
  const requiredTokens = payloadTokens + 3;
  const known = Number.isSafeInteger(selectedContextTokens) && selectedContextTokens >= 3;
  const fits = known && requiredTokens <= selectedContextTokens;
  const reason = fits ? undefined : known
    ? `This message needs ${requiredTokens.toLocaleString()} working-memory tokens including its boundaries; the selected window is ${selectedContextTokens.toLocaleString()}. Send is blocked rather than dropping part of your message. Increase working memory in Device & runtime, or add the material through Learn.`
    : "The selected working-memory window is unavailable. Send is blocked until its size is known.";
  return { fits, payloadTokens, requiredTokens, selectedContextTokens, reason };
}
