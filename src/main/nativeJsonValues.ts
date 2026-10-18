/** Targeted JSON value replacement without changing native numeric lexemes.
 * Python checkpoint hashes distinguish 1.0 from 1; parse/stringify is unsafe.
 * Replacements address structural paths, never arbitrary user text matches.
 */
export function rewriteNativeJsonValues(text: string, replacements: ReadonlyMap<string, string>): string {
  JSON.parse(text); // Full JSON syntax validation before positional traversal.
  const edits: Array<{ start: number; end: number; value: string }> = [];
  const found = new Set<string>();
  let cursor = 0;
  const whitespace = (): void => { while (cursor < text.length && /\s/.test(text[cursor]!)) cursor += 1; };
  const string = (): string => {
    const start = cursor++;
    while (cursor < text.length) {
      const char = text[cursor++];
      if (char === "\\") cursor += 1;
      else if (char === '"') return JSON.parse(text.slice(start, cursor)) as string;
    }
    throw new Error("Native JSON has an incomplete string.");
  };
  const value = (path: string[]): void => {
    whitespace();
    const start = cursor;
    if (text[cursor] === "{") {
      cursor += 1; whitespace(); const keys = new Set<string>();
      while (text[cursor] !== "}") {
        const key = string();
        if (keys.has(key)) throw new Error("Native JSON contains duplicate keys.");
        keys.add(key); whitespace(); cursor += 1; value([...path, key]); whitespace();
        if (text[cursor] !== ",") break;
        cursor += 1; whitespace();
      }
      cursor += 1;
    } else if (text[cursor] === "[") {
      cursor += 1; whitespace(); let index = 0;
      while (text[cursor] !== "]") {
        value([...path, String(index++)]); whitespace();
        if (text[cursor] !== ",") break;
        cursor += 1;
      }
      cursor += 1;
    } else if (text[cursor] === '"') string();
    else while (cursor < text.length && !/[\s,}\]]/.test(text[cursor]!)) cursor += 1;
    const key = JSON.stringify(path), replacement = replacements.get(key);
    if (replacement !== undefined) { JSON.parse(replacement); edits.push({ start, end: cursor, value: replacement }); found.add(key); }
  };
  value([]); whitespace();
  if (cursor !== text.length || found.size !== replacements.size) throw new Error("Native JSON replacement path is unavailable.");
  const sorted = edits.sort((a, b) => a.start - b.start);
  for (let index = 1; index < sorted.length; index += 1) if (sorted[index]!.start < sorted[index - 1]!.end) throw new Error("Native JSON replacement paths overlap.");
  for (const edit of sorted.reverse()) text = text.slice(0, edit.start) + edit.value + text.slice(edit.end);
  return text;
}
