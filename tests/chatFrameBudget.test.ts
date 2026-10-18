import { describe, expect, it } from "vitest";
import { createTextFrameBatcher } from "../src/renderer/src/chatStreaming";

function fixture(limit?: number) {
  const callbacks = new Map<number, () => void>(), commits: string[] = [];
  let next = 0;
  const batcher = createTextFrameBatcher(delta => commits.push(delta), callback => {
    const id = ++next; callbacks.set(id, () => { callbacks.delete(id); callback(); }); return id;
  }, id => { callbacks.delete(id); }, limit);
  return { batcher, callbacks, commits, frame() { const first = callbacks.values().next().value; first?.(); } };
}
describe("ordinary output does not inherit readable-speed pacing", () => {
  it("publishes a large received burst in one frame by default without changing any bytes", () => {
    const { batcher, callbacks, commits, frame } = fixture(); const burst = "Observed output 😀\n".repeat(1000);
    for (let index = 0; index < burst.length; index += 500) batcher.push(burst.slice(index, index + 500));
    expect(callbacks.size).toBe(1); expect(commits).toEqual([]); frame();
    expect(commits).toEqual([burst]); expect(callbacks.size).toBe(0);
  });
  it("keeps explicitly requested pacing independent and never splits a surrogate pair", () => {
    const { batcher, commits, frame } = fixture(2); batcher.push("a😀bc"); frame();
    expect(commits).toEqual(["a😀"]); frame(); expect(commits).toEqual(["a😀", "bc"]);
  });
  it("flushes a paced suffix immediately on the terminal boundary and clears its scheduled work", () => {
    const { batcher, callbacks, commits, frame } = fixture(2); batcher.push("abcdef"); frame(); batcher.flush();
    expect(commits).toEqual(["ab", "cdef"]); expect(callbacks.size).toBe(0);
    batcher.push("new"); batcher.reset(); expect(callbacks.size).toBe(0); expect(commits.join("")).toBe("abcdef");
  });
  it("refuses invalid pacing budgets instead of scheduling a never-draining loop", () => {
    for (const limit of [0, -1, NaN, Number.NEGATIVE_INFINITY]) expect(() => fixture(limit)).toThrow();
  });
});
