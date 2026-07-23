import type { IdeaKind } from "../../shared/types";

const STOP_WORDS = new Set([
  "a",
  "about",
  "after",
  "again",
  "all",
  "also",
  "am",
  "an",
  "and",
  "any",
  "are",
  "as",
  "at",
  "be",
  "because",
  "been",
  "before",
  "being",
  "but",
  "by",
  "can",
  "could",
  "did",
  "do",
  "does",
  "for",
  "from",
  "get",
  "had",
  "has",
  "have",
  "he",
  "her",
  "here",
  "him",
  "his",
  "how",
  "i",
  "if",
  "in",
  "into",
  "is",
  "it",
  "its",
  "just",
  "like",
  "me",
  "more",
  "my",
  "no",
  "not",
  "of",
  "on",
  "or",
  "our",
  "out",
  "she",
  "so",
  "some",
  "that",
  "the",
  "their",
  "them",
  "then",
  "there",
  "these",
  "they",
  "this",
  "to",
  "up",
  "us",
  "was",
  "we",
  "were",
  "what",
  "when",
  "where",
  "which",
  "who",
  "why",
  "will",
  "with",
  "would",
  "you",
  "your"
]);

export interface ExtractedConcept {
  key: string;
  label: string;
  position: number;
  salience: number;
}

export function normalizeConcept(value: string): string {
  return value
    .normalize("NFKC")
    .toLocaleLowerCase()
    .replace(/^['’]+|['’]+$/g, "")
    .replace(/[_-]+/g, " ")
    .trim();
}

export function extractConcepts(text: string, limit = 64): ExtractedConcept[] {
  const matches = text.match(/[\p{L}\p{N}][\p{L}\p{N}'’_-]*/gu) ?? [];
  const seen = new Map<string, ExtractedConcept>();

  matches.forEach((raw, position) => {
    const key = normalizeConcept(raw);
    if (
      key.length < 2 ||
      STOP_WORDS.has(key) ||
      (/^\d+$/.test(key) && key.length < 3)
    ) {
      return;
    }

    const existing = seen.get(key);
    if (existing) {
      existing.salience = Math.min(1, existing.salience + 0.12);
      return;
    }

    const properNounBoost = /^[A-Z][\p{L}\p{N}]+$/u.test(raw) ? 0.18 : 0;
    const numberBoost = /\d/.test(raw) ? 0.12 : 0;
    const lengthBoost = Math.min(0.16, Math.max(0, key.length - 5) * 0.02);
    seen.set(key, {
      key,
      label: raw,
      position,
      salience: Math.min(1, 0.52 + properNounBoost + numberBoost + lengthBoost)
    });
  });

  return [...seen.values()]
    .sort((left, right) => right.salience - left.salience || left.position - right.position)
    .slice(0, limit)
    .sort((left, right) => left.position - right.position);
}

export function splitIntoIdeas(text: string, maxIdeas = 200): string[] {
  return text
    .replace(/\r\n/g, "\n")
    .split(/(?<=[.!?])\s+|\n{2,}|(?<=;)\s+/u)
    .map((part) => part.replace(/\s+/g, " ").trim())
    .filter((part) => part.length >= 12)
    .map((part) => part.slice(0, 700))
    .slice(0, maxIdeas);
}

export function classifyIdea(statement: string): IdeaKind {
  const lower = statement.toLocaleLowerCase().trim();
  if (statement.endsWith("?") || /^(who|what|when|where|why|how|can|could|do|does|is|are)\b/.test(lower)) {
    return "question";
  }
  if (/\b(i (?:like|love|prefer|hate|dislike|want)|my favorite)\b/.test(lower)) {
    return "preference";
  }
  if (/\b(i am|i'm|my name|i live|i work|i study|we are|we're)\b/.test(lower)) {
    return "knowledge";
  }
  return "experience";
}

export function preview(statement: string | undefined, fallback: string, length = 120): string {
  const source = (statement || fallback).replace(/\s+/g, " ").trim();
  return source.length > length ? `${source.slice(0, length - 1)}…` : source;
}

export function isGreeting(input: string): boolean {
  return /^(hey|hello|hi|hiya|yo|good (morning|afternoon|evening))[\s!.,?]*$/i.test(input.trim());
}

export function isQuestion(input: string): boolean {
  const value = input.trim().toLocaleLowerCase();
  return (
    value.endsWith("?") ||
    /^(who|what|when|where|why|how|can|could|do|does|did|is|are|was|were|will|would|should)\b/.test(value)
  );
}
