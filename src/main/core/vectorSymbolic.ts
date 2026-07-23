const DIMENSIONS = 256;
const WORDS = DIMENSIONS / 32;

function fnv1a(input: string): number {
  let hash = 0x811c9dc5;
  for (let index = 0; index < input.length; index += 1) {
    hash ^= input.charCodeAt(index);
    hash = Math.imul(hash, 0x01000193);
  }
  return hash >>> 0;
}

export function seededRandom(seed: number): () => number {
  let state = seed >>> 0 || 0x6d2b79f5;
  return () => {
    state += 0x6d2b79f5;
    let value = state;
    value = Math.imul(value ^ (value >>> 15), value | 1);
    value ^= value + Math.imul(value ^ (value >>> 7), value | 61);
    return ((value ^ (value >>> 14)) >>> 0) / 4_294_967_296;
  };
}

function ngrams(value: string): string[] {
  const normalized = `^${value.toLocaleLowerCase()}$`;
  if (normalized.length <= 3) return [normalized];

  const grams: string[] = [];
  for (let index = 0; index <= normalized.length - 3; index += 1) {
    grams.push(normalized.slice(index, index + 3));
  }
  return grams;
}

function randomHypervector(key: string): Uint32Array {
  const random = seededRandom(fnv1a(key));
  const vector = new Uint32Array(WORDS);
  for (let index = 0; index < vector.length; index += 1) {
    vector[index] = Math.floor(random() * 4_294_967_296) >>> 0;
  }
  return vector;
}

function popcount32(value: number): number {
  let work = value >>> 0;
  work -= (work >>> 1) & 0x55555555;
  work = (work & 0x33333333) + ((work >>> 2) & 0x33333333);
  return (((work + (work >>> 4)) & 0x0f0f0f0f) * 0x01010101) >>> 24;
}

function bitAt(vector: Uint32Array, bit: number): number {
  const word = vector[Math.floor(bit / 32)] ?? 0;
  return (word >>> (bit % 32)) & 1;
}

/**
 * Encodes a concept as a compositional 256-bit hypervector. Character n-gram
 * components make morphologically related labels closer than unrelated labels.
 */
export function encodeConcept(label: string): Uint32Array {
  const components = ngrams(label).map(randomHypervector);
  const output = new Uint32Array(WORDS);

  for (let bit = 0; bit < DIMENSIONS; bit += 1) {
    let vote = 0;
    for (const component of components) {
      vote += bitAt(component, bit) === 1 ? 1 : -1;
    }
    if (vote >= 0) {
      const wordIndex = Math.floor(bit / 32);
      output[wordIndex] = (output[wordIndex] ?? 0) | (1 << (bit % 32));
    }
  }

  return output;
}

export function bundleConcepts(labels: string[]): Uint32Array {
  if (labels.length === 0) return new Uint32Array(WORDS);
  const vectors = labels.map(encodeConcept);
  const output = new Uint32Array(WORDS);

  for (let bit = 0; bit < DIMENSIONS; bit += 1) {
    let vote = 0;
    for (const vector of vectors) {
      vote += bitAt(vector, bit) === 1 ? 1 : -1;
    }
    if (vote >= 0) {
      const wordIndex = Math.floor(bit / 32);
      output[wordIndex] = (output[wordIndex] ?? 0) | (1 << (bit % 32));
    }
  }

  return output;
}

export function similarity(left: Uint32Array, right: Uint32Array): number {
  let distance = 0;
  for (let index = 0; index < WORDS; index += 1) {
    distance += popcount32((left[index] ?? 0) ^ (right[index] ?? 0));
  }
  return 1 - distance / DIMENSIONS;
}

export function encodeFingerprint(labels: string[]): string {
  return Array.from(bundleConcepts(labels))
    .map((word) => word.toString(16).padStart(8, "0"))
    .join("");
}

export function decodeFingerprint(fingerprint: string): Uint32Array {
  if (fingerprint.length !== WORDS * 8) return new Uint32Array(WORDS);
  const output = new Uint32Array(WORDS);
  for (let index = 0; index < WORDS; index += 1) {
    output[index] = Number.parseInt(fingerprint.slice(index * 8, index * 8 + 8), 16) >>> 0;
  }
  return output;
}

export function textSeed(text: string, turn: number): number {
  return (fnv1a(text) ^ Math.imul(turn + 1, 0x9e3779b1)) >>> 0;
}

export const hypervectorDimensions = DIMENSIONS;
