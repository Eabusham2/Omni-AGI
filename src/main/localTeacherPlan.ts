import type { LocalTeacherGeneration, LocalTeacherPlan, LocalTeacherSelection } from "../shared/localTeacher";
export type LocalTeacherMeasurements = LocalTeacherPlan["measured"];
export function localTeacherPlan(selection: LocalTeacherSelection, measured: LocalTeacherMeasurements): LocalTeacherPlan {
  const header = selection.model.header, geometry = header.geometry, architecture = header.architecture;
  const value = (suffix: string): number | undefined => { const result = architecture ? geometry[`${architecture}.${suffix}`] : undefined;
    return typeof result === "number" && Number.isSafeInteger(result) && result > 0 ? result : undefined; };
  const layers = value("block_count"), embedding = value("embedding_length"), heads = value("attention.head_count"), kvHeads = value("attention.head_count_kv") ?? heads;
  const keyLength = value("attention.key_length") ?? (embedding && heads && embedding % heads === 0 ? embedding / heads : undefined);
  const valueLength = value("attention.value_length") ?? keyLength, contextLimit = value("context_length") ?? null;
  const kvBytesPerToken = layers && kvHeads && keyLength && valueLength ? 2 * layers * kvHeads * (keyLength + valueLength) : null;
  const blockers: string[] = [], warnings = ["GGUF geometry is declared, header-only and not tensor/payload validation.",
    "The fit is an estimate, not an OS RSS limit. Actual app/native/teacher RSS and physical headroom remain guarded.",
    "Teacher responses are unverified training data, not guaranteed facts or competence.",
    "Default timeout0 imposes no wall-clock deadline; Pause/Stop and resource guards remain available."];
  if (Object.values(measured).some(item => !Number.isSafeInteger(item) || item < 0) || measured.logicalCpuCount < 1) throw new Error("Current local-teacher host measurements are unavailable.");
  if (!selection.runtime.rawOfflineSupported) blockers.push(`The selected ${selection.runner} lacks verified required raw/offline flags: ${selection.runtime.missingFlags.join(", ")}.`);
  if (!kvBytesPerToken || !contextLimit || !embedding) blockers.push("Teacher fit is unknown: required context/KV geometry is absent or unsupported. No guessed model size or implicit Load will be used.");
  // File+KV is a declared lower bound. Remaining scratch is explicitly
  // estimated and measured at runtime, never labelled a verified peak.
  const allowance = Math.min(measured.selectedRamCeilingBytes - measured.appFamilyResidentBytes,
    Math.floor(measured.physicalFreeBytes * .85));
  const base = Math.ceil(selection.model.bytes * 1.25) + 16 * 1024 * 1024, scratchPerToken = embedding ? embedding * 32 : 0;
  const possible = kvBytesPerToken ? Math.floor((allowance - base) / (kvBytesPerToken + scratchPerToken)) : 0;
  if (allowance <= base || possible < 8 || contextLimit !== null && contextLimit < 8) blockers.push("The teacher and current app/native owners do not fit the selected envelope. Unload an existing instance or change the reviewed allocation; no owner was silently evicted.");
  let recommendation: LocalTeacherGeneration | null = null, reservationBytes: number | null = null;
  if (!blockers.length && contextLimit && kvBytesPerToken) {
    const contextTokens = Math.max(2, Math.min(contextLimit, possible, 0x7fffffff));
    const maxOutputTokens = Math.max(1, Math.floor(contextTokens / 4));
    recommendation = { contextTokens, maxOutputTokens, threads: Math.max(1, Math.floor(measured.logicalCpuCount * .75)),
      temperature: .7, seed: 0, timeoutMs: 0, maxOutputBytes: Math.max(65536, Math.min(0x7fffffff, maxOutputTokens * 64)) };
    reservationBytes = estimateLocalTeacherReservation(selection, recommendation);
  }
  return { brainId: selection.brainId, teacherId: selection.id, recommendation, reservationBytes,
    fit: blockers.length ? !kvBytesPerToken || !contextLimit ? "unknown" : "blocked" : "estimated", measured,
    declared: { modelBytes: selection.model.bytes, kvBytesPerToken, contextLimit, tensorPayloadValidated: false }, blockers, warnings };
}
export function estimateLocalTeacherReservation(selection: LocalTeacherSelection, generation: LocalTeacherGeneration): number {
  const fake = localTeacherPlanGeometry(selection);
  if (!fake.kvBytesPerToken || !fake.embedding || !fake.contextLimit) throw new Error("Teacher resource fit is unknown; required declared geometry is absent.");
  if (generation.contextTokens > fake.contextLimit) throw new Error("The requested teacher context exceeds its declared context; no silent RoPE/scaling or truncation was added.");
  const bytes = Math.ceil(selection.model.bytes * 1.25) + generation.contextTokens * (fake.kvBytesPerToken + fake.embedding * 32) + 16 * 1024 * 1024;
  if (!Number.isSafeInteger(bytes) || bytes < 1) throw new Error("Teacher resource accounting exceeds supported exact integers."); return bytes;
}
function localTeacherPlanGeometry(selection: LocalTeacherSelection): { embedding: number | null; kvBytesPerToken: number | null; contextLimit: number | null } {
  const g = selection.model.header.geometry, a = selection.model.header.architecture;
  if (!a) return { embedding: null, kvBytesPerToken: null, contextLimit: null };
  const n = (key: string) => { const value = g[`${a}.${key}`]; return typeof value === "number" && Number.isSafeInteger(value) && value > 0 ? value : null; };
  const embedding = n("embedding_length"), heads = n("attention.head_count"), layers = n("block_count"), kvHeads = n("attention.head_count_kv") ?? heads;
  const key = n("attention.key_length") ?? (embedding && heads && embedding % heads === 0 ? embedding / heads : null), val = n("attention.value_length") ?? key;
  return { embedding, contextLimit: n("context_length"), kvBytesPerToken: layers && kvHeads && key && val ? 2 * layers * kvHeads * (key + val) : null };
}
