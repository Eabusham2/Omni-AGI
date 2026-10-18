const record = (value: unknown): value is Record<string, unknown> => typeof value === "object" && value !== null && !Array.isArray(value);

/** Fail closed for an explicitly diagnosed stale copy. This does not infer
 * freshness for unmarked older metadata; normal public exports first obtain
 * the native checkpoint/repack receipt and perform existing file verification. */
export function assertSavedInferenceSidecarReady(state: unknown): void {
  if (!record(state)) throw new Error("Saved neural export metadata is invalid.");
  const runtime = record(state.runtime_card) ? state.runtime_card : undefined;
  const status = state.packed_inference_sidecar ?? runtime?.packedInferenceSidecar;
  if (status === undefined) return;
  if (!record(status) || status.format !== "omni-derived-inference-sidecar-status" || status.formatVersion !== 1 ||
      typeof status.refreshRequired !== "boolean" || status.runtimeConsumesSidecar !== false)
    throw new Error("Saved inference-sidecar status is unverified; explicitly checkpoint/repack before export.");
  if (status.refreshRequired || status.status !== "verified-committed-copy")
    throw new Error("The derived inference sidecar requires refresh; explicitly checkpoint/repack before export.");
  const binding = record(status.binding) ? status.binding : undefined;
  const mutable = record(state.mutable_state) ? state.mutable_state : undefined;
  const substrate = record(state.substrate) && record(state.substrate.persistence) ? state.substrate.persistence : undefined;
  const connected = record(state.connected_graph_states) ? state.connected_graph_states.sha256 : null;
  const packed = record(state.packed_ternary_manifest) ? state.packed_ternary_manifest : undefined;
  if (!binding || binding.mutableGeneration !== mutable?.activeGeneration || binding.substrateGeneration !== substrate?.activeGeneration ||
      binding.connectedGraphs !== connected || status.existingSidecarFullyVerified !== true ||
      !packed || packed.contentSha256 !== status.manifestContentSha256 || packed.parameterChecksum !== status.parameterChecksum)
    throw new Error("The inference copy is not bound to this saved generation; explicitly checkpoint/repack before export.");
}
