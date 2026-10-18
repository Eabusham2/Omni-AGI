import { useEffect, useRef, useState } from "react";
import {
  PLUGIN_TRAINING_PARSER_CAPABILITY,
  type PluginTrainingFormatSelection,
  type PluginsApi
} from "../../shared/plugins";

export interface DatasetParserCandidate {
  selection: PluginTrainingFormatSelection;
  pluginName: string;
  formatTitle: string;
  description: string;
  extensions: string[];
  mediaTypes: string[];
  parserKind: "built-in" | "wasm" | "runtime";
}

export interface DatasetParserPickerModel {
  selection?: PluginTrainingFormatSelection;
  candidates: DatasetParserCandidate[];
  loading: boolean;
  status: string;
  select(selection?: PluginTrainingFormatSelection): void;
}

function currentApi(): PluginsApi | undefined {
  return typeof window === "undefined" ? undefined : window.omni?.plugins;
}

export function datasetParserCandidateKey(selection: PluginTrainingFormatSelection): string {
  return JSON.stringify([selection.pluginId, selection.contentSha256, selection.formatId]);
}

export function datasetParserSelectionForKey(
  value: string,
  candidates: readonly DatasetParserCandidate[]
): PluginTrainingFormatSelection | undefined {
  if (!value) return undefined;
  return candidates.find(candidate => datasetParserCandidateKey(candidate.selection) === value)?.selection;
}

function matchesSelection(
  left: PluginTrainingFormatSelection | undefined,
  right: PluginTrainingFormatSelection
): boolean {
  return left?.pluginId === right.pluginId && left.contentSha256 === right.contentSha256 && left.formatId === right.formatId;
}

/**
 * Renderer discovery is advisory. Main revalidates the exact package, access,
 * capability and source match before committing a manifest.
 */
export async function listDatasetParserCandidates(
  api: Pick<PluginsApi, "list" | "instance" | "package">,
  brainId: string
): Promise<DatasetParserCandidate[]> {
  const candidates: DatasetParserCandidate[] = [];
  const entries = (await api.list()).filter(entry => entry.enabled);
  for (const entry of entries) {
    const access = await api.instance(brainId, entry.pluginId);
    if (!access?.enabled || access.contentSha256 !== entry.activeContentSha256) continue;
    const packageValue = await api.package(entry.pluginId, entry.activeContentSha256);
    if (packageValue.manifest.id !== entry.pluginId) throw new Error("Plugin parser package ownership changed.");
    for (const format of packageValue.manifest.trainingFormats) {
      if (format.parser === "declaration-only") continue;
      let parserKind: DatasetParserCandidate["parserKind"] = "built-in";
      if (format.parser.kind === "wasm-stream") {
        const parser = format.parser;
        const capability = access.capabilities.find(item => item.toolId === PLUGIN_TRAINING_PARSER_CAPABILITY);
        const module = packageValue.manifest.trainingModules?.find(item => item.id === parser.moduleId);
        if (!capability?.enabled || !capability.actions.includes(format.id) || !module) continue;
        parserKind = "wasm";
      } else if (format.parser.kind === "runtime-stream") {
        const parser = format.parser;
        const capability = access.capabilities.find(item => item.toolId === PLUGIN_TRAINING_PARSER_CAPABILITY);
        const module = packageValue.manifest.runtimeModules?.find(item => item.id === parser.moduleId);
        if (!capability?.enabled || !capability.actions.includes(format.id) || !module) continue;
        parserKind = "runtime";
      }
      candidates.push({
        selection: { pluginId: entry.pluginId, contentSha256: entry.activeContentSha256, formatId: format.id },
        pluginName: entry.name,
        formatTitle: format.title,
        description: format.description,
        extensions: [...format.extensions],
        mediaTypes: [...format.mediaTypes],
        parserKind
      });
    }
  }
  return candidates.sort((left, right) => {
    const a = `${left.pluginName}\0${left.formatTitle}\0${datasetParserCandidateKey(left.selection)}`;
    const b = `${right.pluginName}\0${right.formatTitle}\0${datasetParserCandidateKey(right.selection)}`;
    return a < b ? -1 : a > b ? 1 : 0;
  });
}

export function useDatasetParserPicker(
  brainId: string,
  api: PluginsApi | undefined = currentApi()
): DatasetParserPickerModel {
  const [chosen, setChosen] = useState<{ brainId: string; value: PluginTrainingFormatSelection }>();
  const [candidates, setCandidates] = useState<DatasetParserCandidate[]>([]);
  const [loading, setLoading] = useState(Boolean(api));
  const [status, setStatus] = useState("Automatic uses built-in detection or one uniquely matched reviewed file format.");
  const generation = useRef(0);
  const chosenRef = useRef(chosen);
  chosenRef.current = chosen;
  const selection = chosen?.brainId === brainId ? chosen.value : undefined;
  const select = (value?: PluginTrainingFormatSelection) => {
    const next = value ? { brainId, value: { ...value } } : undefined;
    chosenRef.current = next;
    setChosen(next);
    setStatus(value
      ? "This exact immutable parser will be required for the next file or folder manifest."
      : "Automatic uses built-in detection or one uniquely matched reviewed file format.");
  };

  useEffect(() => {
    const owner = ++generation.current;
    let current = true;
    chosenRef.current = undefined;
    setChosen(undefined);
    setCandidates([]);
    if (!api) { setLoading(false); setStatus("Automatic uses built-in detection or one uniquely matched reviewed file format."); return; }
    const refresh = async () => {
      const request = ++generation.current;
      setLoading(true);
      try {
        const next = await listDatasetParserCandidates(api, brainId);
        if (!current || request !== generation.current) return;
        setCandidates(next);
        const active = chosenRef.current?.brainId === brainId ? chosenRef.current.value : undefined;
        if (active && !next.some(candidate => matchesSelection(active, candidate.selection))) {
          chosenRef.current = undefined;
          setChosen(undefined);
          setStatus("The selected parser is no longer enabled for this instance. Automatic detection was restored.");
        } else if (!active) {
          setStatus(next.length
            ? "Automatic uses built-in detection or one uniquely matched reviewed file format; choose an exact parser only when required."
            : "Automatic uses built-in detection; no reviewed plugin file formats are currently available.");
        }
      } catch (error) {
        if (!current || request !== generation.current) return;
        setCandidates([]);
        chosenRef.current = undefined;
        setChosen(undefined);
        setStatus(error instanceof Error ? error.message : "Plugin parser choices could not be verified.");
      } finally {
        if (current && request === generation.current) setLoading(false);
      }
    };
    void refresh();
    const unsubscribe = api.onEvent(event => {
      if (event.type === "catalog" || event.brainId === brainId) void refresh();
    });
    return () => { current = false; generation.current = Math.max(generation.current + 1, owner + 1); unsubscribe(); };
  }, [api, brainId]);

  return { selection, candidates, loading, status, select };
}

function selectors(candidate: DatasetParserCandidate): string {
  return [
    ...candidate.extensions.map(extension => `*.${extension}`),
    ...candidate.mediaTypes
  ].join(", ") || "declared source match";
}

export function DatasetParserPicker({ model, disabled = false }: {
  model: DatasetParserPickerModel;
  disabled?: boolean;
}) {
  const value = model.selection ? datasetParserCandidateKey(model.selection) : "";
  const selected = model.candidates.find(candidate => matchesSelection(model.selection, candidate.selection));
  return <label className="ingest-policy dataset-parser-picker" style={{ flexWrap: "wrap" }}>
    <span style={{ flex: "1 1 240px", minWidth: 0 }}>
      <strong>Parser for the next file or folder</strong>
      <small id="dataset-parser-status" role="status">{model.status}</small>
      {selected ? <small title={selected.selection.contentSha256}>
        {selected.pluginName} · {selected.formatTitle} · {selectors(selected)} · SHA-256 {selected.selection.contentSha256.slice(0, 12)}…
      </small> : null}
    </span>
    <select aria-label="Parser for the next dataset manifest" aria-describedby="dataset-parser-status"
      value={value} disabled={disabled || model.loading} style={{ flex: "1 1 240px", minWidth: 0, maxWidth: "100%" }}
      onChange={event => model.select(datasetParserSelectionForKey(event.currentTarget.value, model.candidates))}>
      <option value="">Automatic · built-in or reviewed file format</option>
      {model.candidates.map(candidate => <option key={datasetParserCandidateKey(candidate.selection)}
        value={datasetParserCandidateKey(candidate.selection)}>
        {candidate.pluginName} · {candidate.formatTitle} · {selectors(candidate)} · {candidate.parserKind === "wasm" ? "WASM" : candidate.parserKind === "runtime" ? "Runtime JS" : "Studio stream"} · {candidate.selection.contentSha256.slice(0, 12)}…
      </option>)}
    </select>
  </label>;
}
