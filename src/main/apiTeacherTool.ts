import type { ApiTeacherTrainingRequest } from "../shared/types";
import type { HostActionOriginAuthority } from "../shared/hostActionAuthority";
import { apiTeacherProvider } from "./apiTeacherProviders";
import type { ApiTeacherTrainingService } from "./teacherTraining";

export type ApiTeacherToolService = Pick<ApiTeacherTrainingService,
  "status" | "discoverModels" | "cancelModelDiscovery" | "start" | "list" | "details" |
  "pause" | "resume" | "stop" | "resolveUncertain" | "invalidateToolAuthority">;

type TeacherInput = Omit<ApiTeacherTrainingRequest, "brainId" | "requestId">;
type TeacherCommand =
  | { action: "providers" | "list" }
  | { action: "discover-models"; provider: ApiTeacherTrainingRequest["provider"] }
  | { action: "start"; input: TeacherInput }
  | { action: "status" | "pause" | "resume" | "stop"; jobId: string }
  | { action: "resolve-uncertain"; jobId: string; itemIndex: number; decision: "retry" | "skip" };

function exactFields(args: Record<string, unknown>, fields: readonly string[]): void {
  if (!args || typeof args !== "object" || Array.isArray(args) ||
      Object.keys(args).some(key => !fields.includes(key)))
    throw new Error("API teacher tool input contains unsupported fields; credentials and host authority are never tool arguments.");
}
function jobId(value: unknown): string {
  if (typeof value !== "string" || !/^[a-zA-Z0-9][a-zA-Z0-9._-]{0,127}$/.test(value))
    throw new Error("Choose the exact API teacher job ID.");
  return value;
}

/** Validate before a tool intent/audit can retain an unknown credential field. */
export function checkedApiTeacherToolCommand(action: string, args: Record<string, unknown>): TeacherCommand {
  if (action === "providers" || action === "list") { exactFields(args, []); return { action }; }
  if (action === "discover-models") {
    exactFields(args, ["provider"]); return { action, provider: apiTeacherProvider(args.provider) };
  }
  if (action === "start") {
    exactFields(args, ["provider", "model", "prompts", "maxOutputTokens", "responseModalities"]);
    const provider = apiTeacherProvider(args.provider);
    if (typeof args.model !== "string" || !/^[a-zA-Z0-9][a-zA-Z0-9._:/-]{0,199}$/.test(args.model) ||
        !Array.isArray(args.prompts) || !args.prompts.length || args.prompts.some(value => typeof value !== "string") ||
        args.maxOutputTokens !== undefined && (!Number.isSafeInteger(args.maxOutputTokens) || Number(args.maxOutputTokens) < 1 || Number(args.maxOutputTokens) > 65_536) ||
        args.responseModalities !== undefined && (!Array.isArray(args.responseModalities) || !args.responseModalities.length ||
          args.responseModalities.some(value => typeof value !== "string" || !["text", "image", "audio", "video"].includes(value))))
      throw new Error("Select a provider/model and valid explicit teacher lessons and output modalities.");
    return { action, input: { provider, model: args.model, prompts: [...args.prompts] as string[],
      ...(args.maxOutputTokens === undefined ? {} : { maxOutputTokens: args.maxOutputTokens as number }),
      ...(args.responseModalities === undefined ? {} : { responseModalities: [...args.responseModalities] as TeacherInput["responseModalities"] }) } };
  }
  if (["status", "pause", "resume", "stop"].includes(action)) {
    exactFields(args, ["jobId"]); return { action: action as "status" | "pause" | "resume" | "stop", jobId: jobId(args.jobId) };
  }
  if (action === "resolve-uncertain") {
    exactFields(args, ["jobId", "itemIndex", "decision"]);
    if (!Number.isSafeInteger(args.itemIndex) || Number(args.itemIndex) < 0 || typeof args.decision !== "string" || !["retry", "skip"].includes(args.decision))
      throw new Error("Review the exact uncertain teacher item and choose Retry or Skip.");
    return { action, jobId: jobId(args.jobId), itemIndex: args.itemIndex as number, decision: args.decision as "retry" | "skip" };
  }
  throw new Error("Unknown API teacher operation.");
}

/** Same service as the panel; no credential setter, provider fallback or alternate learner. */
export async function executeApiTeacherTool(service: ApiTeacherToolService, brainId: string,
  command: TeacherCommand, requestId: string, signal: AbortSignal, authority: HostActionOriginAuthority): Promise<unknown> {
  signal.throwIfAborted(); await authority.assertCurrent(); signal.throwIfAborted();
  const read = async <T>(work: () => Promise<T>): Promise<T> => {
    const value = await work(); await authority.assertCurrent(); signal.throwIfAborted(); return value;
  };
  if (command.action === "providers") return read(async () => ({ providers: (await service.status()).map(({ keyHint: _hint, ...value }) => value) }));
  if (command.action === "list") return read(async () => ({ jobs: await service.list(brainId) }));
  if (command.action === "discover-models") {
    const request = { provider: command.provider, requestId };
    const cancel = (): void => { service.cancelModelDiscovery(request); };
    signal.addEventListener("abort", cancel, { once: true });
    try { signal.throwIfAborted(); return await read(() => service.discoverModels(request)); }
    finally { signal.removeEventListener("abort", cancel); }
  }
  if (command.action === "start") return service.start({ ...command.input, brainId, requestId }, authority);
  if (command.action === "status") return read(() => service.details(brainId, command.jobId));
  if (command.action === "pause") return service.pause(brainId, command.jobId);
  if (command.action === "resume") return service.resume(brainId, command.jobId, authority);
  if (command.action === "stop") return service.stop(brainId, command.jobId);
  if (command.action === "resolve-uncertain")
    return service.resolveUncertain(brainId, command.jobId, command.itemIndex, command.decision);
  throw new Error("Unknown API teacher operation.");
}
