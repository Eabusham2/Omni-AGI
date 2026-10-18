import { lstat, mkdir, opendir, readFile } from "node:fs/promises";
import { join } from "node:path";
import type { ImaginationAssemblyManifestSummary, ImaginationAssemblyPlan, ImaginationAssemblyPlanRequest, ImaginationPlanEvent } from "../shared/imaginationAssembly";
import type { ReplaySpoolSeal } from "./imaginationReplaySpool";
import { replayReadBudget } from "./imaginationReplaySpool";
import { createHash } from "node:crypto";
import { writeMutableFileIsolated } from "./mutableFileIsolation";
const hash = (value: unknown) => createHash("sha256").update(JSON.stringify(value)).digest("hex");
export interface ReplayPlanRequestReceipt {
  format: "omni-replay-preparation"; version: 1; brainId: string; originBrainId: string; requestId: string;
  request: ImaginationAssemblyPlanRequest; requestSha256: string; event: ImaginationPlanEvent;
  createdAt: string; updatedAt: string; copiedDormant?: true;
  result?: { plan: ImaginationAssemblyPlan; manifest: ImaginationAssemblyManifestSummary; seal: ReplaySpoolSeal };
}
const requestKey = (brainId: string, requestId: string) => hash({ brainId, requestId });
function checked(value: unknown, owner: string, key: string): ReplayPlanRequestReceipt {
  const envelope = value as { value?: ReplayPlanRequestReceipt; sha256?: string }, row = envelope?.value;
  if (!row || envelope.sha256 !== hash(row) || row.format !== "omni-replay-preparation" || row.version !== 1 || row.brainId !== owner || row.request.brainId !== row.originBrainId || row.request.requestId !== row.requestId || row.requestSha256 !== hash(row.request) || key !== requestKey(row.originBrainId, row.requestId) || row.event.requestId !== row.requestId || row.event.brainId !== owner || !Number.isSafeInteger(row.event.sequence) || row.event.sequence < 1 || !["snapshotting","runtime","sources","spooling","ready","cancelling","cancelled","failed","dormant"].includes(row.event.state)) throw new Error("Replay preparation intent or actual ownership changed.");
  if (row.event.state === "ready" && (!row.result || row.result.plan.id !== hash(row.result.seal) || row.result.manifest.manifestSha256 !== row.result.seal.jsonSha256)) throw new Error("Ready preparation lost its exact immutable plan."); return row;
}
export async function readReplayPlanReceiptFile(root: string, owner: string, key: string): Promise<ReplayPlanRequestReceipt> {
  if (!/^[a-f0-9]{64}$/.test(key)) throw new Error("Invalid preparation receipt identity."); const path = join(root,"imagination-assemblies","requests",`${key}.json`), info = await lstat(path);
  if (!info.isFile() || info.isSymbolicLink()) throw new Error("Preparation receipt requires an owned regular file."); replayReadBudget(info.size); return checked(JSON.parse(await readFile(path,"utf8")),owner,key);
}
export async function readReplayPlanRequest(root: string, owner: string, requestId: string): Promise<ReplayPlanRequestReceipt | undefined> {
  const key = requestKey(owner,requestId); if (!await lstat(join(root,"imagination-assemblies","requests",`${key}.json`)).catch(error=>{if(error.code==="ENOENT")return undefined;throw error;})) return undefined;
  return readReplayPlanReceiptFile(root,owner,key);
}
export async function saveReplayPlanRequest(root: string, value: ReplayPlanRequestReceipt, reserve: (bytes:number)=>Promise<void>): Promise<void> {
  const directory=join(root,"imagination-assemblies","requests"); await mkdir(directory,{recursive:true,mode:0o700}); const info=await lstat(directory); if(!info.isDirectory()||info.isSymbolicLink())throw new Error("Unsafe preparation workspace.");
  const key=requestKey(value.originBrainId,value.requestId), envelope={value,sha256:hash(value)}; checked(envelope,value.brainId,key); const body=Buffer.from(JSON.stringify(envelope)); replayReadBudget(body.byteLength); await reserve(body.byteLength+65536); await writeMutableFileIsolated(join(directory,`${key}.json`),body);
}
export async function* replayPlanReceipts(root:string,owner:string):AsyncGenerator<ReplayPlanRequestReceipt> {
  const directory=await opendir(join(root,"imagination-assemblies","requests")).catch(error=>{if(error.code==="ENOENT")return undefined;throw error;});if(!directory)return;
  for await(const entry of directory){if(!entry.isFile()||entry.isSymbolicLink()||!/^[a-f0-9]{64}\.json$/.test(entry.name))throw new Error("Unsafe preparation history payload.");yield readReplayPlanReceiptFile(root,owner,entry.name.slice(0,-5));}
}
export async function rekeyReplayPlanRequests(root:string,oldOwner:string,newOwner:string):Promise<void>{
  for await(const old of replayPlanReceipts(root,oldOwner)){const next:ReplayPlanRequestReceipt={...old,brainId:newOwner,copiedDormant:true,event:{...old.event,brainId:newOwner,sequence:old.event.sequence+1,state:["ready","cancelled","failed"].includes(old.event.state)?old.event.state:"dormant"}};await saveReplayPlanRequest(root,next,async()=>undefined);}
}
