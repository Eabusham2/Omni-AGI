import { createHash } from "node:crypto";
import { deflateSync } from "node:zlib";
import { mkdtemp, mkdir, readFile, writeFile, rm } from "node:fs/promises";
import { join } from "node:path";
import { tmpdir } from "node:os";
import { afterEach, expect, it, vi } from "vitest";
import { ImaginationCaptureService, type DeliveredImaginationFragment } from "../src/main/imaginationCapture";
import { ImaginationAssemblyService, type ImaginationAssemblyDependencies } from "../src/main/imaginationAssembly";
import { MediaArtifactRegistry } from "../src/main/mediaArtifactRegistry";
import { createReplayRunner, type ReplayRunner } from "../src/main/imaginationReplayCodec";
import { snapshotLearningWorkspace, adoptLearningWorkspace } from "../src/main/learningWorkspacePortability";
import { readReplay, validateReplayWorkspace } from "../src/main/imaginationAssemblyWorkspace";
import { readReplayPlanRequest } from "../src/main/imaginationPlanRequests";
import type { ImaginationAssembly, ImaginationAssemblyPlanRequest, ImaginationPlanEvent } from "../src/shared/imaginationAssembly";
const executable = "/Users/eyadabushama/Library/Python/3.9/lib/python/site-packages/imageio_ffmpeg/binaries/ffmpeg-macos-aarch64-v7.1";
const hash=(data:Buffer)=>createHash("sha256").update(data).digest("hex");
function png(r:number,g:number,b:number):Buffer{const crc=(data:Buffer)=>{let value=0xffffffff;for(const byte of data){value^=byte;for(let n=0;n<8;n++)value=value&1?(value>>>1)^0xedb88320:value>>>1;}return(value^0xffffffff)>>>0;};const chunk=(name:string,data:Buffer)=>{const type=Buffer.from(name),size=Buffer.alloc(4),sum=Buffer.alloc(4);size.writeUInt32BE(data.length);sum.writeUInt32BE(crc(Buffer.concat([type,data])));return Buffer.concat([size,type,data,sum]);};const header=Buffer.alloc(13);header.writeUInt32BE(32,0);header.writeUInt32BE(16,4);header[8]=8;header[9]=2;const pixels=Buffer.alloc((32*3+1)*16);for(let y=0;y<16;y++)for(let x=0;x<32;x++){const p=y*(32*3+1)+1+x*3;pixels[p]=r;pixels[p+1]=g;pixels[p+2]=b;}return Buffer.concat([Buffer.from([137,80,78,71,13,10,26,10]),chunk("IHDR",header),chunk("IDAT",deflateSync(pixels)),chunk("IEND",Buffer.alloc(0))]);}
function wav(samples:number):Buffer{const data=Buffer.alloc(44+samples*2);data.write("RIFF");data.writeUInt32LE(data.length-8,4);data.write("WAVEfmt ",8);data.writeUInt32LE(16,16);data.writeUInt16LE(1,20);data.writeUInt16LE(1,22);data.writeUInt32LE(8000,24);data.writeUInt32LE(16000,28);data.writeUInt16LE(2,32);data.writeUInt16LE(16,34);data.write("data",36);data.writeUInt32LE(samples*2,40);for(let i=0;i<samples;i++)data.writeInt16LE(i<80?3277:19660,44+i*2);return data;}
function dataChunk(bytes:Buffer):Buffer{let p=12;while(p+8<=bytes.length){const n=bytes.readUInt32LE(p+4);if(bytes.toString("ascii",p,p+4)==="data")return bytes.subarray(p+8,p+8+n);p+=8+n+(n%2);}throw new Error("Missing WAV data");}
const cleanups:Array<()=>Promise<void>>=[];
afterEach(async()=>{for(const close of cleanups.splice(0).reverse())await close();});
async function owner(root?:string,id="brain",runFactory?:(deps:Parameters<typeof createReplayRunner>[0])=>Promise<ReplayRunner>){
  const path=root??await mkdtemp(join(tmpdir(),"omni-replay-real-"));if(!root)cleanups.push(()=>rm(path,{recursive:true,force:true}));await mkdir(join(path,"engine"),{recursive:true});await writeFile(join(path,"brain.json"),JSON.stringify({id}));let now=0;
  const registry=new MediaArtifactRegistry(()=>path),capture=new ImaginationCaptureService({brainDirectory:()=>path,now:()=>now,wall:()=>new Date(Date.parse("2026-10-13T12:00:00Z")+now).toISOString(),reserve:async()=>undefined,admit:()=>undefined,mediaUrl:(brain,sha,mime)=>registry.leaseCapture(brain,sha,mime)});
  const prepare=vi.fn<ImaginationAssemblyDependencies["prepareRuntime"]>(async()=>({state:"external",executablePath:executable}));
  const assembly=new ImaginationAssemblyService({brainDirectory:()=>path,snapshotDatabase:(brain,recording,revision,destination)=>capture.snapshotTimelineDatabase(brain,recording,revision,destination),prepareRuntime:prepare,checkResources:async()=>undefined,reserve:async()=>undefined,mediaUrl:(brain,sha,mime)=>registry.leaseReplay(brain,sha,mime),presentationRefreshHz:()=>120,...(runFactory?{runner:runFactory}:{})});
  cleanups.push(async()=>{await assembly.dispose();await capture.dispose();registry.dispose();});
  async function deliver(index:number,bytes:Buffer,modality:"image"|"audio"="image"){const file=join(path,`fixture-${modality}-${index}.data`);await writeFile(file,bytes);const source:DeliveredImaginationFragment={brainId:id,streamId:`job:${id}:owned-${modality}`,origin:{kind:"spontaneous",authority:"main-organic-action",actionEventId:"organic"},kind:"preview-fragment",modality,revision:index,width:modality==="image"?32:undefined,height:modality==="image"?16:undefined,
    mediaTimeline:{format:"omni-native-media-timeline",version:1,producer:"native-byte-encoder",layout:modality==="image"?"snapshot":"replacement-from-zero",unit:modality==="image"?"image":"samples",start:0,count:modality==="image"?1:(bytes.length-44)/2,timebaseNumerator:1,timebaseDenominator:modality==="image"?1:8000},resolve:async()=>({path:file,sha256:hash(bytes),mimeType:modality==="image"?"image/png":"audio/wav",size:bytes.length})};await capture.delivered(source);return source;}
  async function waitComplete(jobId:string):Promise<ImaginationAssembly>{return new Promise((resolve,reject)=>{let unsub=()=>{};const read=async()=>{try{const view=await assembly.get(id,jobId);if(view.state==="complete"){unsub();resolve(view);}else if(["failed","cancelled"].includes(view.state)){unsub();reject(new Error(view.error??view.state));}}catch(error){unsub();reject(error);}};unsub=assembly.subscribe(event=>{if(event.assemblyId===jobId)void read();});void read();});}
  return{path,id,capture,assembly,registry,prepare,deliver,waitComplete,now:(value:number)=>{now=value;}};
}
it("encodes actual progressive image revisions at delivery offsets into a playable checked MP4",async()=>{
  const value=await owner(),record=await value.capture.record({brainId:value.id,requestId:"image-record",lookbackSeconds:0});const first=await value.deliver(0,png(255,0,0));value.now(100);await value.deliver(1,png(0,0,255));value.now(200);const stopped=await value.capture.stop({brainId:value.id,recordingId:record.id,expectedRevision:(await value.capture.status(value.id)).recording!.revision});
  const request:ImaginationAssemblyPlanRequest={brainId:value.id,recordingId:record.id,expectedRevision:stopped.revision,requestId:"image-plan"},plan=await value.assembly.plan(request);expect(plan.options).toEqual({width:32,height:16,fps:10});expect(plan.outputKind).toBe("audiovisual");
  const [a,b]=await Promise.all([value.assembly.assemble({...request,requestId:"image-assemble",planId:plan.id}),value.assembly.assemble({...request,requestId:"image-assemble",planId:plan.id})]);expect(a.id).toBe(b.id);const done=await value.waitComplete(a.id);expect(done).toMatchObject({mimeType:"video/mp4",available:true,completedSpans:2});
  const decoded=await createReplayRunner({runtime:{state:"external",executablePath:executable},estimatedPeakBytes:65536,checkResources:async()=>undefined,reserve:async()=>undefined,timeoutMs:10000});const frames=await decoded(["-loglevel","error","-f","mov","-i",join(value.path,"imagination-assemblies","blobs",`${done.artifactSha256}.data`),"-map","0:v:0","-f","framemd5","-"],value.path,new AbortController().signal);const checksums=[...frames.stdout.matchAll(/^0,.*?,\s*([a-f0-9]{32})\s*$/gm)].map(row=>row[1]);expect(checksums).toHaveLength(2);expect(checksums[0]).not.toBe(checksums[1]);expect((await first.resolve()).sha256).toBe(hash(png(255,0,0)));await validateReplayWorkspace(value.path,value.id);
},30000);
it("reports and stops the exact shared runtime preparation, retaining its cancelled receipt without restarting on retry",async()=>{
  const value=await owner(),record=await value.capture.record({brainId:value.id,requestId:"cancel-record",lookbackSeconds:0});
  await value.deliver(0,png(255,0,0)); value.now(100);
  const stopped=await value.capture.stop({brainId:value.id,recordingId:record.id,expectedRevision:(await value.capture.status(value.id)).recording!.revision});
  let entered=()=>{};const atRuntime=new Promise<void>(resolve=>{entered=resolve;});
  value.prepare.mockImplementationOnce(async(signal,progress)=>{
    progress({state:"downloading",target:"owned-fixture",message:"Admitted byte transfer",completedBytes:7,totalBytes:19});entered();
    await new Promise<void>((_resolve,reject)=>{if(signal.aborted)reject(signal.reason);else signal.addEventListener("abort",()=>reject(signal.reason),{once:true});});
    throw new Error("The cancelled preparation must never return an executable.");
  });
  const events:ImaginationPlanEvent[]=[],unsubscribe=value.assembly.subscribePlans(event=>events.push(event));
  const request:ImaginationAssemblyPlanRequest={brainId:value.id,recordingId:record.id,expectedRevision:stopped.revision,requestId:"cancel-plan"};
  const first=value.assembly.plan(request),retry=value.assembly.plan(request),settled=Promise.allSettled([first,retry]);
  await atRuntime;
  expect(value.prepare).toHaveBeenCalledTimes(1);
  expect(events).toContainEqual(expect.objectContaining({requestId:request.requestId,state:"runtime",runtime:expect.objectContaining({completedBytes:7,totalBytes:19})}));
  await expect(value.assembly.plan({...request,options:{fps:24}})).rejects.toThrow("different exact inputs");
  expect(await value.assembly.cancelPlan({brainId:value.id,requestId:request.requestId})).toEqual({brainId:value.id,requestId:request.requestId,state:"cancelled"});
  expect((await settled).map(row=>row.status)).toEqual(["rejected","rejected"]);
  expect(events.map(row=>row.sequence)).toEqual([...new Set(events.map(row=>row.sequence))].sort((a,b)=>a-b));
  expect(events.at(-1)).toMatchObject({state:"cancelled"});
  const receipt=await readReplayPlanRequest(value.path,value.id,request.requestId);expect(receipt).toMatchObject({request,event:{state:"cancelled"}});
  await expect(value.assembly.plan(request)).rejects.toThrow("already ended or was interrupted");
  expect(value.prepare).toHaveBeenCalledTimes(1);expect((await value.assembly.list(value.id)).entries).toEqual([]);
  expect((await value.capture.recordings(value.id)).entries.find(row=>row.id===record.id)).toMatchObject({revision:stopped.revision,fragmentCount:1});
  unsubscribe();
},10000);
it("outputs genuine audio-only WAV and advances within replacement prefixes without replaying old samples",async()=>{
  const value=await owner(),record=await value.capture.record({brainId:value.id,requestId:"audio-record",lookbackSeconds:0});await value.deliver(0,wav(80),"audio");value.now(5);await value.deliver(1,wav(160),"audio");value.now(20);const stopped=await value.capture.stop({brainId:value.id,recordingId:record.id,expectedRevision:(await value.capture.status(value.id)).recording!.revision});
  const request={brainId:value.id,recordingId:record.id,expectedRevision:stopped.revision,requestId:"audio-plan"},plan=await value.assembly.plan(request);expect(plan.outputKind).toBe("audio-only");expect(plan.durationMs).toBe(20);const job=await value.assembly.assemble({...request,requestId:"audio-assemble",planId:plan.id}),done=await value.waitComplete(job.id);expect(done.mimeType).toBe("audio/wav");
  const audio=dataChunk(await readFile(join(value.path,"imagination-assemblies","blobs",`${done.artifactSha256}.data`)));const sample=(ms:number)=>Math.abs(audio.readInt16LE(Math.floor(ms*48)*4));expect(sample(12)).toBeGreaterThan(sample(8)*3);expect(audio.length).toBe(20*48*4);
},30000);
it("exports a committed pending codec boundary dormant, then resumes the copy only by explicit control",async()=>{
  let release=()=>{},entered=()=>{};const gate=new Promise<void>(resolve=>{release=resolve;}),atSegment=new Promise<void>(resolve=>{entered=resolve;});let held=false,firstRuns=0;
  const value=await owner(undefined,"brain",async deps=>{const run=await createReplayRunner(deps);return async(args,cwd,signal,progress)=>{if(args.at(-1)?.endsWith("segment-0.mkv")){firstRuns++;if(!held){held=true;entered();await gate;}}return run(args,cwd,signal,progress);};});
  try {
  const record=await value.capture.record({brainId:value.id,requestId:"pending-record",lookbackSeconds:0});for(let i=0;i<3;i++){value.now(i*100);await value.deliver(i,png(i===0?255:0,i===1?255:0,i===2?255:0));}value.now(300);const stopped=await value.capture.stop({brainId:value.id,recordingId:record.id,expectedRevision:(await value.capture.status(value.id)).recording!.revision});
  const request={brainId:value.id,recordingId:record.id,expectedRevision:stopped.revision,requestId:"pending-plan"},plan=await value.assembly.plan(request),job=await value.assembly.assemble({...request,requestId:"pending-assemble",planId:plan.id});await atSegment;
  const packageRoot=await mkdtemp(join(tmpdir(),"omni-replay-export-"));cleanups.push(()=>rm(packageRoot,{recursive:true,force:true}));const snapshot=snapshotLearningWorkspace(value.path,join(packageRoot,"archive"),value.id);
  await vi.waitFor(()=>expect((value.assembly as unknown as{snapshotHolds:Set<string>}).snapshotHolds.has(value.id)).toBe(true));release();const archive=(await snapshot)!;const target=join(packageRoot,"copy");await mkdir(join(target,"engine"),{recursive:true});await writeFile(join(target,"brain.json"),JSON.stringify({id:"copy"}));await adoptLearningWorkspace(archive,target,"copy");
  const copied=await owner(target,"copy"),pending=await copied.assembly.get("copy",job.id);expect(pending).toMatchObject({state:"dormant",completedSpans:1});expect(copied.prepare).not.toHaveBeenCalled();await value.waitComplete(job.id);expect(firstRuns).toBe(1);
  await copied.assembly.resume({brainId:"copy",assemblyId:job.id,expectedRevision:pending.revision});expect((await copied.waitComplete(job.id)).state).toBe("complete");expect((await readReplay(target,job.id,"copy")).intent.manifest.fragmentCount).toBe(3);await validateReplayWorkspace(target,"copy");
  } finally { release(); }
},30000);
