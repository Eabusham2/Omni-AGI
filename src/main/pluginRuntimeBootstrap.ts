import { createHash } from "node:crypto";
import type { PluginRuntimeLaunch, PluginRuntimeModuleDefinition } from "../shared/pluginRuntime";
const literal = (value: unknown): string => JSON.stringify(value).replaceAll("<", "\\u003c").replaceAll("\u2028", "\\u2028").replaceAll("\u2029", "\\u2029");

/** Only the exact packaged body executes, in the dedicated sandbox renderer. */
export function pluginRuntimeScript(launch: PluginRuntimeLaunch, module: PluginRuntimeModuleDefinition): string {
  return `"use strict";(()=>{
const transport=globalThis.__omniRuntimeTransport,sessionId=${literal(launch.session.sessionId)},nonce=${literal(launch.nonce)},purpose=${literal(launch.session.purpose??"interactive")},parserContext=${literal(launch.session.parser??null)};
if(!transport||typeof transport.request!=="function"||typeof transport.subscribe!=="function")throw new Error("Isolated runtime transport is unavailable");
const active=new Set(),deactivate=new Set(),events=new Map(),tools=new Map(),waiting=new Map();let tail=Promise.resolve(),closed=false,parserHandlers=null,parserCallOrdinal=0;
const uuid=()=>crypto.randomUUID();
const finish=(id,value)=>{const waiter=waiting.get(id);if(!waiter)return;waiting.delete(id);if(value.receipt)waiter.resolve({receipt:value.receipt,state:value.state});else if(value.state==="complete")waiter.resolve(value.value);else waiter.reject(new Error(value.error||"Runtime operation did not complete"));};
const perform=async(operation,args,id)=>{if(closed)throw new Error("Runtime is closed");const request={format:"omni-plugin-runtime-rpc",version:1,sessionId,nonce,id,operation,arguments:args};for(;;){const result=await transport.request(request);if(result.state==="backpressure"){await new Promise(resolve=>setTimeout(resolve,20));continue;}if(result.state==="approval-required")return new Promise((resolve,reject)=>waiting.set(id,{resolve,reject}));if(result.receipt)return{receipt:result.receipt,state:result.state};if(result.state!=="complete")throw new Error(result.error||"Runtime request is uncertain or failed");return result.value;}};
const rpc=(operation,args,id=uuid())=>{if(closed)return Promise.reject(new Error("Runtime is closed"));if(operation==="tool-result")return perform(operation,args,id);const result=tail.then(()=>perform(operation,args,id));tail=result.then(()=>undefined,()=>undefined);return result;};
const report=(phase,error,eventId)=>{if(closed)return;void rpc("module-error",{phase,error:error instanceof Error?error.message:String(error),...(eventId?{eventId}:{})}).catch(()=>undefined);};
const api=Object.freeze({
context:Object.freeze({brainId:${literal(launch.session.brainId)},pluginId:${literal(launch.session.pluginId)},contentSha256:${literal(launch.session.contentSha256)},moduleId:${literal(module.id)},purpose,...(parserContext?{parser:Object.freeze(parserContext)}:{})}),
onActivate(fn){if(closed)throw new Error("Runtime is closed");if(typeof fn!=="function")throw new Error("Expected activation callback");active.add(fn);return()=>active.delete(fn);},
onDeactivate(fn){if(closed)throw new Error("Runtime is closed");if(typeof fn!=="function")throw new Error("Expected deactivation callback");deactivate.add(fn);return()=>deactivate.delete(fn);},
on(topic,fn){if(closed)throw new Error("Runtime is closed");if(typeof topic!=="string"||typeof fn!=="function")throw new Error("Expected event callback");const set=events.get(topic)||new Set();set.add(fn);events.set(topic,set);return()=>set.delete(fn);},
call(toolId,action,args={},requestId){if(purpose==="training-parser"&&requestId!==undefined)throw new Error("Training-parser host call IDs are transaction-owned");const id=purpose==="training-parser"?"parser-call-"+parserCallOrdinal++:(requestId??uuid());return rpc("call",{toolId,action,arguments:args},id);},
state:Object.freeze({get(key){return rpc("state.get",{key});},read(key,offset,length){return rpc("state.read",{key,offset,length});},put(key,value,expectedRevision){return rpc("state.put",{key,value,expectedRevision});}}),
receipts:Object.freeze({read(id,offset,length){return rpc("receipt.read",{id,offset,length});}}),
async registerTool(definition,handlers){if(closed)throw new Error("Runtime is closed");if(!handlers||typeof handlers!=="object"||definition.actions.some(action=>typeof handlers[action.id]!=="function"))throw new Error("Every declared action needs a local handler");const previous=tools.get(definition.id);tools.set(definition.id,handlers);try{await rpc("register-tool",{definition});}catch(error){if(previous)tools.set(definition.id,previous);else tools.delete(definition.id);throw error;}},
async registerParser(handlers){if(closed)throw new Error("Runtime is closed");if(purpose!=="training-parser")throw new Error("Parser registration requires an exact training-parser session");if(parserHandlers)throw new Error("This parser session already registered its handler");if(!handlers||typeof handlers.push!=="function"||typeof handlers.drain!=="function")throw new Error("Parser handlers require push and drain callbacks");parserHandlers=Object.freeze({push:handlers.push,drain:handlers.drain});try{await rpc("parser-ready",{abi:"omni-runtime-record-stream-1",formatId:parserContext.formatId});}catch(error){parserHandlers=null;throw error;}},
ui:Object.freeze({put(contribution){return rpc("ui.put",{contribution});},remove(id){return rpc("ui.remove",{id});}})
});
Object.defineProperty(globalThis,"omniRuntime",{value:api,writable:false,configurable:false});
transport.subscribe(message=>{if(!message||typeof message!=="object")return;
if(message.kind==="rpc-reply"&&message.reply){finish(message.id,message.reply);return;}
if(message.kind==="deactivate"){if(closed)return;closed=true;for(const waiter of waiting.values())waiter.reject(new Error("Runtime stopped"));waiting.clear();const shutdown=message.shutdown,callbacks=[...deactivate];
void Promise.allSettled(callbacks.map(fn=>Promise.resolve().then(()=>fn()))).then(results=>{if(!shutdown||shutdown.format!=="omni-plugin-runtime-shutdown"||shutdown.version!==1||shutdown.sessionId!==sessionId||typeof shutdown.nonce!=="string")return;
const errors=[];results.forEach((result,callback)=>{if(result.status==="rejected")errors.push({callback,message:result.reason instanceof Error?result.reason.message:String(result.reason)});});
return transport.request({format:"omni-plugin-runtime-rpc",version:1,sessionId,nonce,id:message.id,operation:"shutdown-ack",arguments:{shutdownNonce:shutdown.nonce,callbacks:results.length,completed:results.length-errors.length,failed:errors.length,errors}});}).catch(()=>undefined);return;}
if(closed)return;
if(message.kind==="activate"){for(const fn of active)Promise.resolve().then(()=>fn()).catch(error=>report("activate",error,message.id));return;}
if(message.kind==="event"){for(const fn of events.get(message.topic)||[])Promise.resolve().then(()=>fn(message.data,{id:message.id,topic:message.topic})).catch(error=>report("event",error,message.id));return;}
if(message.kind==="invoke-tool"){const handler=tools.get(message.toolId)?.[message.action];Promise.resolve().then(()=>{if(typeof handler!=="function")throw new Error("Current module handler is unavailable");return handler(message.data,{requestId:message.id});}).then(value=>rpc("tool-result",{id:message.id,...(value===undefined?{}:{value})}),error=>rpc("tool-result",{id:message.id,error:error instanceof Error?error.message:String(error)})).catch(()=>undefined);}
if(message.kind==="parser-step"){const step=message.parserStep,handler=step&&parserHandlers?.[step.operation];Promise.resolve().then(()=>{if(typeof handler!=="function")throw new Error("Current runtime parser handler is unavailable");return handler(Object.freeze({...step}),{requestId:message.id});}).then(value=>rpc("parser-result",{id:message.id,value}),error=>rpc("parser-result",{id:message.id,error:error instanceof Error?error.message:String(error)})).catch(()=>undefined);}
});
try{(()=>{${module.script}\n})();void rpc("ready",{ok:true}).catch(()=>undefined);}
catch(error){void rpc("ready",{ok:false,error:error instanceof Error?error.message:String(error)}).catch(()=>undefined);}
})();`;
}

export function pluginRuntimeDocument(launch: PluginRuntimeLaunch, module: PluginRuntimeModuleDefinition): Response {
  const script = pluginRuntimeScript(launch, module); if (/<\/script\b/i.test(script)) throw new Error("Runtime script cannot escape its hashed element.");
  const digest = createHash("sha256").update(script).digest("base64");
  const csp = `default-src 'none'; script-src 'sha256-${digest}'; style-src 'none'; connect-src 'none'; img-src 'none'; media-src 'none'; font-src 'none'; frame-src 'none'; worker-src 'none'; object-src 'none'; form-action 'none'; base-uri 'none'; sandbox allow-scripts`;
  return new Response(`<!doctype html><html><head><meta charset="utf-8"><meta name="referrer" content="no-referrer"><title>Omni isolated feature runtime</title></head><body><script>${script}</script></body></html>`,
    { headers: { "Content-Type": "text/html; charset=utf-8", "Content-Security-Policy": csp, "X-Content-Type-Options": "nosniff", "Referrer-Policy": "no-referrer", "Cache-Control": "no-store",
      "Permissions-Policy": "camera=(), microphone=(), geolocation=(), clipboard-read=(), clipboard-write=(), display-capture=(), usb=(), serial=(), hid=()" } });
}
