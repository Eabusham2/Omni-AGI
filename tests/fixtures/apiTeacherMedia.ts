import { mkdir } from "node:fs/promises";
import { ApiTeacherStore, apiTeacherCanonical, apiTeacherSha256, type ApiTeacherStoredJob, type ApiTeacherStoredItem } from "../../src/main/apiTeacherStore";
import { storeApiTeacherMediaLesson, storeApiTeacherResponseEnvelope } from "../../src/main/apiTeacherMedia";

/** Source custody only: a paused received response, no native/model/learning receipt. */
export async function pausedTeacherMediaFixture(root: string, owner: string, suffix = "fixture") {
  await mkdir(root, { recursive: true });
  const image = Buffer.from([137,80,78,71,13,10,26,10,0,0,0,0]), prompt = `Show a ${suffix} bird.`;
  const id = `media-${suffix}`, deliveryId = `delivery-${suffix}`, model = "fixture-media-model";
  const responseEnvelope = await storeApiTeacherResponseEnvelope(root, new Response(JSON.stringify({ steps: [{ type: "model_output", content: [
    { type: "image", mime_type: "image/png", data: image.toString("base64") }
  ] }] })), [], {});
  const mediaLesson = await storeApiTeacherMediaLesson(root, { provider: "gemini", sourceEndpoint: "https://generativelanguage.googleapis.com/v1beta/interactions",
    model, deliveryId, prompt, responseModalities: ["image"], parts: [{ type: "image", mimeType: "image/png", data: image.toString("base64") }] });
  const now = new Date().toISOString();
  const job: ApiTeacherStoredJob = { format: "omni-api-teacher-job", version: 1, id, brainId: owner, originalBrainId: owner,
    originalJobId: id, requestId: `request-${suffix}`, requestSha256: apiTeacherSha256(`request-${owner}-${suffix}`), provider: "gemini", model,
    maxOutputTokens: 32, responseModalities: ["image"], promptCount: 1, learned: 0, failed: 0, skipped: 0, nextIndex: 0,
    state: "paused", revision: 1, createdAt: now, updatedAt: now };
  const item: ApiTeacherStoredItem = { format: "omni-api-teacher-item", version: 1, jobId: id, index: 0, prompt,
    promptSha256: apiTeacherSha256(prompt), deliveryId, state: "response-committed", attemptId: `attempt-${suffix}`,
    requestSha256: apiTeacherSha256(`wire-${owner}-${suffix}`), rawResponseSha256: responseEnvelope.sha256, responseEnvelope,
    learnedResponseSha256: apiTeacherSha256(apiTeacherCanonical(mediaLesson.parts)), response: "", mediaLesson };
  const store = await ApiTeacherStore.open(root, owner, async () => undefined);
  try { await store!.createJob(job, [item]); } finally { store?.close(); }
  return { image, imageSha256: apiTeacherSha256(image), imagePath: `api-teachers/assets/${apiTeacherSha256(image)}.png`, prompt, mediaLesson, responseEnvelope, job, item };
}
