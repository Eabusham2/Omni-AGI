import type { ApiTeacherProvider } from "./types";

export type ApiTeacherLessonModality = "text" | "image" | "audio" | "video";
export type ApiTeacherMediaKind = Exclude<ApiTeacherLessonModality, "text">;
export interface ApiTeacherResponseEnvelope {
  format: "omni-api-teacher-response-envelope"; version: 1;
  relativePath: string; sha256: string; bytes: number;
}
export type ApiTeacherLessonPart = { ordinal: number; type: "text"; text: string; sha256: string } | {
  ordinal: number; type: ApiTeacherMediaKind; mimeType: string; relativePath: string; bytes: number; sha256: string;
};
/** Durable content, never a provider credential, request lease or copied execution grant. */
export interface ApiTeacherMediaLesson {
  format: "omni-api-teacher-media-lesson";
  version: 1;
  provider: ApiTeacherProvider;
  sourceEndpoint: string;
  model: string;
  deliveryId: string;
  responseModalities: ApiTeacherLessonModality[];
  parts: ApiTeacherLessonPart[];
  manifestPath: string;
  manifestSha256: string;
}
export interface ApiTeacherMediaCoverage {
  complete: true;
  manifestSha256: string;
  byModality: Partial<Record<ApiTeacherMediaKind, {
    records: number; windows: number; processedUnits: number; complete: true;
  }>>;
}

export function apiTeacherLessonModalities(value: unknown): ApiTeacherLessonModality[] {
  if (!Array.isArray(value) || !value.length || value.some(item => !["text", "image", "audio", "video"].includes(item)) ||
      new Set(value).size !== value.length) throw new Error("Choose distinct, typed API teacher output modalities.");
  return [...value] as ApiTeacherLessonModality[];
}
