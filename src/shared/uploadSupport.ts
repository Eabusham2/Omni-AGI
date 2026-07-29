export type ExperienceUploadKind = "files" | "images" | "audio" | "video";

export interface ExperienceUploadDescriptor {
  title: string;
  filterName: string;
  extensions: readonly string[];
  shortLabel: string;
}

const DOCUMENT_AND_DATASET_EXTENSIONS = [
  "pdf",
  "epub",
  "docx",
  "odt",
  "pptx",
  "txt",
  "md",
  "mdx",
  "rst",
  "html",
  "htm",
  "json",
  "jsonl",
  "ndjson",
  "csv",
  "tsv",
  "parquet",
  "arrow",
  "feather",
  "ipc",
  "sqlite",
  "sqlite3",
  "db",
  "zip",
  "tar",
  "tgz",
  "gz",
  "bz2",
  "xz",
  "py",
  "js",
  "jsx",
  "ts",
  "tsx",
  "rs",
  "go",
  "java",
  "c",
  "cc",
  "cpp",
  "h",
  "hpp",
  "cs",
  "swift",
  "kt",
  "sql",
  "css",
  "scss",
  "yaml",
  "yml",
  "toml"
] as const;

export const IMAGE_UPLOAD_EXTENSIONS = [
  "png",
  "jpg",
  "jpeg",
  "webp",
  "gif",
  "bmp",
  "tif",
  "tiff",
  "avif",
  "heic",
  "heif"
] as const;

export const AUDIO_UPLOAD_EXTENSIONS = [
  "wav",
  "mp3",
  "flac",
  "m4a",
  "aac",
  "ogg",
  "oga",
  "opus",
  "aiff",
  "aif",
  "wma"
] as const;

export const VIDEO_UPLOAD_EXTENSIONS = [
  "mp4",
  "webm",
  "mov",
  "mkv",
  "avi",
  "m4v",
  "mpeg",
  "mpg",
  "wmv",
  "flv"
] as const;

export const EXPERIENCE_UPLOADS: Record<
  ExperienceUploadKind,
  ExperienceUploadDescriptor
> = {
  files: {
    title: "Choose documents, datasets, code, images, audio, or video to learn",
    filterName: "Supported learning material",
    extensions: [
      ...DOCUMENT_AND_DATASET_EXTENSIONS,
      ...IMAGE_UPLOAD_EXTENSIONS,
      ...AUDIO_UPLOAD_EXTENSIONS,
      ...VIDEO_UPLOAD_EXTENSIONS
    ],
    shortLabel: "files and media"
  },
  images: {
    title: "Choose images to learn",
    filterName: "Images",
    extensions: IMAGE_UPLOAD_EXTENSIONS,
    shortLabel: "images"
  },
  audio: {
    title: "Choose audio to learn",
    filterName: "Audio",
    extensions: AUDIO_UPLOAD_EXTENSIONS,
    shortLabel: "audio"
  },
  video: {
    title: "Choose video to learn",
    filterName: "Video",
    extensions: VIDEO_UPLOAD_EXTENSIONS,
    shortLabel: "video"
  }
};

export function isExperienceUploadKind(
  value: unknown
): value is ExperienceUploadKind {
  return (
    typeof value === "string" &&
    Object.prototype.hasOwnProperty.call(EXPERIENCE_UPLOADS, value)
  );
}
