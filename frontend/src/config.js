export const ALLOWED_MIME_TYPES = [
  "image/jpeg",
  "image/png",
  "image/webp",
  "image/bmp",
  "image/tiff",
];

export const MAX_FILE_SIZE_BYTES = 15 * 1024 * 1024; // 15 MB, must match backend

export const QUALITY_LABEL_CLASS = {
  ACCEPTABLE: "acceptable",
  DEGRADED: "degraded",
  DEFECTIVE: "defective",
};