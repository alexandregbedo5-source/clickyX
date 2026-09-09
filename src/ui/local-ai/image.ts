import { ACCEPTED_IMAGE_TYPES, MAX_IMAGE_BYTES, THUMBNAIL_SIZE } from "./constants";

export function isAcceptedImage(file: File): boolean {
  if (ACCEPTED_IMAGE_TYPES.includes(file.type)) return true;
  return /\.(png|jpe?g|webp|bmp|gif)$/i.test(file.name);
}

export function validateImageFile(file: File): string | null {
  if (!isAcceptedImage(file)) {
    return "Use a PNG, JPEG, WebP, BMP, or GIF image.";
  }
  if (file.size > MAX_IMAGE_BYTES) {
    return `Image is too large (max ${Math.round(MAX_IMAGE_BYTES / (1024 * 1024))} MB).`;
  }
  return null;
}

export function readFileAsDataUrl(file: File): Promise<string> {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onerror = () => reject(new Error("Could not read this file."));
    reader.onload = () => {
      if (typeof reader.result === "string") resolve(reader.result);
      else reject(new Error("Could not read this file."));
    };
    reader.readAsDataURL(file);
  });
}

export async function makeThumbnail(dataUrl: string, size = THUMBNAIL_SIZE): Promise<string | null> {
  if (typeof document === "undefined") return null;
  return new Promise((resolve) => {
    const img = new Image();
    img.onload = () => {
      try {
        const canvas = document.createElement("canvas");
        const scale = Math.min(size / img.width, size / img.height, 1);
        canvas.width = Math.max(1, Math.round(img.width * scale));
        canvas.height = Math.max(1, Math.round(img.height * scale));
        const ctx = canvas.getContext("2d");
        if (!ctx) {
          resolve(null);
          return;
        }
        ctx.drawImage(img, 0, 0, canvas.width, canvas.height);
        resolve(canvas.toDataURL("image/jpeg", 0.72));
      } catch {
        resolve(null);
      }
    };
    img.onerror = () => resolve(null);
    img.src = dataUrl;
  });
}
