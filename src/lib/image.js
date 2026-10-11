const MAX_BYTES = 8 * 1024 * 1024;

export const STORAGE_KEY = "loomlet.bench.v1";

export async function tensorFromBin(url) {
  const response = await fetch(url);
  if (!response.ok) throw new Error("This file could not be read as an image.");
  const bytes = new Uint8Array(await response.arrayBuffer());
  if (bytes.length !== 3072) throw new Error("This file could not be read as an image.");
  const nchw = new Float32Array(3 * 1024);
  for (let i = 0; i < 1024; i += 1) {
    nchw[i] = bytes[i * 3] / 255;
    nchw[1024 + i] = bytes[i * 3 + 1] / 255;
    nchw[2048 + i] = bytes[i * 3 + 2] / 255;
  }
  return nchw;
}

export function tensorFromImage(image) {
  const canvas = document.createElement("canvas");
  canvas.width = 32;
  canvas.height = 32;
  const ctx = canvas.getContext("2d", { willReadFrequently: true });
  ctx.imageSmoothingEnabled = true;
  ctx.drawImage(image, 0, 0, 32, 32);
  const data = ctx.getImageData(0, 0, 32, 32).data;
  const nchw = new Float32Array(3 * 1024);
  for (let i = 0; i < 1024; i += 1) {
    nchw[i] = data[i * 4] / 255;
    nchw[1024 + i] = data[i * 4 + 1] / 255;
    nchw[2048 + i] = data[i * 4 + 2] / 255;
  }
  return nchw;
}

function typeAllowed(file) {
  const type = (file.type || "").toLowerCase();
  const ext = (file.name || "").split(".").pop().toLowerCase();
  const typeOk = type === "image/png" || type === "image/jpeg" || type === "image/webp";
  const extOk = ext === "png" || ext === "jpg" || ext === "jpeg" || ext === "webp";
  return typeOk || (type === "" && extOk);
}

export function fileToTensor(file) {
  return new Promise((resolve, reject) => {
    if (!file || !typeAllowed(file)) {
      reject(new Error("Use a PNG, JPEG, or WebP image."));
      return;
    }
    if (file.size > MAX_BYTES) {
      reject(new Error("Use an image under 8 MB."));
      return;
    }
    const url = URL.createObjectURL(file);
    const image = new Image();
    image.onload = () => {
      if (!image.naturalWidth || !image.naturalHeight) {
        URL.revokeObjectURL(url);
        reject(new Error("This file could not be read as an image."));
        return;
      }
      try {
        resolve({
          tensor: tensorFromImage(image),
          url,
          name: file.name || "upload",
        });
      } catch {
        URL.revokeObjectURL(url);
        reject(new Error("This file could not be read as an image."));
      }
    };
    image.onerror = () => {
      URL.revokeObjectURL(url);
      reject(new Error("This file could not be read as an image."));
    };
    image.src = url;
  });
}
