import { decodeMask, mixH2 } from "./infer.js";
import { assignDyes } from "./vision.js";

function clamp01(value) {
  return Math.max(0, Math.min(1, value));
}

function mix3(a, b, amount) {
  return [
    (1 - amount) * a[0] + amount * b[0],
    (1 - amount) * a[1] + amount * b[1],
    (1 - amount) * a[2] + amount * b[2],
  ];
}

export function renderRepeat({ model, h1, h2, tensor, familyIndex, seed, fidelity }) {
  const safeSeed = Math.abs(Math.trunc(seed));
  const amount = Math.max(0, Math.min(100, fidelity)) / 100;
  const mixed = mixH2(model, h2, familyIndex, safeSeed, fidelity);
  const mask = decodeMask(model, h1, mixed);
  const binary = new Uint8Array(mask.length);
  for (let i = 0; i < mask.length; i += 1) binary[i] = mask[i] >= 0.5 ? 1 : 0;
  const photo = assignDyes(tensor, binary);
  const proto = model.classDyes[familyIndex];
  const off = mix3(photo.off, proto[0], amount);
  const on = mix3(photo.on, proto[1], amount);
  const shiftX = safeSeed % 32;
  const shiftY = Math.floor(safeSeed / 3) % 32;
  const width = 256;
  const height = 256;
  const rgba = new Uint8ClampedArray(width * height * 4);
  for (let y = 0; y < height; y += 1) {
    for (let x = 0; x < width; x += 1) {
      const mx = (Math.floor(x / 2) + shiftX) % 32;
      const my = (Math.floor(y / 2) + shiftY) % 32;
      const color = binary[my * 32 + mx] ? on : off;
      const offset = (y * width + x) * 4;
      rgba[offset] = Math.round(clamp01(color[0]) * 255);
      rgba[offset + 1] = Math.round(clamp01(color[1]) * 255);
      rgba[offset + 2] = Math.round(clamp01(color[2]) * 255);
      rgba[offset + 3] = 255;
    }
  }
  return { rgba, width, height };
}
