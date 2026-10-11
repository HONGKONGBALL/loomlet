import { forward } from "./infer.js";

function luma(rgb) {
  return 0.299 * rgb[0] + 0.587 * rgb[1] + 0.114 * rgb[2];
}

function channelByte(value) {
  return Math.max(0, Math.min(255, Math.round(value * 255)))
    .toString(16)
    .padStart(2, "0");
}

export function toHex(rgb) {
  return `#${channelByte(rgb[0])}${channelByte(rgb[1])}${channelByte(rgb[2])}`;
}

export function textureStd(nchw) {
  const count = 1024;
  let worst = 0;
  for (let c = 0; c < 3; c += 1) {
    let sum = 0;
    let sum2 = 0;
    const offset = c * count;
    for (let i = 0; i < count; i += 1) {
      const value = nchw[offset + i];
      sum += value;
      sum2 += value * value;
    }
    const mean = sum / count;
    const variance = Math.max(0, sum2 / count - mean * mean);
    worst = Math.max(worst, Math.sqrt(variance));
  }
  return worst;
}

export function kmeans(nchw, k) {
  const count = 1024;
  const points = new Array(count);
  for (let i = 0; i < count; i += 1) {
    points[i] = [nchw[i], nchw[1024 + i], nchw[2048 + i]];
  }
  const order = points
    .map((rgb, index) => [luma(rgb), index])
    .sort((a, b) => a[0] - b[0]);
  const centers = [];
  for (let c = 0; c < k; c += 1) {
    const index = order[Math.floor(((order.length - 1) * c) / Math.max(1, k - 1))][1];
    centers.push(points[index].slice());
  }
  const assign = new Int16Array(count);
  for (let iter = 0; iter < 10; iter += 1) {
    for (let i = 0; i < count; i += 1) {
      let best = 0;
      let bestDistance = Infinity;
      const point = points[i];
      for (let c = 0; c < k; c += 1) {
        const distance =
          (point[0] - centers[c][0]) ** 2 +
          (point[1] - centers[c][1]) ** 2 +
          (point[2] - centers[c][2]) ** 2;
        if (distance < bestDistance) {
          bestDistance = distance;
          best = c;
        }
      }
      assign[i] = best;
    }
    const sum = Array.from({ length: k }, () => [0, 0, 0, 0]);
    for (let i = 0; i < count; i += 1) {
      const c = assign[i];
      sum[c][0] += points[i][0];
      sum[c][1] += points[i][1];
      sum[c][2] += points[i][2];
      sum[c][3] += 1;
    }
    for (let c = 0; c < k; c += 1) {
      if (sum[c][3] === 0) centers[c] = points[(c * 97) % count].slice();
      else {
        centers[c] = [
          sum[c][0] / sum[c][3],
          sum[c][1] / sum[c][3],
          sum[c][2] / sum[c][3],
        ];
      }
    }
  }
  const counts = Array(k).fill(0);
  for (let i = 0; i < count; i += 1) counts[assign[i]] += 1;
  return centers
    .map((rgb, index) => ({ rgb, count: counts[index] }))
    .sort((a, b) => b.count - a.count || luma(a.rgb) - luma(b.rgb))
    .map((row) => row.rgb);
}

export function paletteHex(nchw) {
  return kmeans(nchw, 5).map(toHex);
}

export function assignDyes(nchw, binary) {
  const colors = kmeans(nchw, 2);
  let best = null;
  for (const pair of [
    [colors[0], colors[1]],
    [colors[1], colors[0]],
  ]) {
    let error = 0;
    for (let i = 0; i < 1024; i += 1) {
      const color = binary[i] ? pair[1] : pair[0];
      error += (color[0] - nchw[i]) ** 2;
      error += (color[1] - nchw[1024 + i]) ** 2;
      error += (color[2] - nchw[2048 + i]) ** 2;
    }
    if (!best || error < best.error) best = { off: pair[0], on: pair[1], error };
  }
  return { off: best.off.slice(), on: best.on.slice() };
}

export function cosine(a, b) {
  let dot = 0;
  let na = 0;
  let nb = 0;
  for (let i = 0; i < a.length; i += 1) {
    dot += a[i] * b[i];
    na += a[i] * a[i];
    nb += b[i] * b[i];
  }
  const denom = Math.sqrt(na) * Math.sqrt(nb);
  if (denom === 0) return 0;
  return dot / denom;
}

export async function occlusionMap(model, nchw, pred, grid = 8) {
  const base = forward(model, nchw).probs[pred];
  const heat = new Float32Array(grid * grid);
  const cell = 32 / grid;
  for (let cy = 0; cy < grid; cy += 1) {
    for (let cx = 0; cx < grid; cx += 1) {
      const copy = new Float32Array(nchw);
      const y0 = Math.floor(cy * cell);
      const y1 = Math.min(32, Math.floor((cy + 1) * cell));
      const x0 = Math.floor(cx * cell);
      const x1 = Math.min(32, Math.floor((cx + 1) * cell));
      for (let c = 0; c < 3; c += 1) {
        for (let y = y0; y < y1; y += 1) {
          for (let x = x0; x < x1; x += 1) copy[(c * 32 + y) * 32 + x] = 0.5;
        }
      }
      heat[cy * grid + cx] = base - forward(model, copy).probs[pred];
    }
    if (cy % 2 === 1) await new Promise((resolve) => setTimeout(resolve, 0));
  }
  return { heat, grid, base };
}

export function paintHeat(canvas, heat, grid) {
  const scale = 32;
  canvas.width = grid * scale;
  canvas.height = grid * scale;
  const ctx = canvas.getContext("2d", { willReadFrequently: true });
  let min = Infinity;
  let max = -Infinity;
  for (let i = 0; i < heat.length; i += 1) {
    min = Math.min(min, heat[i]);
    max = Math.max(max, heat[i]);
  }
  const span = Math.max(1e-6, max - min);
  for (let i = 0; i < heat.length; i += 1) {
    const t = (heat[i] - min) / span;
    const x = (i % grid) * scale;
    const y = Math.floor(i / grid) * scale;
    const r = Math.round(239 + (163 - 239) * t);
    const g = Math.round(231 + (59 - 231) * t);
    const b = Math.round(218 + (43 - 218) * t);
    ctx.fillStyle = `rgb(${r}, ${g}, ${b})`;
    ctx.fillRect(x, y, scale, scale);
  }
}
