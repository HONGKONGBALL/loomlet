// Convolutional forward pass for one 32×32 RGB image.
// Weight layout matches PyTorch: conv (Cout, Cin, KH, KW), dense (Out, In), NCHW.
// The classifier and the motif segmenter do not share weights.
// Classifier: 32×32 → 16×16 → 8×8, then a 48-d embedding and 6 logits.
// Segmenter: 32×32 skip (16 ch) plus 16×16 map (32 ch), nearest ×2, concat, then a 1×1 head.

export function parseModel(json) {
  const f32 = (values) => Float32Array.from(values);
  const layer = (block) => ({
    w: f32(block.w),
    b: f32(block.b),
    shape: block.wShape,
  });
  return {
    classes: json.classes.slice(),
    sameFamilyCosineMin: json.sameFamilyCosineMin,
    compareOn: json.compareOn || "embedding",
    embeddingCenter: json.embeddingCenter ? f32(json.embeddingCenter) : null,
    cls1: layer(json.cls1),
    cls2: layer(json.cls2),
    cls3: layer(json.cls3),
    fc1: layer(json.fc1),
    fc2: layer(json.fc2),
    seg1: layer(json.seg1),
    seg2: layer(json.seg2),
    up: layer(json.up),
    head: layer(json.head),
    classH2Means: json.classH2Means.map((row) => f32(row)),
    seedBasis: json.seedBasis.map((row) => f32(row)),
    classDyes: json.classDyes,
  };
}

function relu(x) {
  const y = new Float32Array(x.length);
  for (let i = 0; i < x.length; i += 1) y[i] = x[i] > 0 ? x[i] : 0;
  return y;
}

function linear(x, layer) {
  const [outN, inN] = layer.shape;
  const y = new Float32Array(outN);
  for (let o = 0; o < outN; o += 1) {
    let sum = layer.b[o];
    const row = o * inN;
    for (let i = 0; i < inN; i += 1) sum += x[i] * layer.w[row + i];
    y[o] = sum;
  }
  return y;
}

function conv2d(x, xShape, layer, stride, pad) {
  const [c, height, width] = xShape;
  const [cout, cin, kh, kw] = layer.shape;
  const hp = height + 2 * pad;
  const wp = width + 2 * pad;
  const xp = new Float32Array(c * hp * wp);
  for (let ic = 0; ic < c; ic += 1) {
    for (let y = 0; y < height; y += 1) {
      const src = (ic * height + y) * width;
      const dst = (ic * hp + y + pad) * wp + pad;
      xp.set(x.subarray(src, src + width), dst);
    }
  }
  const hout = Math.floor((height + 2 * pad - kh) / stride) + 1;
  const wout = Math.floor((width + 2 * pad - kw) / stride) + 1;
  const yout = new Float32Array(cout * hout * wout);
  for (let oc = 0; oc < cout; oc += 1) {
    const wBase = oc * cin * kh * kw;
    for (let oy = 0; oy < hout; oy += 1) {
      const y0 = oy * stride;
      for (let ox = 0; ox < wout; ox += 1) {
        const x0 = ox * stride;
        let sum = layer.b[oc];
        let wi = wBase;
        for (let ic = 0; ic < cin; ic += 1) {
          const cBase = ic * hp * wp;
          for (let ky = 0; ky < kh; ky += 1) {
            const row = cBase + (y0 + ky) * wp + x0;
            for (let kx = 0; kx < kw; kx += 1) sum += xp[row + kx] * layer.w[wi++];
          }
        }
        yout[(oc * hout + oy) * wout + ox] = sum;
      }
    }
  }
  return { data: yout, shape: [cout, hout, wout] };
}

function up2(x, shape) {
  const [c, height, width] = shape;
  const h2 = height * 2;
  const w2 = width * 2;
  const y = new Float32Array(c * h2 * w2);
  for (let ic = 0; ic < c; ic += 1) {
    for (let y0 = 0; y0 < height; y0 += 1) {
      for (let x0 = 0; x0 < width; x0 += 1) {
        const value = x[(ic * height + y0) * width + x0];
        const oy = y0 * 2;
        const ox = x0 * 2;
        y[(ic * h2 + oy) * w2 + ox] = value;
        y[(ic * h2 + oy) * w2 + ox + 1] = value;
        y[(ic * h2 + oy + 1) * w2 + ox] = value;
        y[(ic * h2 + oy + 1) * w2 + ox + 1] = value;
      }
    }
  }
  return { data: y, shape: [c, h2, w2] };
}

function concatChannels(a, aShape, b) {
  const [ca, h, w] = aShape;
  const y = new Float32Array(a.length + b.length);
  y.set(a, 0);
  y.set(b, a.length);
  return { data: y, shape: [ca + b.length / (h * w), h, w] };
}

export function decodeMask(model, h1, h2) {
  const up = up2(h2, [32, 16, 16]);
  const joined = concatChannels(up.data, up.shape, h1);
  let mid = conv2d(joined.data, joined.shape, model.up, 1, 1);
  mid = { data: relu(mid.data), shape: mid.shape };
  const raw = conv2d(mid.data, mid.shape, model.head, 1, 0);
  const mask = new Float32Array(32 * 32);
  for (let i = 0; i < mask.length; i += 1) mask[i] = 1 / (1 + Math.exp(-raw.data[i]));
  return mask;
}

export function softmax(logits) {
  let max = -Infinity;
  for (let i = 0; i < logits.length; i += 1) if (logits[i] > max) max = logits[i];
  const probs = new Float32Array(logits.length);
  let sum = 0;
  for (let i = 0; i < logits.length; i += 1) {
    probs[i] = Math.exp(logits[i] - max);
    sum += probs[i];
  }
  for (let i = 0; i < probs.length; i += 1) probs[i] /= sum;
  return probs;
}

export function forward(model, nchw) {
  let c = conv2d(nchw, [3, 32, 32], model.cls1, 1, 1);
  c = { data: relu(c.data), shape: c.shape };
  c = conv2d(c.data, c.shape, model.cls2, 2, 1);
  c = { data: relu(c.data), shape: c.shape };
  c = conv2d(c.data, c.shape, model.cls3, 2, 1);
  const flat = relu(c.data);
  const emb = relu(linear(flat, model.fc1));
  const logits = linear(emb, model.fc2);
  const probs = softmax(logits);
  let pred = 0;
  for (let i = 1; i < probs.length; i += 1) if (probs[i] > probs[pred]) pred = i;

  const h1t = conv2d(nchw, [3, 32, 32], model.seg1, 1, 1);
  const h1 = relu(h1t.data);
  const h2t = conv2d(h1, h1t.shape, model.seg2, 2, 1);
  const h2 = relu(h2t.data);
  const mask = decodeMask(model, h1, h2);
  return { logits, probs, pred, family: model.classes[pred], embedding: emb, h1, h2, mask };
}

export function mixH2(model, h2, familyIndex, seed, fidelity) {
  const mean = model.classH2Means[familyIndex];
  const amount = fidelity / 100;
  const mixed = new Float32Array(h2.length);
  for (let i = 0; i < mixed.length; i += 1) mixed[i] = (1 - amount) * h2[i] + amount * mean[i];
  const basis = model.seedBasis[Math.abs(seed) % model.seedBasis.length];
  const scale = 0.2 * (((Math.abs(seed) % 5) - 2) / 2);
  for (let ch = 0; ch < 32; ch += 1) {
    const delta = scale * basis[ch];
    const base = ch * 256;
    for (let s = 0; s < 256; s += 1) mixed[base + s] += delta;
  }
  return mixed;
}
