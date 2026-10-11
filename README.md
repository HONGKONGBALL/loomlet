# Loomlet

Loomlet is a textile bench that runs in the browser. Open a fictional fabric lot, or upload a PNG, JPEG, or WebP photo. A small convolutional network reads a 32×32 view of it, names which of six invented weave families it belongs to, and shows every probability. An occlusion grid covers blocks of the photo and paints where that score drops. k-means reads five dyes off the pixels. A decoder turns the network's feature map into a motif, colors it with those dyes, and tiles a seamless repeat you can download.

Photos are not uploaded. After the page and its weight file have loaded, reading a scrap does not call a server.

The twelve lots (Pier, Fen, Kiln, Lamp, Quay, Loom) are synthetic sample data from `train/train_loomlet.py`. Every card says "Fictional sample". The holdout number on the About view is from that same generator. It is not a measurement of real cloth, and Loomlet is not a commercial fabric identifier.

Loomlet 是一个完全在浏览器里运行的布料样本工作台。它用一个小型卷积网络判断照片更接近哪一种虚构纹样，标出影响判断的区域，并按照片的颜色织出可平铺的重复纹。照片不会上传。示例图片都是虚构数据。

## Try it

```bash
npm install
npm run build
npm start
```

Open `http://127.0.0.1:4178`. No environment variables and no API key.

1. On the bench, choose a lot such as HS-014 and press **Read this scrap**.
2. Press **Show what mattered**, then **Weave repeat**, then **Download repeat PNG**.
3. Open **Compare** and pair two lots.
4. Upload your own PNG, JPEG, or WebP if you want. A flat image is marked unreliable.

Refreshing the bench restores the last fictional lot and runs the network again. An upload clears that memory. Uploaded bytes are not stored.

## Model

Training is local and optional. The checked-in weights already run.

```bash
python3 -m venv --system-site-packages .venv
.venv/bin/pip install torch --index-url https://download.pytorch.org/whl/cpu
.venv/bin/python train/train_loomlet.py
```

The runtime copy of the network is `src/lib/infer.js`. It uses the same layout as the PyTorch module: a classifier (three convolutions, a 48-number embedding, six logits) and a separate motif segmenter (a 32×32 skip joined to an upsampled 16×16 map). The two networks do not share weights. Compare uses the cosine of the six family probabilities. `node tests/parity.mjs` checks that JavaScript matches the exported reference on lot HS-014.

## Tests

Acceptance checks are listed for the hackathon notes and implemented in `tests/ac.spec.js`. They use system Chrome through Playwright.

```bash
npm test
```

## Fonts

Fraunces and Outfit are bundled in `public/fonts/` under the SIL Open Font License, Version 1.1. See `public/fonts/OFL.txt`.

## License

MIT License. Copyright (c) 2026 Zhenhua Liu.

Solo project by Zhenhua (GitHub HONGKONGBALL) for Banana Hacks 2026.
