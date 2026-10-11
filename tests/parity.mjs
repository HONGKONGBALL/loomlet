import fs from "node:fs";
import path from "node:path";
import { forward, parseModel } from "../src/lib/infer.js";

const root = path.resolve(import.meta.dirname, "..");
const weights = JSON.parse(fs.readFileSync(path.join(root, "public/model/loomlet-weights.json"), "utf8"));
const reference = JSON.parse(fs.readFileSync(path.join(root, "public/model/reference.json"), "utf8"));
const bin = fs.readFileSync(path.join(root, "public/samples/pixels/HS-014.bin"));
if (bin.length !== 3072) {
  console.error(`HS-014.bin length ${bin.length}`);
  process.exit(1);
}

const nchw = new Float32Array(3 * 1024);
for (let i = 0; i < 1024; i += 1) {
  nchw[i] = bin[i * 3] / 255;
  nchw[1024 + i] = bin[i * 3 + 1] / 255;
  nchw[2048 + i] = bin[i * 3 + 2] / 255;
}

const out = forward(parseModel(weights), nchw);

function maxAbs(left, right) {
  let worst = 0;
  if (left.length !== right.length) return Infinity;
  for (let i = 0; i < left.length; i += 1) worst = Math.max(worst, Math.abs(left[i] - right[i]));
  return worst;
}

const logitErr = maxAbs(out.logits, reference.logits);
const maskErr = maxAbs(out.mask, reference.mask);
const result = {
  id: "AC-21",
  status: logitErr < 1e-3 && maskErr < 1e-3 ? "PASS" : "FAIL",
  logitErr,
  maskErr,
  family: out.family,
};
fs.mkdirSync(path.join(root, "qa-evidence"), { recursive: true });
fs.writeFileSync(path.join(root, "qa-evidence/AC-21.json"), JSON.stringify(result, null, 2));
console.log(JSON.stringify(result));
if (result.status !== "PASS") process.exit(1);
