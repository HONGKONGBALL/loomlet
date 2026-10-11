import { useState } from "react";
import { forward } from "../lib/infer.js";
import { fileToTensor, tensorFromBin } from "../lib/image.js";
import { cosine } from "../lib/vision.js";

function readingVector(model, read) {
  if (model.compareOn === "softmax") return read.probs;
  if (model.compareOn === "logits") {
    const centered = new Float32Array(read.logits.length);
    let mean = 0;
    for (let i = 0; i < read.logits.length; i += 1) mean += read.logits[i];
    mean /= read.logits.length;
    for (let i = 0; i < centered.length; i += 1) centered[i] = read.logits[i] - mean;
    return centered;
  }
  if (model.compareOn === "centered" && model.embeddingCenter) {
    const centered = new Float32Array(read.embedding.length);
    const center = model.embeddingCenter;
    for (let i = 0; i < centered.length; i += 1) centered[i] = read.embedding[i] - center[i];
    return centered;
  }
  return read.embedding;
}

export default function CompareView({ model, samples }) {
  const [left, setLeft] = useState(null);
  const [right, setRight] = useState(null);
  const [hint, setHint] = useState("Choose two scraps.");
  const [verdict, setVerdict] = useState("");
  const [score, setScore] = useState(null);
  const [error, setError] = useState("");

  async function loadId(id) {
    const sample = samples.find((item) => item.id === id);
    if (!sample) return null;
    const tensor = await tensorFromBin(sample.pixels);
    const read = forward(model, tensor);
    return {
      id: sample.id,
      label: `${sample.id} · ${read.family}`,
      vector: readingVector(model, read),
      family: read.family,
    };
  }

  async function onSelect(side, id) {
    setVerdict("");
    setScore(null);
    setError("");
    if (!id) {
      if (side === "left") setLeft(null);
      else setRight(null);
      setHint("Choose two scraps.");
      return;
    }
    try {
      const loaded = await loadId(id);
      if (side === "left") setLeft(loaded);
      else setRight(loaded);
      setHint("");
    } catch (err) {
      setError(err.message || "This file could not be read as an image.");
    }
  }

  async function onUpload(side, file) {
    setVerdict("");
    setScore(null);
    setError("");
    try {
      const loaded = await fileToTensor(file);
      const read = forward(model, loaded.tensor);
      URL.revokeObjectURL(loaded.url);
      const packed = {
        id: loaded.name,
        label: `${loaded.name} · ${read.family}`,
        vector: readingVector(model, read),
        family: read.family,
      };
      if (side === "left") setLeft(packed);
      else setRight(packed);
      setHint("");
    } catch (err) {
      setError(err.message || "This file could not be read as an image.");
    }
  }

  function compare() {
    if (!left || !right) {
      setVerdict("");
      setScore(null);
      setHint("Choose two scraps.");
      return;
    }
    const value = cosine(left.vector, right.vector);
    setScore(value);
    setHint("");
    setVerdict(
      value >= model.sameFamilyCosineMin
        ? "Same family — these scraps sit close in the network's reading."
        : "Different family — do not treat these as one lot.",
    );
  }

  return (
    <section className="panel compare-panel">
      <div className="panel-head">
        <h2>Compare two scraps</h2>
        <p>
          {model.compareOn === "softmax"
            ? "Cosine of the six family probabilities."
            : model.compareOn === "logits"
              ? "Cosine of the six family scores, with their average removed."
            : model.compareOn === "centered"
              ? "Cosine of the 48-number embedding, with the shared offset removed."
              : "Cosine of the 48-number embedding."}{" "}
          Same family if the score is at least {model.sameFamilyCosineMin.toFixed(3)}.
        </p>
      </div>
      <div className="compare-grid">
        {["left", "right"].map((side) => {
          const current = side === "left" ? left : right;
          return (
            <div key={side} className="compare-slot">
              <label>
                {side === "left" ? "First scrap" : "Second scrap"}
                <select
                  data-testid={side === "left" ? "compare-select-a" : "compare-select-b"}
                  aria-label={side === "left" ? "First scrap" : "Second scrap"}
                  defaultValue=""
                  onChange={(event) => onSelect(side, event.target.value)}
                >
                  <option value="">Choose a scrap</option>
                  {samples.map((sample) => (
                    <option key={sample.id} value={sample.id}>
                      {sample.id} {sample.family}
                    </option>
                  ))}
                </select>
              </label>
              <label className="upload-label slim">
                Or upload
                <input
                  data-testid={side === "left" ? "compare-upload-a" : "compare-upload-b"}
                  aria-label={side === "left" ? "Upload first scrap" : "Upload second scrap"}
                  type="file"
                  accept="image/png,image/jpeg,image/webp,.png,.jpg,.jpeg,.webp"
                  onChange={(event) => {
                    const file = event.target.files?.[0];
                    event.target.value = "";
                    if (file) onUpload(side, file);
                  }}
                />
              </label>
              {current && <p className="quiet">{current.label}</p>}
            </div>
          );
        })}
      </div>
      <div className="action-row">
        <button type="button" data-testid="compare-run" onClick={compare}>
          Compare
        </button>
      </div>
      {error && (
        <p className="alert" role="alert">
          {error}
        </p>
      )}
      {hint && (
        <p data-testid="compare-hint" className="empty">
          {hint}
        </p>
      )}
      {verdict && (
        <div className="verdict-block">
          <p data-testid="verdict">{verdict}</p>
          <p data-testid="cosine">Cosine {score.toFixed(3)}</p>
        </div>
      )}
    </section>
  );
}
