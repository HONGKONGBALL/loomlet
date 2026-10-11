import { useEffect, useRef, useState } from "react";
import { forward } from "../lib/infer.js";
import { fileToTensor, STORAGE_KEY, tensorFromBin } from "../lib/image.js";
import { occlusionMap, paintHeat, paletteHex, textureStd } from "../lib/vision.js";
import { renderRepeat } from "../lib/weave.js";

function decorate(read, tensor) {
  const texture = textureStd(tensor);
  return {
    family: read.family,
    pred: read.pred,
    probs: Array.from(read.probs),
    embedding: read.embedding,
    h1: read.h1,
    h2: read.h2,
    palette: paletteHex(tensor),
    texture,
    uncertain: texture < 0.03,
  };
}

export default function Bench({ model, samples }) {
  const [selectedId, setSelectedId] = useState(null);
  const [tensor, setTensor] = useState(null);
  const [preview, setPreview] = useState("");
  const [previewAlt, setPreviewAlt] = useState("");
  const [source, setSource] = useState("");
  const [result, setResult] = useState(null);
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(false);
  const [seed, setSeed] = useState("1");
  const [fidelity, setFidelity] = useState(0);
  const [weaveMeta, setWeaveMeta] = useState("");
  const [woven, setWoven] = useState(false);
  const [restored, setRestored] = useState(false);
  const [heat, setHeat] = useState(null);
  const [mapping, setMapping] = useState(false);
  const [dragOver, setDragOver] = useState(false);
  const loadGen = useRef(0);
  const mapGen = useRef(0);
  const weaveRef = useRef(null);
  const heatRef = useRef(null);
  const blobRef = useRef("");
  const didRestore = useRef(false);

  function clearBlob() {
    if (blobRef.current) URL.revokeObjectURL(blobRef.current);
    blobRef.current = "";
  }

  async function chooseSample(sample, autoRead) {
    const gen = ++loadGen.current;
    mapGen.current += 1;
    setSelectedId(sample.id);
    setSource("sample");
    setError("");
    setResult(null);
    setHeat(null);
    setWoven(false);
    setWeaveMeta("");
    setLoading(true);
    setTensor(null);
    clearBlob();
    try {
      const next = await tensorFromBin(sample.pixels);
      if (gen !== loadGen.current) return;
      setTensor(next);
      setPreview(sample.image);
      setPreviewAlt(`Fictional sample. ${sample.family}, lot ${sample.id}.`);
      if (autoRead) {
        const read = decorate(forward(model, next), next);
        setResult(read);
        localStorage.setItem(STORAGE_KEY, JSON.stringify({ sampleId: sample.id }));
      }
    } catch (err) {
      if (gen !== loadGen.current) return;
      setError(err.message || "This file could not be read as an image.");
      setPreview("");
    } finally {
      if (gen === loadGen.current) setLoading(false);
    }
  }

  useEffect(() => {
    if (!model || !samples || didRestore.current) return;
    didRestore.current = true;
    let saved = null;
    try {
      saved = JSON.parse(localStorage.getItem(STORAGE_KEY) || "null");
    } catch {
      saved = null;
    }
    const sample = samples.find((item) => item.id === saved?.sampleId);
    if (!sample) return;
    setRestored(true);
    chooseSample(sample, true);
    // chooseSample is recreated each render; restore runs once.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [model, samples]);

  useEffect(() => {
    if (!heat || !heatRef.current) return;
    paintHeat(heatRef.current, heat.heat, heat.grid);
  }, [heat]);

  async function onFile(file) {
    const gen = ++loadGen.current;
    mapGen.current += 1;
    setSelectedId(null);
    setSource("upload");
    setResult(null);
    setHeat(null);
    setWoven(false);
    setWeaveMeta("");
    setError("");
    setRestored(false);
    setLoading(true);
    setTensor(null);
    clearBlob();
    try {
      const loaded = await fileToTensor(file);
      if (gen !== loadGen.current) return;
      blobRef.current = loaded.url;
      setTensor(loaded.tensor);
      setPreview(loaded.url);
      setPreviewAlt(`Uploaded photo ${loaded.name}.`);
      localStorage.removeItem(STORAGE_KEY);
    } catch (err) {
      if (gen !== loadGen.current) return;
      setError(err.message || "This file could not be read as an image.");
      setPreview("");
    } finally {
      if (gen === loadGen.current) setLoading(false);
    }
  }

  function readScrap() {
    if (!tensor) return;
    const read = decorate(forward(model, tensor), tensor);
    setResult(read);
    setHeat(null);
    setWoven(false);
    setWeaveMeta("");
    if (source === "sample" && selectedId) {
      localStorage.setItem(STORAGE_KEY, JSON.stringify({ sampleId: selectedId }));
    } else {
      localStorage.removeItem(STORAGE_KEY);
    }
  }

  function weave() {
    const parsed = Number(seed);
    if (seed === "" || !result || !Number.isInteger(parsed) || parsed < 0 || parsed > 999) {
      setWeaveMeta("");
      setWoven(false);
      setError("Enter a seed from 0 to 999.");
      return;
    }
    setError("");
    const tile = renderRepeat({
      model,
      h1: result.h1,
      h2: result.h2,
      tensor,
      familyIndex: result.pred,
      seed: parsed,
      fidelity: Number(fidelity),
    });
    const canvas = weaveRef.current;
    canvas.width = tile.width;
    canvas.height = tile.height;
    const ctx = canvas.getContext("2d", { willReadFrequently: true });
    ctx.putImageData(new ImageData(tile.rgba, tile.width, tile.height), 0, 0);
    setWoven(true);
    setWeaveMeta(`Seed ${parsed} · fidelity ${Number(fidelity)}`);
  }

  function downloadRepeat() {
    const canvas = weaveRef.current;
    if (!canvas?.width) return;
    canvas.toBlob((blob) => {
      if (!blob) return;
      const url = URL.createObjectURL(blob);
      const link = document.createElement("a");
      link.href = url;
      link.download = `loomlet-repeat-${selectedId || "upload"}-seed${seed}.png`;
      document.body.appendChild(link);
      link.click();
      link.remove();
      URL.revokeObjectURL(url);
    }, "image/png");
  }

  async function showMap() {
    if (!result || !tensor) return;
    const gen = ++mapGen.current;
    setMapping(true);
    const next = await occlusionMap(model, tensor, result.pred, 4);
    if (gen !== mapGen.current) return;
    setHeat(next);
    setMapping(false);
  }

  return (
    <div className="bench-grid">
      <section className="panel" data-testid="gallery" aria-label="Fictional sample lots">
        <div className="panel-head">
          <h2>Scrap bench</h2>
          <p>Twelve fictional lots. The network has not seen a real mill.</p>
        </div>
        <div className="sample-grid">
          {samples.map((sample) => (
            <button
              key={sample.id}
              type="button"
              className="sample"
              data-testid={`sample-${sample.id}`}
              aria-pressed={selectedId === sample.id}
              onClick={() => {
                setRestored(false);
                chooseSample(sample, false);
              }}
            >
              <img src={sample.image} alt="" width="256" height="256" />
              <span className="sample-id">{sample.id}</span>
              <span className="sample-family">{sample.family}</span>
              <span className="sample-note">Fictional sample · {sample.nickname}</span>
            </button>
          ))}
        </div>
        <div
          className={dragOver ? "drop drop-on" : "drop"}
          onDragOver={(event) => {
            event.preventDefault();
            setDragOver(true);
          }}
          onDragLeave={() => setDragOver(false)}
          onDrop={(event) => {
            event.preventDefault();
            setDragOver(false);
            const file = event.dataTransfer.files?.[0];
            if (file) onFile(file);
          }}
        >
          <label className="upload-label">
            Upload a photo
            <input
              data-testid="upload"
              aria-label="Upload a photo"
              type="file"
              accept="image/png,image/jpeg,image/webp,.png,.jpg,.jpeg,.webp"
              onChange={(event) => {
                const file = event.target.files?.[0];
                event.target.value = "";
                if (file) onFile(file);
              }}
            />
          </label>
          <p>PNG, JPEG, or WebP. Under 8 MB. It stays in this browser.</p>
        </div>
      </section>

      <section className="panel stage" data-testid="stage" aria-label="Reading">
        {!tensor && !error && (
          <p className="empty" data-testid="empty-hint">
            Choose a sample scrap or upload a photo.
          </p>
        )}
        {loading && <p className="quiet">Opening the scrap…</p>}
        {error && (
          <p className="alert" role="alert" data-testid="form-error">
            {error}
          </p>
        )}
        {preview && (
          <figure className="preview-figure">
            <img src={preview} alt={previewAlt} />
            <figcaption>{previewAlt}</figcaption>
          </figure>
        )}
        {restored && result && (
          <p className="quiet" data-testid="restored-note">
            Last scrap restored from this browser. The network just read it again.
          </p>
        )}
        <div className="action-row">
          <button type="button" data-testid="classify" onClick={readScrap} disabled={!tensor || loading}>
            Read this scrap
          </button>
        </div>

        {result && (
          <div className="result" aria-live="polite">
            <p className="eyebrow">Nearest fictional family</p>
            <h3 data-testid="family-name">{result.family}</h3>
            <p className="confidence">
              <span data-testid="confidence">{(result.probs[result.pred] * 100).toFixed(1)}</span>
              <span aria-hidden="true">%</span>
            </p>
            {result.uncertain && (
              <p className="alert" data-testid="uncertain">
                This photo has almost no pattern, so the family call is unreliable.
              </p>
            )}
            <ol className="probs">
              {model.classes.map((name, index) => (
                <li
                  key={name}
                  className={index === result.pred ? "prob is-top" : "prob"}
                  data-testid={`prob-${index}`}
                  data-p={result.probs[index].toFixed(6)}
                >
                  <span>{name}</span>
                  <span>{(result.probs[index] * 100).toFixed(1)}%</span>
                  <span className="bar" aria-hidden="true">
                    <span style={{ width: `${result.probs[index] * 100}%` }} />
                  </span>
                </li>
              ))}
            </ol>

            <h3 className="section-label">Measured dyes</h3>
            <div className="swatches" data-testid="palette">
              {result.palette.map((hex, index) => (
                <div key={`${hex}-${index}`} className="swatch" data-testid="swatch">
                  <i style={{ background: hex }} />
                  <span data-testid="swatch-hex">{hex}</span>
                </div>
              ))}
            </div>

            <div className="action-row">
              <button type="button" className="secondary" data-testid="occlusion" onClick={showMap} disabled={mapping}>
                {mapping ? "Mapping pixels…" : "Show what mattered"}
              </button>
            </div>
            {heat && (
              <figure className="map-figure">
                <canvas ref={heatRef} data-testid="occlusion-map" aria-label="Occlusion map" />
                <figcaption>Darker red means covering that block hurt the predicted family more.</figcaption>
              </figure>
            )}

            <h3 className="section-label">Weave a repeat</h3>
            <p className="quiet">
              A second network lifts a motif from this scrap. The seed shifts its phase. Fidelity pulls the motif and the dyes toward the family prototype.
            </p>
            <div className="weave-controls">
              <label>
                Seed
                <input
                  data-testid="seed"
                  type="number"
                  min="0"
                  max="999"
                  value={seed}
                  onChange={(event) => setSeed(event.target.value)}
                />
              </label>
              <label>
                Pull toward family prototype
                <input
                  data-testid="fidelity"
                  type="range"
                  min="0"
                  max="100"
                  value={fidelity}
                  onChange={(event) => setFidelity(Number(event.target.value))}
                />
                <span data-testid="fidelity-value">{fidelity}</span>
              </label>
            </div>
            <div className="action-row">
              <button type="button" data-testid="weave" onClick={weave}>
                Weave repeat
              </button>
              <button
                type="button"
                className="secondary"
                data-testid="download-repeat"
                onClick={downloadRepeat}
                disabled={!woven}
              >
                Download repeat PNG
              </button>
            </div>
            <canvas
              ref={weaveRef}
              data-testid="weave-preview"
              className={woven ? "weave-preview" : "weave-preview hidden-canvas"}
              aria-label="Woven repeat preview"
            />
            {weaveMeta && (
              <p className="quiet" data-testid="weave-meta">
                {weaveMeta}
              </p>
            )}
          </div>
        )}
      </section>
    </div>
  );
}
