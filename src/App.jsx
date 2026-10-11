import { useEffect, useState } from "react";
import About from "./components/About.jsx";
import Bench from "./components/Bench.jsx";
import CompareView from "./components/CompareView.jsx";
import { parseModel } from "./lib/infer.js";

function viewFromHash(hash) {
  if (hash === "#compare") return "compare";
  if (hash === "#about") return "about";
  return "bench";
}

export default function App() {
  const [view, setView] = useState(viewFromHash(window.location.hash));
  const [model, setModel] = useState(null);
  const [metrics, setMetrics] = useState(null);
  const [samples, setSamples] = useState(null);
  const [bootError, setBootError] = useState("");

  useEffect(() => {
    const onHash = () => setView(viewFromHash(window.location.hash));
    window.addEventListener("hashchange", onHash);
    return () => window.removeEventListener("hashchange", onHash);
  }, []);

  useEffect(() => {
    let cancel = false;
    (async () => {
      try {
        const [weightsRes, metricsRes, manifestRes] = await Promise.all([
          fetch("/model/loomlet-weights.json"),
          fetch("/model/metrics.json"),
          fetch("/samples/manifest.json"),
        ]);
        if (!weightsRes.ok || !metricsRes.ok || !manifestRes.ok) {
          throw new Error("The on-device model files did not load.");
        }
        const [weights, nextMetrics, manifest] = await Promise.all([
          weightsRes.json(),
          metricsRes.json(),
          manifestRes.json(),
        ]);
        if (cancel) return;
        setModel(parseModel(weights));
        setMetrics(nextMetrics);
        setSamples(manifest.samples);
      } catch (err) {
        if (!cancel) setBootError(err.message || "The on-device model files did not load.");
      }
    })();
    return () => {
      cancel = true;
    };
  }, []);

  return (
    <>
      <a className="skip" href="#content">
        Skip to the bench
      </a>
      <header className="top">
        <a className="brand" href="#bench">
          <svg viewBox="0 0 32 32" aria-hidden="true">
            <rect width="32" height="32" rx="7" fill="#241c16" />
            <path d="M7 10h18M7 16h18M7 22h18M11 7v18M16 7v18M21 7v18" stroke="#f4ecdf" strokeWidth="1.4" />
          </svg>
          <span>
            <h1>Loomlet</h1>
            <small>textile bench</small>
          </span>
        </a>
        <nav aria-label="Views">
          <a href="#bench" aria-current={view === "bench" ? "page" : undefined}>
            Bench
          </a>
          <a href="#compare" aria-current={view === "compare" ? "page" : undefined}>
            Compare
          </a>
          <a href="#about" aria-current={view === "about" ? "page" : undefined}>
            About
          </a>
        </nav>
        <p className="pill">On-device</p>
      </header>
      <main id="content">
        <p className="lede">
          Read the weave. Keep the dye. Loomlet runs a small convolutional network in this page, names a fictional family, and weaves a repeat. Photos never leave this browser.
        </p>
        {bootError && (
          <p className="alert" role="alert">
            {bootError}
          </p>
        )}
        {!model || !samples ? (
          !bootError && <p className="quiet">Loading the weight file…</p>
        ) : (
          <>
            {view === "bench" && <Bench model={model} samples={samples} />}
            {view === "compare" && <CompareView model={model} samples={samples} />}
            {view === "about" && <About metrics={metrics} />}
          </>
        )}
      </main>
      <footer>
        MIT License · Copyright 2026 Zhenhua Liu · Fictional sample data only
      </footer>
    </>
  );
}
