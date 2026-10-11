export default function About({ metrics }) {
  const pct = metrics ? (metrics.holdoutAccuracy * 100).toFixed(1) : null;
  return (
    <article className="panel about">
      <div className="panel-head">
        <h2>How Loomlet was built</h2>
        <p>A solo bench for Banana Hacks 2026. Zhenhua, GitHub HONGKONGBALL.</p>
      </div>

      <h3>Brainstorming</h3>
      <p>
        The theme is image AI. I wanted the model to be the product, not a button that calls a hosted generator, and I wanted the network panel to stay quiet while a scrap is read. A ripeness sorter needed a dataset I should not invent. A poster checker was mostly geometry. The bench that remained looks at a scrap, names a fictional weave family, and weaves a repeat from that same photo.
      </p>

      <h3>Inspiration</h3>
      <p>
        Student costume shops and print clubs sort donated cloth from phone photos. Matching dyes get called the same weave, and the repeat is then cut from the wrong structure. The twelve lots — Pier, Fen, Kiln, Lamp, Quay, and Loom — are invented. Each card says fictional sample so the demo never pretends to know a real mill.
      </p>

      <h3>Design</h3>
      <p>
        One page, three views. The bench reads a 32×32 view of the photo. Compare checks cosine distance between two readings. About keeps the limits next to the score. The occlusion grid covers blocks of the photo and shows where the predicted score drops. Five dyes are measured from the pixels with k-means. The repeat is a decoded motif mask, colored with the scrap's dyes, phased by a seed.
      </p>

      <h3>Resources</h3>
      <p>
        PyTorch trains the weights on this machine. The page runs the same convolutions in JavaScript, using the JSON file shipped with the project. Fraunces and Outfit are bundled under the SIL Open Font License. There is no API key. The sample images are drawn by <code>train/train_loomlet.py</code>. You can use the bench without retraining.
      </p>

      <h3>The bench</h3>
      <p>
        The family call is a convolution network, not a lookup of the file name. Open a lot, read it, check the map, weave a repeat, and compare two lots. A flat gray photo is marked unreliable. A text file is refused. Refreshing restores the last fictional lot and runs the network again.
      </p>

      {metrics ? (
        <>
          <p data-testid="about-accuracy">
            {pct}% on the synthetic holdout ({metrics.holdoutCount} images).
          </p>
          <p>
            On the twelve fictional lots, same-family cosine is at least {metrics.samePairMin.toFixed(3)} and different-family cosine is at most {metrics.diffPairMax.toFixed(3)}. The compare view uses {metrics.sameFamilyCosineMin.toFixed(3)}.
          </p>
          <p>{metrics.note}</p>
        </>
      ) : (
        <p>Holdout score did not load.</p>
      )}

      <h3>Limits</h3>
      <p>
        The taxonomy is six synthetic families. A real tweed is outside that generator. The bench still names the nearest family, which is why low-texture photos are flagged. This is not a commercial fabric identifier.
      </p>
    </article>
  );
}
