import fs from "node:fs";
import path from "node:path";
import { expect, test } from "@playwright/test";

const LOTS = [
  "HS-014",
  "HS-027",
  "MG-008",
  "MG-033",
  "CD-002",
  "CD-019",
  "NS-041",
  "NS-056",
  "TW-011",
  "TW-028",
  "CC-006",
  "CC-022",
];

const evidenceDir = path.resolve("qa-evidence");
fs.mkdirSync(evidenceDir, { recursive: true });

async function hashPreview(page) {
  return page.evaluate(() => {
    const canvas = document.querySelector("[data-testid='weave-preview']");
    const data = canvas.getContext("2d").getImageData(0, 0, canvas.width, canvas.height).data;
    let hash = 2166136261;
    for (let i = 0; i < data.length; i += 7) {
      hash ^= data[i];
      hash = Math.imul(hash, 16777619);
    }
    return hash >>> 0;
  });
}

async function readLot(page, id) {
  await page.getByTestId(`sample-${id}`).click();
  await expect(page.getByTestId("classify")).toBeEnabled();
  await page.getByTestId("classify").click();
  await expect(page.getByTestId("family-name")).toBeVisible();
}

test.beforeEach(async ({ page }) => {
  page.on("pageerror", (error) => {
    throw error;
  });
});

test.afterEach(async ({ page }, testInfo) => {
  const id = (testInfo.title.match(/AC-\d+/) || ["setup"])[0];
  const shot = path.join(evidenceDir, `${id}.png`);
  if (!page.isClosed()) await page.screenshot({ path: shot, fullPage: true }).catch(() => {});
  fs.appendFileSync(
    path.join(evidenceDir, "results.jsonl"),
    `${JSON.stringify({
      id,
      title: testInfo.title,
      status: testInfo.status,
      evidence: shot,
      error: testInfo.error ? String(testInfo.error.message).split("\n")[0] : null,
    })}\n`,
  );
});

test("AC-01 page load names Loomlet and stays on-device", async ({ page }) => {
  const response = await page.goto("/");
  expect(response.status()).toBe(200);
  await expect(page.getByRole("heading", { name: "Loomlet" })).toBeVisible();
  await expect(page.getByText("On-device")).toBeVisible();
  await expect(page.getByText("Photos never leave this browser.")).toBeVisible();
});

test("AC-02 weight file is JSON with six families", async ({ request }) => {
  const response = await request.get("/model/loomlet-weights.json");
  expect(response.status()).toBe(200);
  const body = await response.json();
  expect(body.classes).toHaveLength(6);
  expect(body.classes[0]).toBe("Harbor Stripe");
  expect(body.cls1.w.length).toBeGreaterThan(10);
  expect(body.sameFamilyCosineMin).toBeGreaterThan(0);
  expect(body.sameFamilyCosineMin).toBeLessThan(1);
});

test("AC-03 every sample PNG is served", async ({ request }) => {
  for (const id of LOTS) {
    const response = await request.get(`/samples/${id}.png`);
    expect(response.status(), id).toBe(200);
    expect(response.headers()["content-type"]).toContain("image/png");
  }
});

test("AC-04 gallery lists twelve fictional lots", async ({ page }) => {
  await page.goto("/");
  await expect(page.getByTestId("sample-HS-014")).toBeVisible();
  await expect(page.locator("[data-testid^='sample-']")).toHaveCount(12);
  for (const id of LOTS) {
    const card = page.getByTestId(`sample-${id}`);
    await expect(card).toContainText(id);
    await expect(card).toContainText("Fictional sample");
  }
});

test("AC-05 HS-014 reads as Harbor Stripe", async ({ page }) => {
  await page.goto("/");
  await readLot(page, "HS-014");
  await expect(page.getByTestId("family-name")).toHaveText("Harbor Stripe");
  const confidence = Number(await page.getByTestId("confidence").textContent());
  expect(confidence).toBeGreaterThanOrEqual(70);
});

test("AC-06 probabilities cover the six families and sum to 100", async ({ page }) => {
  await page.goto("/");
  await readLot(page, "HS-014");
  const rows = await page.locator("[data-testid^='prob-']").evaluateAll((nodes) =>
    nodes.map((node) => ({
      p: Number(node.dataset.p),
      text: node.textContent,
    })),
  );
  expect(rows).toHaveLength(6);
  let sum = 0;
  for (const row of rows) {
    expect(row.p * 100).toBeGreaterThanOrEqual(0);
    expect(row.p * 100).toBeLessThanOrEqual(100);
    sum += row.p * 100;
    const shown = Number(row.text.match(/(\d+\.\d)%/)[1]);
    expect(Math.abs(shown - row.p * 100)).toBeLessThan(0.06);
  }
  expect(Math.abs(sum - 100)).toBeLessThanOrEqual(1.5);
});

test("AC-07 palette shows five hex dyes", async ({ page }) => {
  await page.goto("/");
  await readLot(page, "HS-014");
  const hexes = page.getByTestId("swatch-hex");
  await expect(hexes).toHaveCount(5);
  for (const text of await hexes.allTextContents()) expect(text).toMatch(/^#[0-9a-f]{6}$/);
});

test("AC-08 weave seed is deterministic and downloads a PNG", async ({ page }) => {
  await page.goto("/");
  await readLot(page, "HS-014");
  await page.getByTestId("seed").fill("1");
  await page.getByTestId("weave").click();
  await expect(page.getByTestId("weave-meta")).toHaveText("Seed 1 · fidelity 0");
  const first = await hashPreview(page);
  await page.getByTestId("seed").fill("7");
  await page.getByTestId("weave").click();
  await expect(page.getByTestId("weave-meta")).toHaveText("Seed 7 · fidelity 0");
  const second = await hashPreview(page);
  expect(second).not.toBe(first);
  await page.getByTestId("seed").fill("1");
  await page.getByTestId("weave").click();
  expect(await hashPreview(page)).toBe(first);
  const [download] = await Promise.all([
    page.waitForEvent("download"),
    page.getByTestId("download-repeat").click(),
  ]);
  const filePath = await download.path();
  const bytes = fs.readFileSync(filePath);
  expect(bytes.subarray(0, 4).toString("hex")).toBe("89504e47");
  expect(bytes.length).toBeGreaterThan(800);
  fs.writeFileSync(path.join(evidenceDir, "AC-08-repeat.png"), bytes);
});

test("AC-09 compare agrees with family and rejects a mismatch", async ({ page }) => {
  await page.goto("/#compare");
  await page.getByTestId("compare-select-a").selectOption("HS-014");
  await page.getByTestId("compare-select-b").selectOption("HS-027");
  await expect(page.locator(".compare-slot").nth(1).locator(".quiet")).toContainText("HS-027");
  await page.getByTestId("compare-run").click();
  await expect(page.getByTestId("verdict")).toContainText("Same family");
  await expect(page.getByTestId("cosine")).toContainText("Cosine");
  await page.getByTestId("compare-select-b").selectOption("MG-008");
  await expect(page.locator(".compare-slot").nth(1).locator(".quiet")).toContainText("MG-008");
  await page.getByTestId("compare-run").click();
  await expect(page.getByTestId("verdict")).toContainText("Different family");
  await expect(page.getByTestId("cosine")).toContainText("Cosine");
});

test("AC-10 uploaded tide wave is not a gallery lookup", async ({ page }) => {
  await page.goto("/");
  await page.getByTestId("upload").setInputFiles("tests/fixtures/tide-wave.png");
  await expect(page.getByTestId("classify")).toBeEnabled();
  await page.getByTestId("classify").click();
  await expect(page.getByTestId("family-name")).toHaveText("Tide Wave");
  const confidence = Number(await page.getByTestId("confidence").textContent());
  expect(confidence).toBeGreaterThanOrEqual(70);
});

test("AC-11 a flat gray photo is marked unreliable", async ({ page }) => {
  await page.goto("/");
  await page.getByTestId("upload").setInputFiles("tests/fixtures/flat-gray.png");
  await expect(page.getByTestId("classify")).toBeEnabled();
  await page.getByTestId("classify").click();
  await expect(page.getByTestId("uncertain")).toContainText("almost no pattern");
});

test("AC-12 a text file is refused", async ({ page }) => {
  await page.goto("/");
  await page.getByTestId("upload").setInputFiles("tests/fixtures/notes.txt");
  await expect(page.getByTestId("form-error")).toContainText("PNG, JPEG, or WebP");
  await expect(page.getByTestId("family-name")).toHaveCount(0);
});

test("AC-13 an empty bench asks for a scrap", async ({ page }) => {
  await page.goto("/");
  await expect(page.getByTestId("empty-hint")).toHaveText("Choose a sample scrap or upload a photo.");
  await expect(page.getByTestId("classify")).toBeDisabled();
});

test("AC-14 occlusion map is not a flat color", async ({ page }) => {
  await page.goto("/");
  await readLot(page, "HS-014");
  await page.getByTestId("occlusion").click();
  await expect(page.getByTestId("occlusion-map")).toBeVisible();
  const std = await page.getByTestId("occlusion-map").evaluate((canvas) => {
    const data = canvas.getContext("2d").getImageData(0, 0, canvas.width, canvas.height).data;
    let sum = 0;
    let sum2 = 0;
    let count = 0;
    for (let i = 0; i < data.length; i += 4) {
      sum += data[i];
      sum2 += data[i] * data[i];
      count += 1;
    }
    const mean = sum / count;
    return Math.sqrt(sum2 / count - mean * mean);
  });
  expect(std).toBeGreaterThan(2);
});

test("AC-15 a fast second click wins", async ({ page }) => {
  await page.goto("/");
  await page.getByTestId("sample-HS-014").click();
  await page.getByTestId("sample-MG-008").click();
  await expect(page.getByTestId("sample-MG-008")).toHaveAttribute("aria-pressed", "true");
  await expect(page.getByTestId("classify")).toBeEnabled();
  await page.getByTestId("classify").click();
  await expect(page.getByTestId("family-name")).toHaveText("Moss Grid");
});

test("AC-16 reload reads the last fictional lot again", async ({ page }) => {
  await page.goto("/");
  await readLot(page, "HS-014");
  await page.reload();
  await expect(page.getByTestId("sample-HS-014")).toHaveAttribute("aria-pressed", "true");
  await expect(page.getByTestId("family-name")).toHaveText("Harbor Stripe");
});

test("AC-17 mobile width does not overflow", async ({ page }) => {
  await page.setViewportSize({ width: 390, height: 844 });
  await page.goto("/");
  await readLot(page, "HS-014");
  await expect(page.getByTestId("classify")).toBeVisible();
  const benchOverflow = await page.evaluate(
    () => document.documentElement.scrollWidth - document.documentElement.clientWidth,
  );
  expect(benchOverflow).toBeLessThanOrEqual(1);
  await page.goto("/#compare");
  await expect(page.getByRole("heading", { name: "Compare two scraps" })).toBeVisible();
  const compareOverflow = await page.evaluate(
    () => document.documentElement.scrollWidth - document.documentElement.clientWidth,
  );
  expect(compareOverflow).toBeLessThanOrEqual(1);
  await page.goto("/#about");
  await expect(page.getByRole("heading", { name: "How Loomlet was built" })).toBeVisible();
  const aboutOverflow = await page.evaluate(
    () => document.documentElement.scrollWidth - document.documentElement.clientWidth,
  );
  expect(aboutOverflow).toBeLessThanOrEqual(1);
});

test("AC-18 desktop bench places the gallery beside the stage", async ({ page }) => {
  await page.setViewportSize({ width: 1280, height: 800 });
  await page.goto("/");
  await readLot(page, "HS-014");
  const box = await page.evaluate(() => {
    const gallery = document.querySelector("[data-testid='gallery']").getBoundingClientRect();
    const stage = document.querySelector("[data-testid='stage']").getBoundingClientRect();
    return { galleryRight: gallery.right, stageLeft: stage.left, galleryTop: gallery.top, stageTop: stage.top };
  });
  expect(box.stageLeft).toBeGreaterThan(box.galleryRight - 8);
  expect(Math.abs(box.galleryTop - box.stageTop)).toBeLessThan(40);
});

test("AC-19 about shows the synthetic holdout score from the file", async ({ page, request }) => {
  const metrics = await (await request.get("/model/metrics.json")).json();
  await page.goto("/#about");
  await expect(page.getByText("convolution", { exact: false })).toBeVisible();
  await expect(page.getByText("synthetic holdout", { exact: false })).toBeVisible();
  const shown = (metrics.holdoutAccuracy * 100).toFixed(1);
  await expect(page.getByTestId("about-accuracy")).toContainText(`${shown}%`);
  await expect(page.getByTestId("about-accuracy")).toContainText("synthetic holdout");
});

test("AC-20 reading a scrap stays on this origin", async ({ page }) => {
  const hosts = new Set();
  const bad = [];
  page.on("request", (req) => {
    const type = req.resourceType();
    if (["document", "stylesheet", "script", "image", "font", "fetch", "xhr", "manifest"].includes(type)) {
      hosts.add(new URL(req.url()).host);
    }
  });
  page.on("response", (res) => {
    if (res.status() >= 400) bad.push(`${res.status()} ${res.url()}`);
  });
  await page.goto("/");
  await readLot(page, "HS-014");
  await page.getByTestId("seed").fill("1");
  await page.getByTestId("weave").click();
  expect([...hosts]).toEqual(["127.0.0.1:4178"]);
  expect(bad).toEqual([]);
});

test("AC-22 a truncated PNG does not decode", async ({ page }) => {
  await page.goto("/");
  await page.getByTestId("upload").setInputFiles("tests/fixtures/broken.png");
  await expect(page.getByTestId("form-error")).toContainText("could not be read");
  await expect(page.getByTestId("family-name")).toHaveCount(0);
});

test("AC-23 Enter selects a focused lot", async ({ page }) => {
  await page.goto("/");
  await page.getByTestId("sample-CD-002").focus();
  await page.keyboard.press("Enter");
  await expect(page.getByTestId("sample-CD-002")).toHaveAttribute("aria-pressed", "true");
});

test("AC-24 fidelity changes the repeat", async ({ page }) => {
  await page.goto("/");
  await readLot(page, "HS-014");
  await page.getByTestId("seed").fill("1");
  await page.getByTestId("fidelity").fill("0");
  await page.getByTestId("weave").click();
  const low = await hashPreview(page);
  await page.getByTestId("fidelity").fill("100");
  await page.getByTestId("weave").click();
  await expect(page.getByTestId("weave-meta")).toHaveText("Seed 1 · fidelity 100");
  expect(await hashPreview(page)).not.toBe(low);
});

test("AC-25 compare with one scrap does not invent a verdict", async ({ page }) => {
  await page.goto("/#compare");
  await page.getByTestId("compare-select-a").selectOption("HS-014");
  await expect(page.locator(".compare-slot").nth(0).locator(".quiet")).toContainText("HS-014");
  await page.getByTestId("compare-run").click();
  await expect(page.getByTestId("compare-hint")).toHaveText("Choose two scraps.");
  await expect(page.getByTestId("verdict")).toHaveCount(0);
});

test("AC-26 an uploaded JPEG is read by the network", async ({ page }) => {
  await page.goto("/");
  await page.getByTestId("upload").setInputFiles("tests/fixtures/moss-grid.jpg");
  await expect(page.getByTestId("classify")).toBeEnabled();
  await page.getByTestId("classify").click();
  await expect(page.getByTestId("family-name")).toHaveText("Moss Grid");
  const confidence = Number(await page.getByTestId("confidence").textContent());
  expect(confidence).toBeGreaterThanOrEqual(70);
});

test("AC-27 an 8 MB image is refused before it is read", async ({ page }) => {
  const bigPath = path.join(evidenceDir, "big.png");
  const bytes = Buffer.alloc(8 * 1024 * 1024 + 1, 7);
  bytes.write("\x89PNG\r\n\x1a\n");
  fs.writeFileSync(bigPath, bytes);
  await page.goto("/");
  await page.getByTestId("upload").setInputFiles(bigPath);
  await expect(page.getByTestId("form-error")).toContainText("8 MB");
  await expect(page.getByTestId("family-name")).toHaveCount(0);
});
