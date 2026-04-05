/**
 * ForgeNet-X — Frontend Logic
 * Handles: file uploads, API calls, UI updates, dashboard rendering
 */

// ── State ────────────────────────────────────────────────────────────────────
const STATE = {
  hwFilename   : null,   // uploaded handwriting filename
  genFilename  : null,   // generated output filename
  origSig      : null,   // original signature filename
  testSig      : null,   // test signature filename
  analysis     : {},     // last analysis result dict
};

// ── DOM shortcuts ─────────────────────────────────────────────────────────────
const $  = id => document.getElementById(id);
const qs = sel => document.querySelector(sel);

// ── Overlay helpers ───────────────────────────────────────────────────────────
function showOverlay(msg = "Processing…") {
  $("overlay-msg").textContent = msg;
  $("overlay").classList.remove("hidden");
}
function hideOverlay() {
  $("overlay").classList.add("hidden");
}

// ── Generic fetch wrapper ─────────────────────────────────────────────────────
async function apiFetch(url, options = {}) {
  const res  = await fetch(url, options);
  const data = await res.json();
  if (!res.ok) throw new Error(data.error || "Server error");
  return data;
}

// ── Show result message ────────────────────────────────────────────────────────
function showResult(elId, msg, type = "success") {
  const el = $(elId);
  el.textContent = msg;
  el.className = `result-box ${type}`;
  el.classList.remove("hidden");
}

// ══════════════════════════════════════════════════════════════════════════════
// MODULE 1 — Handwriting
// ══════════════════════════════════════════════════════════════════════════════

// Drag-and-drop for handwriting upload zone
(function setupHwDrop() {
  const zone = $("hw-drop-zone");
  const input = $("hw-file");

  zone.addEventListener("dragover", e => {
    e.preventDefault();
    zone.classList.add("drag-over");
  });
  zone.addEventListener("dragleave", () => zone.classList.remove("drag-over"));
  zone.addEventListener("drop", e => {
    e.preventDefault();
    zone.classList.remove("drag-over");
    const f = e.dataTransfer.files[0];
    if (f) previewHw(f);
  });
  zone.addEventListener("click", () => input.click());
  input.addEventListener("change", () => {
    if (input.files[0]) previewHw(input.files[0]);
  });
})();

function previewHw(file) {
  const reader = new FileReader();
  reader.onload = e => {
    $("hw-preview").src = e.target.result;
    $("hw-preview-wrap").classList.remove("hidden");
    $("hw-drop-zone").classList.add("hidden");
  };
  reader.readAsDataURL(file);
  // Store file reference
  const dt = new DataTransfer();
  dt.items.add(file);
  $("hw-file").files = dt.files;
}

function clearHw() {
  $("hw-preview-wrap").classList.add("hidden");
  $("hw-drop-zone").classList.remove("hidden");
  $("hw-file").value = "";
  STATE.hwFilename = null;
}

async function uploadHandwriting() {
  const file = $("hw-file").files[0];
  if (!file) { alert("Please select a handwriting image first."); return; }

  showOverlay("Analysing handwriting…");
  const formData = new FormData();
  formData.append("file", file);

  try {
    const data = await apiFetch("/upload_handwriting", {
      method: "POST", body: formData
    });
    STATE.hwFilename = data.filename;
    showResult("hw-result",
      `✓ Processed  |  Characters detected: ${data.char_count}  |  File: ${data.filename}`,
      "success"
    );
    activateStep(2);
  } catch (err) {
    showResult("hw-result", "✗ " + err.message, "error");
  } finally {
    hideOverlay();
  }
}

async function generateText() {
  if (!STATE.hwFilename) {
    alert("Please upload and analyse a handwriting sample first.");
    return;
  }
  const text = $("gen-text").value.trim();
  if (!text) { alert("Please enter some text to generate."); return; }

  showOverlay("Generating handwriting…");
  try {
    const data = await apiFetch("/generate_text", {
      method  : "POST",
      headers : { "Content-Type": "application/json" },
      body    : JSON.stringify({ hw_filename: STATE.hwFilename, text }),
    });
    STATE.genFilename = data.output_file;
    showResult("gen-result", `✓ Generated: ${data.output_file}`, "success");

    // Show preview via download URL
    $("gen-preview").src = data.download_url;
    $("gen-download").href = data.download_url;
    $("gen-preview-wrap").classList.remove("hidden");
  } catch (err) {
    showResult("gen-result", "✗ " + err.message, "error");
  } finally {
    hideOverlay();
  }
}

// ══════════════════════════════════════════════════════════════════════════════
// MODULE 2 — Signatures
// ══════════════════════════════════════════════════════════════════════════════

// Setup drag-drop for both signature zones
["orig", "test"].forEach(prefix => {
  const zone  = $(`${prefix}-drop-zone`);
  const input = $(`${prefix}-sig-file`);

  zone.addEventListener("dragover", e => { e.preventDefault(); zone.classList.add("drag-over"); });
  zone.addEventListener("dragleave", () => zone.classList.remove("drag-over"));
  zone.addEventListener("drop", e => {
    e.preventDefault();
    zone.classList.remove("drag-over");
    const f = e.dataTransfer.files[0];
    if (f) previewSig(f, prefix);
  });
  zone.addEventListener("click", () => input.click());
  input.addEventListener("change", () => {
    if (input.files[0]) previewSig(input.files[0], prefix);
  });
});

function previewSig(file, prefix) {
  const reader = new FileReader();
  reader.onload = e => {
    $(`${prefix}-preview`).src = e.target.result;
    $(`${prefix}-preview-wrap`).classList.remove("hidden");
  };
  reader.readAsDataURL(file);

  const dt = new DataTransfer();
  dt.items.add(file);
  $(`${prefix}-sig-file`).files = dt.files;
}

async function uploadSignatures() {
  const origFile = $("orig-sig-file").files[0];
  const testFile = $("test-sig-file").files[0];

  if (!origFile || !testFile) {
    alert("Please select both original and test signature images.");
    return;
  }

  showOverlay("Uploading signatures…");
  const formData = new FormData();
  formData.append("original", origFile);
  formData.append("test",     testFile);

  try {
    const data = await apiFetch("/upload_signature", {
      method: "POST", body: formData
    });
    STATE.origSig = data.original;
    STATE.testSig = data.test;
    showResult("sig-upload-result",
      `✓ Signatures uploaded  |  Original: ${data.original}  |  Test: ${data.test}`,
      "success"
    );
    $("btn-analyze").disabled = false;
    activateStep(3);
  } catch (err) {
    showResult("sig-upload-result", "✗ " + err.message, "error");
  } finally {
    hideOverlay();
  }
}

async function analyzeSignatures() {
  if (!STATE.origSig || !STATE.testSig) {
    alert("Upload both signatures first.");
    return;
  }

  showOverlay("Running forgery analysis…");
  try {
    const data = await apiFetch("/analyze_signature", {
      method  : "POST",
      headers : { "Content-Type": "application/json" },
      body    : JSON.stringify({ original: STATE.origSig, test: STATE.testSig }),
    });
    STATE.analysis = data;
    renderDashboard(data);
    activateStep(4);
  } catch (err) {
    alert("Analysis error: " + err.message);
  } finally {
    hideOverlay();
  }
}

// ── Dashboard rendering ───────────────────────────────────────────────────────
function renderDashboard(data) {
  $("analysis-dashboard").classList.remove("hidden");

  // Score ring (circumference = 2πr = 2π×50 ≈ 314)
  const pct    = data.similarity_score / 100;
  const offset = 314 - (314 * pct);
  const ring   = $("ring-fill");
  ring.style.strokeDashoffset = offset;
  const riskColor = { green: "#27ae60", orange: "#f39c12", red: "#c0392b" };
  ring.style.stroke = riskColor[data.risk_color] || "#4f8ef7";

  $("score-text").textContent = data.similarity_score.toFixed(1) + "%";

  // Risk badge
  const badge = $("risk-badge");
  badge.textContent = data.risk_level;
  badge.className   = "risk-badge";
  if (data.risk_color === "orange") badge.classList.add("orange");
  if (data.risk_color === "red")    badge.classList.add("red");

  // Metric bars
  setBar("ssim",  data.ssim_score);
  setBar("hu",    data.hu_score);
  setBar("hist",  data.hist_score);
  setBar("pixel", data.pixel_score);

  // Verdict
  $("verdict-box").innerHTML =
    `<strong>🔍 Verdict:</strong> ${data.verdict}`;

  // Feature comparison table
  if (data.features_original && data.features_test) {
    renderFeatureTable(data.features_original, data.features_test);
  }
}

function setBar(key, value) {
  const pct = Math.min(100, Math.max(0, value || 0));
  $(`bar-${key}`).style.width  = pct + "%";
  $(`val-${key}`).textContent  = pct.toFixed(1) + "%";
  // Colour by score
  $(`bar-${key}`).style.background =
    pct >= 80 ? "var(--green)" :
    pct >= 50 ? "var(--orange)" : "var(--red)";
}

function renderFeatureTable(orig, test) {
  const labels = {
    ink_density_pct    : "Ink Density (%)",
    aspect_ratio       : "Aspect Ratio",
    stroke_width_px    : "Avg Stroke Width (px)",
    n_stroke_segments  : "Stroke Segments",
    centroid_x         : "Centroid X",
    centroid_y         : "Centroid Y",
  };

  let rows = "";
  for (const [key, label] of Object.entries(labels)) {
    const ov = orig[key] ?? "—";
    const tv = test[key] ?? "—";
    let deltaHtml = "—";
    if (typeof ov === "number" && typeof tv === "number") {
      const d = tv - ov;
      const cls = d > 0 ? "delta-pos" : d < 0 ? "delta-neg" : "";
      deltaHtml = `<span class="${cls}">${d > 0 ? "+" : ""}${d.toFixed(3)}</span>`;
    }
    rows += `<tr>
      <td>${label}</td>
      <td>${ov}</td>
      <td>${tv}</td>
      <td>${deltaHtml}</td>
    </tr>`;
  }

  $("features-table-wrap").innerHTML = `
    <table class="feature-table" style="margin-top:1rem">
      <thead>
        <tr>
          <th>Feature</th><th>Original</th><th>Test</th><th>Δ Delta</th>
        </tr>
      </thead>
      <tbody>${rows}</tbody>
    </table>`;
}

// ══════════════════════════════════════════════════════════════════════════════
// MODULE 3 — Report
// ══════════════════════════════════════════════════════════════════════════════

async function downloadReport() {
  showOverlay("Generating PDF report…");
  try {
    const res = await fetch("/download_report", {
      method  : "POST",
      headers : { "Content-Type": "application/json" },
      body    : JSON.stringify({
        hw_filename  : STATE.hwFilename  || "",
        gen_filename : STATE.genFilename || "",
        original_sig : STATE.origSig     || "",
        test_sig     : STATE.testSig     || "",
        analysis     : STATE.analysis,
      }),
    });

    if (!res.ok) {
      const err = await res.json();
      throw new Error(err.error || "Report generation failed");
    }

    // Trigger download
    const blob = await res.blob();
    const url  = URL.createObjectURL(blob);
    const a    = document.createElement("a");
    a.href     = url;
    a.download = "ForgeNet-X_Report.pdf";
    a.click();
    URL.revokeObjectURL(url);

    showResult("report-result", "✓ Report downloaded successfully", "success");
  } catch (err) {
    showResult("report-result", "✗ " + err.message, "error");
  } finally {
    hideOverlay();
  }
}

// ── Step indicator helper ─────────────────────────────────────────────────────
function activateStep(n) {
  document.querySelectorAll(".step").forEach((el, i) => {
    el.classList.toggle("active", i + 1 <= n);
  });
}

// ── Init ──────────────────────────────────────────────────────────────────────
document.addEventListener("DOMContentLoaded", () => {
  activateStep(1);
  console.log("ForgeNet-X v1.0 — Ready");
});
