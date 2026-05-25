/**
 * ForgeNet-X — Frontend Logic
 * Detection-first forensic tool. Generation is a research/stress-test sub-feature.
 */

// ── State ────────────────────────────────────────────────────────────────────
const STATE = {
  hwFilename   : null,   // uploaded handwriting filename
  genFilename  : null,   // generated synthetic output filename
  origSig      : null,   // original signature filename
  testSig      : null,   // test signature filename
  analysis     : {},     // last analysis result dict
  provenance   : {},     // last origin analysis result
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
// MODULE A — Handwriting Authentication
// ══════════════════════════════════════════════════════════════════════════════

(function setupHwDrop() {
  const zone  = $("hw-drop-zone");
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
  const dt = new DataTransfer();
  dt.items.add(file);
  $("hw-file").files = dt.files;
}

function clearHw() {
  $("hw-preview-wrap").classList.add("hidden");
  $("hw-drop-zone").classList.remove("hidden");
  $("hw-file").value = "";
  STATE.hwFilename = null;
  $("provenance-card").classList.add("hidden");
  $("hw-result").classList.add("hidden");
}

async function uploadHandwriting() {
  const file = $("hw-file").files[0];
  if (!file) { alert("Please select a handwriting image first."); return; }

  showOverlay("Authenticating handwriting…");
  const formData = new FormData();
  formData.append("file", file);

  try {
    const data = await apiFetch("/upload_handwriting", {
      method: "POST", body: formData
    });
    STATE.hwFilename = data.filename;
    STATE.provenance = data.provenance || {};

    // Structural-context quality badge
    const q    = data.seg_quality || {};
    const det  = q.detected  != null ? q.detected : data.char_count;
    const exp  = q.expected  != null ? q.expected : "?";
    const qual = q.quality   || "unknown";
    const qCls = qual === "exact"        ? "seg-qual-ok"
               : qual.startsWith("near") ? "seg-qual-warn"
               :                           "seg-qual-bad";
    const qBadge = `<span class="seg-quality-badge ${qCls}"
        title="Detected ${det} of ${exp} expected characters (${qual})">${
        qual === "exact" ? "✓ Labels OK" : `⚠ ${det} / ${exp} chars`
      }</span>`;

    showResult("hw-result",
      `✓ Processed  |  Characters detected: ${det}  |  File: ${data.filename}  ${qBadge}`,
      "success"
    );

    renderProvenance(data.provenance);
    activateStep(2);
  } catch (err) {
    showResult("hw-result", "✗ " + err.message, "error");
  } finally {
    hideOverlay();
  }
}

// ── Provenance / Origin Analysis display ──────────────────────────────────────
function renderProvenance(prov) {
  if (!prov) return;

  const card   = $("provenance-card");
  const badge  = $("origin-badge");
  const flags  = $("provenance-flags");
  const summary = $("provenance-summary");

  card.classList.remove("hidden");

  badge.textContent = prov.origin_label || "—";
  badge.className   = "origin-badge";
  if (prov.origin_color === "orange") badge.classList.add("orange");
  if (prov.origin_color === "red")    badge.classList.add("red");

  flags.innerHTML = "";
  (prov.flags || []).forEach(flag => {
    const li = document.createElement("li");
    li.textContent = flag;
    flags.appendChild(li);
  });

  summary.textContent = prov.summary || "";
}

// ══════════════════════════════════════════════════════════════════════════════
// MODULE B — Stress-Test Generator (sub-feature of Module A)
// ══════════════════════════════════════════════════════════════════════════════

async function generateText() {
  if (!STATE.hwFilename) {
    alert("Please upload and authenticate a handwriting sample first.");
    return;
  }

  const consent = $("consent-check");
  if (!consent.checked) {
    alert("Please read and accept the research-purpose declaration before generating a synthetic sample.");
    consent.focus();
    return;
  }

  const text = $("gen-text").value.trim();
  if (!text) { alert("Please enter some text to generate."); return; }

  showOverlay("Generating synthetic sample…");
  try {
    const data = await apiFetch("/generate_text", {
      method  : "POST",
      headers : { "Content-Type": "application/json" },
      body    : JSON.stringify({ hw_filename: STATE.hwFilename, text }),
    });
    STATE.genFilename = data.output_file;
    showResult("gen-result",
      `✓ Synthetic sample generated & watermarked: ${data.output_file}`,
      "success"
    );

    $("gen-preview").src = data.download_url;
    $("gen-download").href = data.download_url;
    $("gen-preview-wrap").classList.remove("hidden");
    activateStep(3);
  } catch (err) {
    showResult("gen-result", "✗ " + err.message, "error");
  } finally {
    hideOverlay();
  }
}

// ══════════════════════════════════════════════════════════════════════════════
// MODULE 1 — Signature Forensics
// ══════════════════════════════════════════════════════════════════════════════

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

  const pct    = data.similarity_score / 100;
  const offset = 314 - (314 * pct);
  const ring   = $("ring-fill");
  ring.style.strokeDashoffset = offset;
  const riskColor = { green: "#27ae60", orange: "#f39c12", red: "#c0392b" };
  ring.style.stroke = riskColor[data.risk_color] || "#4f8ef7";

  $("score-text").textContent = data.similarity_score.toFixed(1) + "%";

  const badge = $("risk-badge");
  badge.textContent = data.risk_level;
  badge.className   = "risk-badge";
  if (data.risk_color === "orange") badge.classList.add("orange");
  if (data.risk_color === "red")    badge.classList.add("red");

  setBar("ssim",  data.ssim_score);
  setBar("hu",    data.hu_score);
  setBar("hist",  data.hist_score);
  setBar("pixel", data.pixel_score);

  $("verdict-box").innerHTML = `<strong>🔍 Forensic Verdict:</strong> ${data.verdict}`;

  if (data.features_original && data.features_test) {
    renderFeatureTable(data.features_original, data.features_test);
  }
}

function setBar(key, value) {
  const pct = Math.min(100, Math.max(0, value || 0));
  $(`bar-${key}`).style.width  = pct + "%";
  $(`val-${key}`).textContent  = pct.toFixed(1) + "%";
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
// MODULE 2 — Forensic Report
// ══════════════════════════════════════════════════════════════════════════════

async function downloadReport() {
  showOverlay("Generating forensic PDF report…");
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

    const blob = await res.blob();
    const url  = URL.createObjectURL(blob);
    const a    = document.createElement("a");
    a.href     = url;
    a.download = "ForgeNet-X_Forensic_Report.pdf";
    a.click();
    URL.revokeObjectURL(url);

    showResult("report-result", "✓ Forensic report downloaded successfully", "success");
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
  console.log("ForgeNet-X v1.0 — Forensic Auth Ready");
});
