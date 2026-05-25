# ForgeNet-X
## Forensic Handwriting & Signature Authenticity System

> Final Year AI/ML Project — Offline Flask Web Application (Windows)

---

## Mission

ForgeNet-X is a **forensic detection tool first**. Its primary purpose is to authenticate handwriting samples and detect signature forgeries using computer vision. The synthetic handwriting generator is a **research sub-feature** — it exists to stress-test the detector and produce augmented training data. All generated outputs are automatically watermarked as `[SYNTHETIC – ForgeNet-X]` to prevent misuse.

See [`ETHICS.md`](ETHICS.md) for the full responsible-use statement.

---

## System Architecture

### Module A — Forensic Analysis Engine (Core)
| Component | Role |
|-----------|------|
| Handwriting Authentication | Preprocesses uploaded samples, extracts characters, runs provenance/origin analysis |
| Signature Forgery Detection | Compares original vs. test signatures via SSIM, Hu Moments, histogram, pixel distance |
| PDF Forensic Report | Bundles all results, visuals, origin analysis, and verdict into a downloadable report |

### Module B — Stress-Test Generator (Research Support)
| Component | Role |
|-----------|------|
| Synthetic Handwriting Engine | Renders typed text in a captured style, bakes `[SYNTHETIC]` watermark |
| Generation Audit Log | Records every generation event with timestamp and declared purpose |

The generator **exists to make the detector better** — the same adversarial relationship used in robustness research. It does not operate as a standalone tool.

---

## Project Structure

```
ForgeNet-X/
│
├── app.py                        ← Flask app & all routes
├── requirements.txt              ← Python dependencies
├── ETHICS.md                     ← Responsible-use & dual-use statement
│
├── templates/
│   └── index.html                ← Forensic UI (detection-first layout)
│
├── static/
│   ├── css/style.css             ← Dark forensics UI
│   └── js/app.js                 ← Frontend logic (consent gate, provenance display)
│
├── uploads/
│   ├── handwriting/              ← Uploaded handwriting images
│   └── signatures/               ← Uploaded signature images
│
├── outputs/
│   ├── generated/                ← Synthetic outputs (watermarked)
│   ├── reports/                  ← PDF forensic report files
│   ├── visuals/                  ← Preprocessing & segmentation annotation PNGs
│   └── generation_log.jsonl      ← Audit trail for every synthetic generation
│
└── utils/
    ├── preprocessing.py          ← CLAHE + adaptive threshold pipeline (8 stages)
    ├── segmentation.py           ← Vertical projection valley character extraction
    ├── handwriting.py            ← Synthetic generator + watermark baking
    ├── signature_analysis.py     ← SSIM, Hu, histogram, pixel forgery metrics
    ├── pdf_generator.py          ← ReportLab forensic PDF builder
    └── provenance.py             ← EXIF metadata + synthetic origin classifier
```

---

## Windows Setup & Installation

### Requirements
- Windows 10 / 11
- Python 3.10 or newer — https://python.org/downloads
  - During install, check **"Add Python to PATH"** ✅
- No internet connection required after first install

### Method 1 — Double-click (easiest)

1. Double-click `setup.bat`
2. Wait for installation to finish
3. Double-click `run.bat`
4. Open browser → **http://localhost:5000**

### Method 2 — Command Prompt

```cmd
cd ForgeNet-X
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
python app.py
```

### Method 3 — PowerShell

```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
.\setup.ps1
.\venv\Scripts\Activate.ps1
python app.py
```

---

## API Endpoints

| Method | Route                  | Description                                      |
|--------|------------------------|--------------------------------------------------|
| GET    | `/`                    | Homepage                                         |
| POST   | `/upload_handwriting`  | Upload, preprocess, segment, run origin analysis |
| POST   | `/generate_text`       | Generate watermarked synthetic sample, log event |
| POST   | `/upload_signature`    | Upload original + test signatures                |
| POST   | `/analyze_signature`   | Run forgery analysis, return JSON result         |
| POST   | `/download_report`     | Generate and stream forensic PDF report          |
| GET    | `/download/<f>/<n>`    | Download a generated file                        |

---

## Module Descriptions

### `utils/preprocessing.py`
Eight-stage adaptive binarisation pipeline:
- Bilateral filter → CLAHE → Adaptive Gaussian + Otsu AND-blend
- Morphological open/close → connected-component blob filter → resize

### `utils/segmentation.py`
Vertical projection valley character extraction (replaces contour-first approach):
- Handles `i`, `j`, `k` correctly — no fragile dot-merge heuristics
- Produces `manifest.json` + `line_map.json` for accurate atlas labelling

### `utils/provenance.py`
Origin analysis classifier:
- EXIF metadata extraction (camera, software, date)
- Heuristic scoring: background purity, sensor noise, software flags
- Output: `origin_label` (Likely Human / Possibly Synthetic / Likely Synthetic)

### `utils/handwriting.py`
Synthetic stress-test generator:
- Atlas-based character stitching with vertical jitter and kerning variation
- Bakes `[SYNTHETIC – ForgeNet-X]` watermark via PIL alpha-compositing

### `utils/signature_analysis.py`
Multi-metric forgery detection:
- SSIM (40%) + Hu Moments (25%) + Histogram (20%) + Pixel (15%)
- Composite score → Low / Moderate / High Risk + feature comparison table

### `utils/pdf_generator.py`
Professional forensic PDF via ReportLab:
- Cover page, origin analysis, preprocessing pipeline, segmentation atlas
- Signature metric breakdown, risk badge, feature delta table, verdict

---

## Similarity Score & Risk Levels

| Score | Risk Level | Meaning |
|-------|------------|---------|
| ≥ 80% | Low Risk   | Signatures closely match — likely genuine |
| 50–79% | Moderate Risk | Partial match — manual review recommended |
| < 50% | High Risk  | Signatures differ significantly — likely forged |

---

## Ethical Guardrails

| Guardrail | Implementation |
|-----------|----------------|
| Consent gate | Checkbox required before any synthetic generation |
| Automatic watermark | `[SYNTHETIC – ForgeNet-X]` baked into every output image |
| Generation audit log | `outputs/generation_log.jsonl` records timestamp + declared purpose |
| Origin analysis | Every uploaded sample is scored for synthetic vs. human origin |
| Full ethics statement | See `ETHICS.md` |

---

## Troubleshooting (Windows)

| Problem | Fix |
|---------|-----|
| `python` not recognised | Re-install Python, check "Add to PATH" |
| `pip install` fails on scikit-image | Run: `pip install --only-binary :all: scikit-image` |
| Port 5000 already in use | Change `port=5000` in `app.py` to `5001` |
| `cv2` import error | Run: `pip install opencv-python` |
| PowerShell script blocked | Run: `Set-ExecutionPolicy RemoteSigned -Scope CurrentUser` |
| Blank generated image | Upload a high-contrast, clean handwriting sheet |

---

## Post-MVP Improvements

### Stronger Detection
- **Siamese CNN** with contrastive loss for signature verification — replaces hand-crafted metrics
- **Synthetic-vs-real classifier** trained on IAM dataset — upgrades the heuristic provenance scorer
- **Writer identification** — determine whether two handwriting samples share the same author

### Generative Research Tools
- **ScrabbleGAN / HandwritingGAN** — conditional GAN for higher-fidelity synthetic samples
- **Stroke-level augmentation** — vary pressure, speed, and slant for more diverse stress-tests

### Deployment
```
Backend  : Gunicorn + Nginx
Container: Docker + docker-compose
Cloud    : AWS EC2 / GCP Cloud Run
Queue    : Celery + Redis (async report generation)
Auth     : JWT / OAuth2 — required before any generation feature is accessible
```
