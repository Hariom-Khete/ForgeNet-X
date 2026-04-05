# ForgeNet-X
## AI Handwriting Imitation & Signature Forgery Analysis System

> Final Year AI/ML Project — Offline Flask Web Application (Windows)

---

## Overview

ForgeNet-X is a modular, fully offline web application that:
1. Accepts handwriting sample images and extracts writing style features
2. Generates synthetic handwriting from typed text using the extracted style
3. Compares original vs. test signature images using computer vision
4. Outputs a similarity score, risk classification, and a downloadable PDF report

---

## Project Structure

```
ForgeNet-X/
│
├── app.py                        ← Flask app & all routes (pathlib paths)
├── requirements.txt              ← Python dependencies
├── setup.bat                     ← Double-click to install (Windows CMD)
├── setup.ps1                     ← PowerShell alternative
├── run.bat                       ← Double-click to start server
│
├── templates/
│   └── index.html                ← Main UI (Jinja2 template)
│
├── static/
│   ├── css/style.css             ← Dark forensics UI
│   └── js/app.js                 ← Frontend logic (Fetch API + drag-drop)
│
├── uploads/
│   ├── handwriting/              ← Uploaded handwriting images
│   └── signatures/               ← Uploaded signature images
│
├── outputs/
│   ├── generated/                ← Generated handwriting images
│   └── reports/                  ← PDF report files
│
├── models/                       ← (Future) trained model weights
│
└── utils/
    ├── __init__.py
    ├── preprocessing.py          ← Grayscale, denoise, binarise, resize
    ├── segmentation.py           ← Contour-based character extraction
    ├── handwriting.py            ← Character atlas + text stitching
    ├── signature_analysis.py     ← SSIM, Hu, histogram, pixel metrics
    └── pdf_generator.py          ← ReportLab PDF report builder
```

---

## Windows Setup & Installation

### Requirements
- Windows 10 / 11
- Python 3.10 or newer — https://python.org/downloads
  - During install, check **"Add Python to PATH"** ✅
- No internet connection required after first install

---

### Method 1 — Double-click (easiest)

1. Double-click `setup.bat`
2. Wait for installation to finish
3. Double-click `run.bat`
4. Open browser → **http://localhost:5000**

---

### Method 2 — Command Prompt

```cmd
cd ForgeNet-X

:: Create & activate virtual environment
python -m venv venv
venv\Scripts\activate

:: Install dependencies
pip install -r requirements.txt

:: Run the app
python app.py
```

---

### Method 3 — PowerShell

```powershell
cd ForgeNet-X

# Allow script execution (run once)
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser

# Run setup
.\setup.ps1

# Then start the server
.\venv\Scripts\Activate.ps1
python app.py
```

---

## API Endpoints

| Method | Route                  | Description                              |
|--------|------------------------|------------------------------------------|
| GET    | `/`                    | Homepage                                 |
| POST   | `/upload_handwriting`  | Upload & preprocess handwriting image    |
| POST   | `/generate_text`       | Generate handwriting from typed text     |
| POST   | `/upload_signature`    | Upload original + test signatures        |
| POST   | `/analyze_signature`   | Run forgery analysis, return JSON result |
| POST   | `/download_report`     | Generate and stream PDF report           |
| GET    | `/download/<f>/<n>`    | Download a generated file                |

---

## Module Descriptions

### `utils/preprocessing.py`
Converts raw images to clean binary representation:
- Grayscale conversion → Gaussian blur → Otsu thresholding
- Morphological cleaning, aspect-preserving resize
- Deskew utility, style feature extraction (stroke width, slant, line spacing)

### `utils/segmentation.py`
Extracts individual characters:
- External contour detection with size filtering
- Reading-order sort (top→bottom, left→right)
- Character crops saved as PNGs with JSON manifest

### `utils/handwriting.py`
Generates handwriting from text:
- Character atlas from uploaded sample
- Per-character random vertical jitter for natural look
- Windows font fallback: Arial → Calibri → Courier → PIL default
- Explicit temp-dir cleanup (Windows GC is less aggressive than Linux)

### `utils/signature_analysis.py`
Multi-metric signature comparison:
- SSIM (40%) + Hu Moments (25%) + Histogram (20%) + Pixel (15%)
- Composite score → Low / Moderate / High Risk

### `utils/pdf_generator.py`
Professional PDF using ReportLab (no system fonts needed):
- Cover page, handwriting section, signature section
- Metric breakdown table, risk badge, feature delta table
- Page header/footer on every page

---

## Similarity Score & Risk Levels

| Score | Risk Level | Meaning |
|-------|-----------|---------|
| ≥ 80% | 🟢 Low Risk | Signatures closely match — likely genuine |
| 50–79% | 🟡 Moderate Risk | Partial match — manual review recommended |
| < 50% | 🔴 High Risk | Signatures differ significantly — likely forged |

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

### Deep Learning (Handwriting Realism)
Replace the stitching engine in `utils/handwriting.py` with:
- **ScrabbleGAN / HandwritingGAN** — conditional GAN, trains on IAM dataset
- **CNN + BiLSTM** — sequence model with attention for stroke prediction
- **ViT Transformer** — patch-based, best quality, highest GPU cost

### SaaS / Cloud Deployment
```
Backend    : Gunicorn + Nginx (Linux server)
Container  : Docker + docker-compose
Cloud      : AWS EC2 / GCP Cloud Run / Azure App Service
Queue      : Celery + Redis (async report generation)
Storage    : S3 / Azure Blob Storage
Auth       : JWT / OAuth2
DB         : PostgreSQL (user accounts, report history)
```

### GPU Acceleration (Windows)
```
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
# Replace numpy ops with CuPy for preprocessing
# Use torch.cuda.amp for mixed-precision inference
```

---

## Disclaimer

This tool is for academic and research purposes only. All results must be
reviewed by a qualified forensic document examiner before any legal,
financial, or investigative action is taken.
