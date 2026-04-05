#!/bin/bash
# ═══════════════════════════════════════════════════════
# ForgeNet-X — One-shot setup & run script
# Usage:  bash setup.sh
# ═══════════════════════════════════════════════════════

set -e

echo "╔══════════════════════════════════════╗"
echo "║         ForgeNet-X  Setup            ║"
echo "╚══════════════════════════════════════╝"

# 1. Python virtual environment
echo ""
echo "→ Creating virtual environment..."
python3 -m venv venv
source venv/bin/activate

# 2. Install dependencies
echo "→ Installing Python dependencies..."
pip install --upgrade pip -q
pip install -r requirements.txt -q

# 3. Create runtime directories
echo "→ Creating runtime directories..."
mkdir -p uploads/handwriting uploads/signatures
mkdir -p outputs/generated   outputs/reports
mkdir -p models

echo ""
echo "✓ Setup complete!"
echo ""
echo "To run the application:"
echo "  source venv/bin/activate"
echo "  python app.py"
echo ""
echo "Then open:  http://localhost:5000"
