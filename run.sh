#!/bin/bash
# JEBI Hackathon 2026 - Grupo 05
# Shovel Intelligence: Cycle Detection + Payload Estimation + Dashboard

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
INPUTS="$SCRIPT_DIR/inputs"
OUTPUTS="$SCRIPT_DIR/outputs"

echo "======================================================"
echo "  JEBI 2026 - Grupo 05 - Shovel Intelligence"
echo "======================================================"
echo "Inputs:"
ls "$INPUTS/" 2>/dev/null || echo "  (directorio vacio)"

mkdir -p "$OUTPUTS"

# Instalar dependencias
echo ""
echo "Instalando dependencias..."
pip install -q --no-warn-script-location \
    "opencv-python>=4.8.0" \
    "numpy>=1.24.0" \
    "pandas>=2.0.0" \
    "scipy>=1.11.0" || true

# Intentar instalar easyocr (OCR para numeros de camion)
# Si falla, el sistema usa clasificacion visual como fallback
pip install -q easyocr>=1.7.0 2>/dev/null || \
    echo "  easyocr no disponible - usando fallback visual"

# Correr pipeline principal
echo ""
echo "Corriendo pipeline..."
python "$SCRIPT_DIR/solution/pipeline.py" \
    --inputs  "$INPUTS" \
    --outputs "$OUTPUTS"

echo ""
echo "Outputs generados:"
ls -la "$OUTPUTS/"
echo "Done."
