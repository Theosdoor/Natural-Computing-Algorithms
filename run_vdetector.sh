#!/bin/bash
#
# Train and test V-Detector
#
# Usage: ./run_vdetector.sh
#

set -e  # Exit on error

echo "==================================="
echo "V-Detector Training and Testing"
echo "==================================="

# Train the detector
echo ""
echo "Step 1: Training V-Detector..."
echo "-----------------------------------"
uv run python src/VD_train.py

# Test with the most recent detector (VD_test.py defaults to most recent)
echo ""
echo "Step 2: Testing V-Detector..."
echo "-----------------------------------"
uv run python src/VD_test.py

echo ""
echo "==================================="
echo "V-Detector pipeline complete!"
echo "==================================="
