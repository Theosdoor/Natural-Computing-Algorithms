#!/bin/bash
#
# Run Discrete Cuckoo Search for all configuration files
#
# Usage: ./run_cuckoo_all.sh
#

set -e  # Exit on error

echo "==================================="
echo "Running Cuckoo Search for All Configs"
echo "==================================="

# Get all yaml config files
CONFIG_DIR="configs"

if [ ! -d "$CONFIG_DIR" ]; then
    echo "Error: configs/ directory not found"
    exit 1
fi

# Count config files
CONFIG_COUNT=$(ls -1 "$CONFIG_DIR"/*.yaml 2>/dev/null | wc -l)

if [ "$CONFIG_COUNT" -eq 0 ]; then
    echo "Error: No .yaml config files found in $CONFIG_DIR/"
    exit 1
fi

echo "Found $CONFIG_COUNT configuration file(s)"
echo ""

# Run for each config file
for config in "$CONFIG_DIR"/*.yaml; do
    if [ -f "$config" ]; then
        echo "==================================="
        echo "Running: $(basename "$config")"
        echo "==================================="
        uv run python src/discrete_cuckoo.py "$config"
        echo ""
    fi
done

echo "==================================="
echo "All configurations complete!"
echo "==================================="
echo ""
echo "Witness files saved to outputs/"
ls -lt outputs/Witness*.txt | head -n "$CONFIG_COUNT"
