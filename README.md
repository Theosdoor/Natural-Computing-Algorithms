# Natural Computing Algorithms Coursework

Submitted as part of 3rd year **Natural Computing Algorithms** module (MSci Natural Sciences) to the Department of Computer Science at Durham University.

**Final Grade: 73% (1st Class)**

| Algorithm | Mark | Out of |
|-----------|------|--------|
| Negative Selection (NegSel) | 24.63 | 34 |
| Continuous Optimization (NatAlgReal) | 10.49 | 18 |
| Discrete Optimization (NatAlgDiscrete) | 37.27 | 48 |
| **TOTAL** | **72.39** | **100** |

## Project Overview

This repository implements three distinct natural computing algorithms for different optimization and detection tasks:

1. **Negative Selection (VDetector)** - Anomaly detection using immune-inspired principles
2. **Continuous Optimization (Cuckoo Search)** - Function minimization on a 4D test function
3. **Discrete Optimization (Cuckoo Search)** - Graph partitioning problem (3 problem instances)

All algorithms are implemented from scratch using only Python's standard library.

## Repository Structure

```
src/                             # Main implementation folder
├── VD_train.py                 # VDetector anomaly detection training
├── VD_test.py                  # Detector set validation
├── VD_grid_search.py           # Parameter tuning experiments
├── real_cuckoo.py              # Cuckoo Search continuous optimization
├── discrete_cuckoo.py          # Cuckoo Search graph partitioning
└── utils.py                    # Shared utility functions

configs/                         # Configuration files
├── config_A.yaml               # Graph A (160 vertices)
├── config_B.yaml               # Graph B (400 vertices)
└── config_C.yaml               # Graph C (800 vertices)

data/                            # Training and test datasets
├── self_training.txt           # Training data (Self)
├── self_testing.txt            # Test data (Self)
└── non_self_testing.txt        # Test data (non-Self)

outputs/                         # Generated detector sets and witness files
├── detector_<timestamp>.txt    # VDetector output files
└── Witness{A,B,C}_<timestamp>.txt  # Cuckoo Search results

GraphFiles/                      # Input data for graph problems
├── CLGraph{A,B,C}.txt          # Coloring problem instances
├── GCGraph{A,B,C}.txt          # Graph coloring instances
└── GPGraph{A,B,C}.txt          # Graph partitioning instances (used)

results/                         # Optimal solutions for discrete problems
├── WitnessA.txt                # Best partition for GPGraphA (159 conflicts)
├── WitnessB.txt                # Best partition for GPGraphB (268 conflicts)
└── WitnessC.txt                # Best partition for GPGraphC (212 conflicts)

Scripts:
├── run_vdetector.sh            # Train and test V-Detector
└── run_cuckoo_all.sh           # Run Cuckoo Search for all configs
```

## How to Run

### Negative Selection

**Training - Generate Detector Set:**
```bash
uv run python src/VD_train.py
# Or use the convenience script:
./run_vdetector.sh
# Output: outputs/detector_<timestamp>.txt
```

**Testing - Validate Detector Set:**
```bash
# Test most recent detector (default)
uv run python src/VD_test.py

# Or test a specific detector set
uv run python src/VD_test.py detector_Jan28200143.txt

# Or provide full path
uv run python src/VD_test.py outputs/detector_Jan28200143.txt
```

**Output metrics:**
- Detection rate: TP/(TP+FN) - percentage of non-Self correctly identified
- False alarm rate: FP/(FP+TN) - percentage of Self incorrectly flagged

### Continuous Optimization
```bash
uv run python src/real_cuckoo.py
# Output: Best minimum value and location printed to stdout
```

### Discrete Optimization

**Run All Configs:**
```bash
./run_cuckoo_all.sh
# Runs discrete_cuckoo.py for all configs in configs/
# Output: outputs/Witness{A,B,C}_<timestamp>.txt
```

**Run Individual Configs:**
```bash
# Run with configuration files for each problem
uv run python src/discrete_cuckoo.py configs/config_A.yaml  # Graph A (160 vertices)
uv run python src/discrete_cuckoo.py configs/config_B.yaml  # Graph B (400 vertices)
uv run python src/discrete_cuckoo.py configs/config_C.yaml  # Graph C (800 vertices)
# Output: outputs/Witness<digit>_<timestamp>.txt
# Features: tqdm progress bars, parameterized configs, cleaner code
```

## Algorithms & Results

### Negative Selection (VDetector)

**Feedback:** "Your detection rate was good or very good. Your false alarm rate was excellent or better. Your timings were perfect."

- **Detection Rate:** 74.83%
- **False Alarm Rate:** 6.37%
- **Detectors Generated:** 341 (from 600 intended)
- **Build Time:** 13.0 seconds
- **Key Parameters:** `threshold=0.027`, `c0=c1=0.9999`
- **Enhancement:** Alpha scaling (α=0.36) to fill gaps between adjacent detectors

**Key Insight:** Grid search with 10 evenly-spaced values per parameter dimension, repeated 3 times for averaging. Discovered that c1 (outer coverage) was significantly more important than c0, and that alpha scaling increased detection rate by ~7% while keeping false alarms under control.

### Continuous Optimization (Cuckoo Search)

**Feedback** 
> "Your minimum was excellent or better. Your timings were perfect."

- **Best Minimum Found:** -61.367 at [7.914, 7.756, 7.912, 7.905]
- **Elapsed Time:** 10.0 seconds
- **Parameters:** `num_cyc=24000, N=50, p=0.6, q=0.25, alpha=1.6, beta=1.5`
- **Function:** 4D sine-based function (minimization)

**Note:** Discretization and algorithmic enhancements were not pursued for this component.

### Discrete Optimization (Graph Partitioning)

**Feedback** 
> "Your enhancements were very good and showed aspects of originality and depth."

Performance on 3 problem instances:

| Graph | Vertices | Edges | Best Conflicts | Time | Quartile |
|-------|----------|-------|-----------------|------|----------|
| A | 160 | 425 | 159 | 1010.5s | 1st (solution), 3rd (runtime) |
| B | 400 | 867 | 268 | 1907.3s | 1st (solution), 2nd (runtime) |
| C | 800 | 920 | 212 | 3251.7s | 1st (solution), 2nd (runtime) |

**Key Enhancements:**
1. **Alpha Decay** - Levy flight scaling decreases over iterations for convergence
2. **Ranked Nest Generation** - New nests generated from best solutions with weighted probabilities
3. **FM-inspired Levy Flight** - After swapping vertices, check neighbors for further improvements (limited impact)
4. **Efficient Conflict Calculation** - Custom Vertex class stores neighbors to avoid full adjacency matrix checks

**Key Insight:** Exploration vs. exploitation trade-off managed through `p` (local search fraction) and `q` (abandon fraction). Larger graphs benefited from higher p and q values to explore more effectively despite increased runtime.

## Technical Details

For methodology, see the detailed reports in the project root:
- [NegSelReport.md](NegSelReport.md) - NegSel parameter tuning and alpha scaling enhancement
- [NatAlgReport.md](NatAlgReport.md) - Cuckoo Search discretization strategies and enhancements

## Implementation Notes

- **Language:** Python 3.7+
- **Dependencies:** numpy, tqdm, pyyaml (managed via `uv`)
- **Package Management:** Use `uv` commands for dependency management
  - `uv sync` - synchronize environment with lockfile
  - `uv run <script>` - execute scripts in managed environment
  - `uv add <package>` - add new dependencies
- **Procedural style:** No classes except where essential (e.g., Vertex class for graph representation)
- **File I/O:** Timestamps used to prevent accidental overwrites of output files
- **Performance:** Code optimized for accuracy rather than speed (except where noted)

