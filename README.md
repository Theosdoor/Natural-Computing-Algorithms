# Natural Computing Algorithms Coursework

Submitted as part of 3rd year **Natural Computing Algorithms** module (MSci Natural Sciences) to the Department of Computer Science at Durham University.

**Final Grade: 73% (1st Class)**

| Algorithm | My Mark | Class Average |
|-----------|---------|---------------|
| Negative Selection (NegSel) | 72.4% | 66% |
| Continuous Optimization (NatAlgReal) | 58.2% | 69% |
| Discrete Optimization (NatAlgDiscrete) | 77.6% | 66% |
| **TOTAL** | **73%** | **67%** |

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

## Detailed Methodology

### Negative Selection (V-Detector)

#### Parameter Tuning Methodology

For parameter tuning, I used a grid search approach to explore various parameter combinations. This involved exploring wide intervals for each parameter at first, and then systematically narrowing them to investigate promising results. Within every interval, I sampled 10 evenly spaced values. Each parameter combination was repeated 3 times, and the results were averaged to account for randomness.

My first grid search checked 10⁴ combinations of 10 equally spaced values for each parameter within the following intervals:
- **c0 and c1** – [0.75, 0.95]. Note that the possible values for c0 and c1 were the same, but every c0 was tried against every c1 so we get 10² combinations from this interval.
- **Threshold** – [0.01, 0.1]
- **N** – [100, 1000]

**Initial Findings:** This was surprisingly quick, because most (c0, c1) pairs where one was below 0.9 generated very low detector counts. If the threshold was small enough (less than 0.04) then there was always at least 1 detector produced in my tests for any (c0, c1) pair.

The general trend was that (c0, c1) pairs closer to (1, 1) produced more detectors and gave better results. Somewhat surprising however, was that **c1 was much more significant than c0** for the output quality:
- A typical **(high c0, low c1)** pair like (0.975, 0.85) produced 8 detectors (with low DR and FAR)
- A typical **(low c0, high c1)** pair like (0.825, 0.975) gave 36 detectors with DR = 22.32%

Despite this observation, (high c0, high c1) pairs continued to give the most consistently high-quality results. Narrowing the interval for possible (c0, c1) pairs and repeating the grid search led me to finally choose **c0=c1=0.9999**.

**Threshold Optimization:** As for thresholds on [0.005, 0.05], DR was > 70% on [0.005, 0.035) and FAR was < 10% on [0.025, 0.05]. This suggested [0.025, 0.035) as the optimal range. By repeatedly narrowing this interval, I settled on **0.027 as my threshold**, which gave DR = 78.38% and FAR = 6.07%.

**Population Size (N):** Given my threshold = 0.027, I set **N = 600** since it took longer than 13 seconds to generate more than this, at which point the runtime is too long.

#### Enhancement: Alpha Scaling

When one detector is placed next to another, they touch at most once, which is only when the distance between them = threshold. In this case, there is a gap of potentially non-self between the curves of the detectors, where no new detector can fit. I decided to slightly extend the radius of detectors at the moment they are added to the detector set to cover this gap.

I added an **alpha scaling factor**, which is multiplied by the threshold and added to the detector radius. For c0=c1=0.9999, threshold=0.27, N=600:
- **Alpha = 0** (vanilla): Averaged DR = 73.77 and FAR = 4.4 across 10 detector sets
- **Alpha = 0.36**: Averaged DR = 81.05 and FAR = 8.95, as well as ~70 fewer detectors

---

### Continuous Optimization (Cuckoo Search)

The best minimum I achieved was at `[7.922588886680929, 7.942046903938513, 7.930403753699924, 7.926600109343327]` with a value of **-62.15358247837975**. This took 9.9 seconds with parameters: `num_cyc = 24000, N = 50, p = 0.6, q = 0.25, alpha = 1.6, beta = 1.5`.

---

### Discrete Optimization (Graph Partitioning)

#### Discretization Strategy

To discretize the algorithm, I created discrete Levy and local flight functions. To accurately represent the philosophy behind cuckoo search, I ensured that:
- **Local flights** from a nest only minimally disrupted its associated partition
- **Levy flights** from a nest would generally be small changes with the chance of a large disruption

I defined:
- **Local flight:** Randomly swapping two vertices in different partitions
- **Levy flight:** Performing local swaps M times, where M is the integer part of `alpha * Levy_step`

**Initial Results:** Without any enhancements, within a minute:
- **Graph A:** Halved conflicts (298 → 173)
- **Graph B:** Halved conflicts (626 → 382)
- **Graph C:** Reduced by about a third (779 → 509)

This suggests the discretization works, since the convergence was fast as expected in cuckoo search. However, this speed is also partly due to my method of calculating conflicts which avoided checking the entire adjacency matrix and used a custom Vertex class to store neighbors instead.

#### Parameter Tuning

**Population Size (N):** Number of nests had a significant impact on runtime and convergence time. This highlights the **exploration vs. exploitation trade-off** in cuckoo search, where larger N encourages greater exploration and smaller N encourages exploitation. However, larger N significantly affected runtime for larger graphs.

**Local Search and Abandon Fractions (p, q):** A low p and high q resulted in very fast convergence, whereas high p and low q caused the opposite but allowed greater exploration. I found that for larger graphs, a larger p and q was more useful:
- **Graph A:** Generally converged to ~170 for any p in [0.4, 0.9] and q in [0.1, 0.9]
- **Graph C:** Searches with larger p and q converged substantially quicker after 30+ seconds

#### Enhancements

**1. Alpha Decay and Levy Flights from New Nests** ([Source](https://doi.org/10.1016/j.chaos.2011.06.004))

When generating a new nest to replace an abandoned one, I do a Levy flight from it after generation. I introduce a parameter `alphat`, which decreases as t increases. This encouraged exploration when generating new nests, but the decaying alpha meant that these Levy flights would eventually be ineffective, improving convergence speed.

**2. Ranked Nest Generation**

Inspired by ranked ACO, I select w (user-chosen parameter) of the best nests found in a given cycle. When replacing abandoned nests, the newly generated nest is a Levy flight from one of these best nests, selected with probabilities proportional to their fitness. I used `p_ranked = 0.8` and `w = 6` for good exploitation while allowing 20% of newly generated nests to still be completely random to help escape local optima. This vastly improved convergence time. In graph C, solutions fewer than 300 conflicts were consistently found within 1 minute.

**3. Fiduccia-Mattheyses (FM) Inspired Levy Flight**

Inspired by the FM heuristic, this Levy flight swaps 2 random vertices, then checks their neighbors and swaps them if it reduces conflicts. A `neighbour_limit` parameter limits the number of neighbors checked. Unfortunately, this didn't have a significant impact compared to the vanilla operator.

**4. Greedy Nest Generation (Unsuccessful)**

I attempted to seed the initial population with 'greedy' partitions. Each vertex is added to the smallest partition that doesn't conflict. Unfortunately, this made the initial population worse (initial best of 880 vs 780 for random). They gradually converged to the same number after several cycles, so this enhancement was not useful.

---

## Technical Details

- **Language:** Python 3.7+
- **Dependencies:** numpy, tqdm, pyyaml (managed via `uv`)
- **Package Management:** Use `uv` commands for dependency management
  - `uv sync` - synchronize environment with lockfile
  - `uv run <script>` - execute scripts in managed environment
  - `uv add <package>` - add new dependencies
- **Procedural style:** No classes except where essential (e.g., Vertex class for graph representation)
- **File I/O:** Timestamps used to prevent accidental overwrites of output files
- **Performance:** Code optimized for accuracy rather than speed (except where noted)

