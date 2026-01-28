# Natural Computing Algorithms - AI Agent Guidelines

## Project Context
Academic coursework implementing **V-Detector** (negative selection) and **Cuckoo Search** (continuous + discrete optimization). Optimized for accuracy over speed with procedural style. Final grade: 73% (1st class).

## Architecture Overview

**Three Independent Pipelines:**
1. **V-Detector (VD_\*.py)** - Anomaly detection using immune-inspired detectors
   - `VD_train.py` generates detectors → `outputs/detector_<timestamp>.txt`
   - `VD_test.py` validates against Self/non-Self data (defaults to most recent detector)
   - `VD_grid_search.py` tunes hyperparameters (c0, c1, threshold, alpha)

2. **Cuckoo Search - Continuous (`real_cuckoo.py`)**
   - Minimizes 4D sine-based function using Levy flights + local search
   - Hardcoded parameters (no config files)

3. **Cuckoo Search - Discrete (`discrete_cuckoo.py`)**
   - Graph partitioning via YAML configs in `configs/`
   - Outputs `Witness{A,B,C}_<timestamp>.txt` to `outputs/`
   - Key enhancement: FM-inspired Levy flight with neighbor swapping

**Shared Infrastructure:**
- `utils.py` - Common functions (Euclidean distance, file paths, timestamps)
- All scripts use `project_root` pattern: `os.path.dirname(os.path.dirname(__file__))`
- Timestamps prevent overwrites: `MmmDDHHMMSS` format (e.g., `Jan28143045`)

## Critical Workflows

**Package Management (REQUIRED):**
```bash
# Always use uv (not pip)
uv run python src/script.py     # Execute with managed environment
uv add <package>                # Add dependencies
uv sync                         # Sync from lockfile
```

**Run Pipelines:**
```bash
./run_vdetector.sh              # Train + test V-Detector
./run_cuckoo_all.sh             # Run discrete_cuckoo for all configs
uv run python src/VD_test.py    # Test most recent detector (auto-selects)
uv run python src/discrete_cuckoo.py configs/config_A.yaml
```

## Project-Specific Conventions

**File I/O Patterns:**
- Input data: `data/` (self_training.txt, self_testing.txt, non_self_testing.txt)
- Graph files: `GraphFiles/GPGraph{A,B,C}.txt` (adjacency matrices)
- All outputs: `outputs/` (detectors + witness files)
- Detector file format: Header lines then JSON-like arrays `[x1,x2,...,radius]`
- Witness file format: Metadata header + partition assignments (1-indexed)

**Parameter Conventions:**
- `c0/c1` (V-Detector): Coverage thresholds, typically 0.9999
- `alpha` (Cuckoo): Levy flight scaling (discrete uses decay: `alpha/sqrt(t+1)`)
- `p/q` (Cuckoo): Local search fraction / abandon fraction (0-1)
- `levy_strat`: `"R"` (random swaps) or `"FM"` (Fiduccia-Mattheyses heuristic)

**Code Style:**
- Procedural (no classes except `Vertex` for graph efficiency)
- Legacy function names preserved (e.g., `Euclidean_distance` wraps `euclidean_distance`)
- Progress bars: `tqdm` with `pbar.write()` for time limit messages
- No external algorithm libraries - implement from scratch

**Graph Partitioning Specifics:**
- Partition arrays are 1-indexed (values 1 to `sets_in_partition`)
- Fitness = number of edge conflicts (minimize to 0)
- Balanced partitions enforced: `min_size` vertices per set, with `rem` sets getting +1
- `Vertex` class pre-computes neighbor lists to avoid repeated adjacency lookups

**Testing Behavior:**
- `VD_test.py` auto-selects newest detector if no argument provided
- Detector files support multiple algorithms (NS, RV, VD) via `alg_code` parsing
- Time limits: V-Detector defaults to 13s training, Cuckoo configurable via YAML

## Integration Points

**Cross-Component Data:**
- `utils.location_of_witness_set()` builds full path to `outputs/Witness<digit>_<timestamp>.txt`
- `utils.location_of_GraphFiles()` resolves graph files from problem code + digit
- Detector files are self-describing (include algorithm code, dimensions, thresholds)

**YAML Config Structure (discrete_cuckoo.py):**
```yaml
graph_digit: "A"          # Which graph to solve
num_cyc: 100000           # Iteration limit
N: 50                     # Population size
alpha_decay: true         # Enable exploration→exploitation shift
levy_strat: "FM"          # Use enhanced Levy flight
timed: true               # Enforce max_time limit
```

**Common Pitfalls:**
- Don't use `pip` - strictly `uv` commands
- Paths are relative to project root, not `src/`
- V-Detector radius is stored as last element: `detector[-1]`
- Graph partitions must be balanced (check `get_set_sizes()` in discrete_cuckoo.py)
- Algorithm codes hardcoded: VD files write `"VD"`, cuckoo writes `"CS"`

## Key Files for Pattern Reference

- [src/utils.py](src/utils.py) - Path resolution + distance calculations
- [src/discrete_cuckoo.py](src/discrete_cuckoo.py) - YAML config loading, Vertex class, FM heuristic
- [src/VD_train.py](src/VD_train.py) - V-Detector algorithm with tqdm integration
- [configs/config_A.yaml](configs/config_A.yaml) - Complete parameter example
- [run_cuckoo_all.sh](run_cuckoo_all.sh) - Batch execution pattern
