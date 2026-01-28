# NatAlgDiscrete Refactoring Summary

## Changes Made

Successfully consolidated three nearly-identical scripts ([NatAlgDiscreteA.py](NatAlgDiscreteA.py), [NatAlgDiscreteB.py](NatAlgDiscreteB.py), [NatAlgDiscreteC.py](NatAlgDiscreteC.py)) into:

1. **One unified script**: [NatAlgDiscrete.py](NatAlgDiscrete.py)
2. **Three YAML config files**: [config_A.yaml](config_A.yaml), [config_B.yaml](config_B.yaml), [config_C.yaml](config_C.yaml)

## Key Improvements

### 1. Configuration Management
- Parameters now externalized to YAML files
- Easy to modify/tune without editing code
- Clear documentation of what each parameter does

### 2. Progress Tracking
- Replaced periodic print statements with `tqdm` progress bars
- Shows real-time:
  - Current cycle number
  - Best conflicts found
  - Time remaining
- Much better user experience during long runs

### 3. Code Quality
- Single source of truth - no duplicate code
- Easier to maintain and debug
- Clear separation of configuration and logic

## Configuration Differences

The only differences between problem variants are algorithm parameters:

| Parameter | A | B | C | Description |
|-----------|---|---|---|-------------|
| `graph_digit` | "A" | "B" | "C" | Which graph to solve |
| `num_cyc` | 100,000 | 30,000 | 25,000 | Number of optimization cycles |
| `N` | 50 | 40 | 30 | Population size (nests) |
| `p` | 0.5 | 0.7 | 0.7 | Fraction of local flights |
| `q` | 0.25 | 0.25 | 0.4 | Fraction of nests to abandon |

All other parameters (alpha, beta, enhancement strategies) remain identical.

## Usage

```bash
# New unified approach
uv run python nchw73/NatAlgDiscrete.py nchw73/config_A.yaml
uv run python nchw73/NatAlgDiscrete.py nchw73/config_B.yaml
uv run python nchw73/NatAlgDiscrete.py nchw73/config_C.yaml

# Legacy scripts still available
uv run python nchw73/NatAlgDiscreteA.py
uv run python nchw73/NatAlgDiscreteB.py
uv run python nchw73/NatAlgDiscreteC.py
```

## Dependencies Added

- `pyyaml>=6.0` - For YAML configuration parsing
- `tqdm>=4.66.0` - For progress bars

Both automatically installed via `uv add` and tracked in `pyproject.toml`.

## Backward Compatibility

Original scripts remain untouched for reference/comparison. The unified script produces identical results while offering:
- Better visibility during execution
- Easier parameter experimentation
- Cleaner codebase
