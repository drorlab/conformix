# ConforMix Benchmarking Suite

A comprehensive benchmarking framework for evaluating ConforMix on standard protein conformational datasets.

## Overview

This module provides automated tools to:
- Run ConforMix on benchmark datasets (domain motion, fold-switching, cryptic pockets)
- Compute evaluation metrics (RMSD, TM-score, conformational coverage)
- Generate professional Markdown and HTML reports

## Installation

The benchmarking suite is included in the ConforMix repository. Ensure you have the main package installed:

```bash
pip install ./conformix_boltz
```

Then install benchmark dependencies:

```bash
pip install pandas pyyaml numpy mdtraj pytest
```

## Quick Start

### Run a Benchmark

```bash
# Run on domain motion dataset
python -m benchmarks.run_benchmark --config benchmarks/configs/domainmotion.yaml

# Run on specific proteins only
python -m benchmarks.run_benchmark \
    --config benchmarks/configs/domainmotion.yaml \
    --proteins P0205 P69441

# Dry run (see what would be executed)
python -m benchmarks.run_benchmark \
    --config benchmarks/configs/domainmotion.yaml \
    --dry-run
```

### Compute Metrics

```bash
python -m benchmarks.evaluate_metrics \
    --results benchmark_results/domainmotion/all_results.json
```

### Generate Reports

```bash
python -m benchmarks.generate_report \
    --metrics benchmark_results/domainmotion/metrics.json
```

## Available Datasets

| Dataset | Config File | Proteins | Description |
|---------|-------------|----------|-------------|
| Domain Motion | `configs/domainmotion.yaml` | 38 | Large-scale domain movements |
| Fold-Switching | `configs/foldswitching.yaml` | 15 | Proteins that change secondary structure |
| Cryptic Pockets | `configs/crypticpockets.yaml` | 34 | Hidden binding sites |
| Membrane Transporters | `configs/membranetransporters.yaml` | - | Conformational changes in transport |

## Configuration Options

Create a YAML configuration file:

```yaml
dataset_name: "my_dataset"
csv_path: "datasets/my_dataset.csv"
output_dir: "benchmark_results/my_dataset"

# Sampling parameters
num_twist_targets: 5        # Number of RMSD targets
samples_per_target: 2       # Samples per target
twist_strength: 15.0        # Twist potential strength
structured_regions_only: true

# Execution settings
timeout_seconds: 3600       # Timeout per protein
skip_existing: true         # Skip already processed
```

## Metrics Computed

| Metric | Description |
|--------|-------------|
| **Min RMSD to Alt** | Minimum RMSD from any sample to alternate structure |
| **Mean RMSD to Alt** | Average RMSD across all samples |
| **Conformational Coverage** | How close best sample is to known alternate |
| **RMSD Diversity** | Average pairwise RMSD between samples |

## Output Structure

```
benchmark_results/
└── domainmotion/
    ├── config.json           # Configuration used
    ├── all_results.json      # Raw benchmark results
    ├── metrics.json          # Computed metrics
    ├── report.md             # Markdown report
    ├── report.html           # HTML report
    ├── .cache/               # Downloaded structures
    └── P0205/                # Per-protein outputs
        ├── result.json
        └── samples.cif
```

## Running Tests

```bash
# Run all tests
pytest benchmarks/tests/ -v

# Run specific test class
pytest benchmarks/tests/test_benchmark.py::TestRMSDComputation -v
```

## API Usage

```python
from benchmarks import run_benchmark, compute_metrics, generate_report
from benchmarks.run_benchmark import BenchmarkConfig

# Load configuration
config = BenchmarkConfig.from_yaml("benchmarks/configs/domainmotion.yaml")

# Run benchmark
results = run_benchmark(config)

# Compute metrics
from benchmarks.evaluate_metrics import compute_all_metrics
metrics = compute_all_metrics(config.output_dir / "all_results.json")

# Generate reports
generate_report(config.output_dir / "metrics.json")
```

## Adding Custom Datasets

1. Create a CSV file with columns:
   - `system_id`: Unique identifier
   - `pdb1`: First PDB ID with chain (e.g., `1AKE_A`)
   - `pdb2`: Second PDB ID with chain (alternate state)
   - `RMSD`: Ground truth RMSD between states
   - `TM-score`: (optional) Structural similarity

2. Create a YAML config pointing to your CSV

3. Run the benchmark

## Contributing

When adding new metrics or features:
1. Add implementation to appropriate module
2. Add tests to `tests/test_benchmark.py`
3. Update this README

## License

MIT License - same as ConforMix main repository.
