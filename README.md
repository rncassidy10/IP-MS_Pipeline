# IP-MS Analysis Pipeline

A reusable Python package for analyzing immunoprecipitation mass spectrometry (IP-MS) data — from raw proteomics input to publication-ready figures and statistical results.

## Features

- **Flexible**: Works with any IP-MS experiment through YAML config files
- **Comprehensive**: Complete workflow from raw data to publication figures
- **Reproducible**: All parameters documented in config files
- **User-friendly**: Simple function calls, clear console output

## Installation

```bash
# Clone this repository
git clone https://github.com/rncassidy10/IP-MS_Pipeline.git
cd IP-MS_Pipeline

# Install as a package (editable mode for development)
pip install -e ".[dev]"
```

## Quick Start

1. Place your proteomics data in `data/raw/`
2. Create a config file in `config/` (see `config/example_config.yaml`)
3. Run analysis:

```python
from ipms import prep_ip, qc_ip, norm_ip, stat_ip, viz_ip, venn_ip, summary_ip

# Load and prepare data
data = prep_ip('config/your_experiment.yaml')

# Quality control
qc_ip(data)

# Normalize
data = norm_ip(data, method='log2')

# Statistical analysis
data = stat_ip(data)

# Visualizations
viz_ip(data)
venn_ip(data)

# Generate report
summary_ip(data)
```

## Workflow Steps

1. **prep_ip()** — Load and format raw data, remove contaminants
2. **qc_ip()** — Generate quality control plots (missing values, PCA, correlations)
3. **drop_samples()** — Remove problematic samples after QC review
4. **norm_ip()** — Normalize and impute missing values
5. **stat_ip()** — Statistical analysis (t-tests, fold changes, FDR correction)
6. **viz_ip()** — Create volcano plots and heatmaps
7. **venn_ip()** — Show overlap between significant protein sets
8. **boxplot_ip()** — Boxplots of enriched protein intensities
9. **summary_ip()** — Generate analysis report

## Project Structure

```
IP-MS_Pipeline/
├── config/              # Experiment configuration files
├── data/
│   ├── raw/             # Raw input data (not tracked by git)
│   └── processed/       # Processed data (not tracked by git)
├── ipms/                # Main package
│   ├── __init__.py      # Package exports
│   ├── prep.py          # Data preparation and contaminant removal
│   ├── qc.py            # Quality control plots and sample dropping
│   ├── normalization.py # Normalization and imputation
│   ├── statistics.py    # Statistical testing
│   ├── visualization.py # Volcano plots, heatmaps, Venn diagrams, boxplots
│   └── utils.py         # Helpers (config loading, save/load, directory setup)
├── notebooks/           # Example Jupyter notebooks
├── tests/               # Unit tests
├── pyproject.toml       # Package metadata and dependencies
├── requirements.txt     # Pinned dependencies (for environments without pip install)
└── results/             # Output figures and tables (not tracked by git)
```

## Requirements

- Python >= 3.8
- See `pyproject.toml` for package dependencies

## Running Tests

```bash
pytest
```

## Author

Richard Cassidy

## License

MIT License
