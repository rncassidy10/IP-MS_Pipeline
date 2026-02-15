# IP-MS Analysis Pipeline

A simple, reusable Python package for analyzing immunoprecipitation mass spectrometry (IP-MS) data.

## Features

- **Flexible**: Works with any IP-MS experiment through config files
- **Comprehensive**: Complete workflow from raw data to publication figures
- **Reproducible**: All parameters documented in config files
- **User-friendly**: Simple function calls, clear outputs

## Installation

```bash
# Clone this repository
git clone https://github.com/yourusername/ipms-pipeline.git
cd ipms-pipeline

# Install dependencies
pip install -r requirements.txt
```

## Quick Start

1. Place your proteomics data in `data/raw/`
2. Create a config file in `config/` (see `config/example_config.yaml`)
3. Run analysis:

```python
from ipms.analysis import prep_ip, qc_ip, norm_ip, stat_ip, viz_ip, summary_ip

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

# Generate report
summary_ip(data)
```

## Workflow Steps

1. **prep_ip()** - Load and format raw data
2. **qc_ip()** - Generate quality control plots
3. **norm_ip()** - Normalize and impute missing values
4. **stat_ip()** - Statistical analysis vs control
5. **viz_ip()** - Create volcano plots, heatmaps, Venn diagrams
6. **venn_ip()** - Show overlap between groups
6. **summary_ip()** - Generate analysis report

## Project Structure

```
ipms-pipeline/
├── config/              # Experiment configuration files
├── data/
│   ├── raw/            # Raw input data (not tracked by git)
│   └── processed/      # Processed data (not tracked by git)
├── ipms/               # Main package
│   ├── __init__.py
│   └── analysis.py     # Core analysis functions
├── notebooks/          # Example Jupyter notebooks
└── results/            # Output figures and tables (not tracked by git)
```

## Requirements

- Python >= 3.8
- See `requirements.txt` for package dependencies

## Author

Your Name

## License

MIT License
