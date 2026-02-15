"""
IP-MS Analysis Pipeline
========================

A simple, reusable package for analyzing immunoprecipitation mass spectrometry data.

Main Functions:
--------------
prep_ip()       - Load and format raw proteomics data
qc_ip()         - Generate quality control plots
drop_samples()  - Remove problematic samples after QC
save_data()     - Save analysis data for later
load_data()     - Load saved analysis data

Sequential Workflow Example:
----------------------------
>>> # Step 1: Prepare data
>>> data = prep_ip('config/experiment.yaml')
>>> 
>>> # Step 2: Quality control
>>> data = load_data('results/data_after_prep.pkl')
>>> qc_ip(data)
>>> 
>>> # Step 3: Drop bad samples (optional)
>>> data = drop_samples(data)
>>> save_data(data, 'results/data_after_qc.pkl')
>>> 
>>> # Step 4: Normalize
>>> data = load_data('results/data_after_qc.pkl')
>>> data = norm_ip(data)
"""


# Import implemented functions
from .analysis import (
    prep_ip,
    qc_ip,
    drop_samples,
    norm_ip,
    stat_ip,
    viz_ip,
    venn_ip,
    boxplot_ip,
    save_data,
    load_data
)


__version__ = "0.1.0"
__author__ = "RNPC"


__all__ = [
    'prep_ip',
    'qc_ip',
    'drop_samples',
    'norm_ip',
    'stat_ip',
    'viz_ip',
    'venn_ip',
    'boxplot_ip',
    'save_data',
    'load_data'
]