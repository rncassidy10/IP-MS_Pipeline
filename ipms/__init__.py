"""
IP-MS Analysis Pipeline
========================

A reusable Python package for analyzing immunoprecipitation mass spectrometry data.

Main Functions
--------------
prep_ip()       - Load and format raw proteomics data
qc_ip()         - Generate quality control plots
drop_samples()  - Remove problematic samples after QC
norm_ip()       - Normalize and impute missing values
stat_ip()       - Statistical analysis (t-tests, fold changes)
viz_ip()        - Volcano plots and heatmaps
venn_ip()       - Venn diagrams of significant protein overlap
boxplot_ip()    - Boxplots of enriched protein intensities
string_ip()     - STRING network analysis of significant proteins
summary_ip()    - Generate analysis report
save_data()     - Save analysis data for later
load_data()     - Load saved analysis data

Example Workflow
----------------
>>> from ipms import prep_ip, qc_ip, norm_ip, stat_ip, viz_ip
>>>
>>> data = prep_ip('config/experiment.yaml')
>>> qc_ip(data)
>>> data = norm_ip(data)
>>> data = stat_ip(data)
>>> viz_ip(data)
"""

from .prep import prep_ip
from .qc import qc_ip, drop_samples
from .normalization import norm_ip
from .statistics import stat_ip
from .visualization import viz_ip, boxplot_ip, summary_ip
from .utils import save_data, load_data
from .venn_ip import venn_ip
from .string_ip import string_ip


__version__ = "0.1.0"
__author__ = "Richard Cassidy"

__all__ = [
    'prep_ip',
    'qc_ip',
    'drop_samples',
    'norm_ip',
    'stat_ip',
    'viz_ip',
    'venn_ip',
    'string_ip',
    'boxplot_ip',
    'summary_ip',
    'save_data',
    'load_data',
]
