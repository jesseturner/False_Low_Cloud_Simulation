# False Low Cloud Simulation

## Overview
* Code used in Turner et al., 2025 (DOI: 10.1029/2025GL115366)
* Generates a model-driven simulation of an 11.2 um - 3.9 um satellite image
* Cloud-free simulation displays potential false low cloud signals as positives

## Requirements
* Code is written in Python (within and without Jupyter) and Bash for data collection

## Usage
* `sst_data/sst_data_download.sh` is used to collect the OISST data
* `model_data/model_data_download.sh` is used to collect the GFS data
* **`FLCI.py` is the main script which generates simulation results**
* `FLCI_composite.ipynb` is used to generate longer-term images based on the positive mean
* `water_vapor_abs.ipynb` is used to recreate the (pre-generated) mass extinction LUTs in `tables`

## Contact
* Corresponding author: jesse.turner@colostate.edu
