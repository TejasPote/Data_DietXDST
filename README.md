##  ENEL 645 Final Project
This repository contains the code for the project Dynamic Sparse Training on a Data Diet done as a submission towards the course ENEL 645.


##  Directory Structure
```
Data_DietXDST/
│
├── data/                   # Scripts for data loading and pruning
├── models/                 # Model definitions and initialization
├── utils.py                # Helper utilities
│
├── dense_train.py          # Training script for dense model
├── sparse_train.py         # Training script for sparse model
│
├── dense_config.yaml       # Configuration for dense training
├── sparse_config.yaml      # Configuration for sparse training
│
├── requirements.txt        # Python dependencies

```
---

## Installation

To set up your environment using `pip` and the provided `requirements.txt` file:

```
# Create and activate a virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows use: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
``` 

o reproduce the results from the paper or experiment:

1. Train the Dense Models
Train dense models using `dense_train.py` with the `dense_config.yaml` configuration file for 10 different random seeds:
```
python dense_train.py --seed <SEED>
```

2. Train the Sparse Model
Once all dense models are trained, use their checkpoints to train the sparse model using `sparse_train.py` and `sparse_config.yaml`:

```
python sparse_train.py --seed <SEED> --param_sparsity <param_sparsity> --data_sparsity <data_sparsity>
```
