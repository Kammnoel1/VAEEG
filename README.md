# Learning Latent Representations of EEG Signals Using Autoencoders

A deep learning framework for analyzing EEG data using Variational Autoencoders (VAE) and Deterministic Autoencoders (DAE) to extract latent representations across different frequency bands for seizure detection.

## Overview

This project implements a VAE/DAE architecture designed specifically for EEG signal analysis. The model processes EEG data across five frequency bands (delta, theta, alpha, low beta, high beta) to learn compact latent representations that can be used for seizure detection and analysis.

## Features

- **Flexible Architecture**: Support for both VAE and DAE 
- **Multi-Band Processing**: Analyzes EEG across 5 frequency bands (delta: 1-4Hz, theta: 4-8Hz, alpha: 8-13Hz, low beta: 13-20Hz, high beta: 20-30Hz)
- **Distributed Training**: Multi-GPU support via PyTorch DDP
- **Comprehensive Analysis**: Notebooks for latent space visualization, reconstruction quality, and classification
- **HPC Integration**: SLURM batch scripts for cluster computing

## Project Structure

```
VAEEG/
├── cluster/              # SLURM batch scripts for HPC
│   ├── data_gen.sbatch
│   ├── split_data.sbatch
│   ├── train_vae.sbatch
│   └── test_vae.sbatch
├── configs/              # Configuration files
│   ├── train.yaml
│   └── test.yaml
├── scripts/              
│   ├── gen_data.py       # EDF to clip generation with labels
│   ├── split_dataset.py  # Train/test split with label tracking
│   ├── train_vae.py      # Model training
│   └── test_vae.py       # Model evaluation
├── src/
│   ├── model/
│   │   ├── net/          # Model architectures
│   │   │   ├── modelA.py         
│   │   │   ├── old_modelA.py     
│   │   │   └── layers.py        
│   │   └── opts/         # Training utilities
│   │       ├── dataset.py       
│   │       ├── ckpt.py          
│   │       ├── losses.py        
│   │       └── mig.py           
│   └── utils/
│       ├── io.py         
│       ├── labels.py     
│       └── interval.py   
├── notebooks/            # Analysis notebooks
│   ├── classifier.ipynb           
│   ├── latent_analysis.ipynb      
│   ├── latent_visualization.ipynb 
│   └── reconstruction_analysis.ipynb
└── weights/              # Trained model checkpoints
```

## Installation
```bash
# Clone repository 
git clone https://github.com/Kammnoel1/VAEEG.git
cd VAEEG

# Create virtual environment
python -m venv venvs/vaeeg
source venvs/vaeeg/bin/activate

# Install dependencies
pip install -r requirements.txt
```

## Preprocessing

### 1. Generate Clips from EDF Files

The [`gen_data.py`](scripts/gen_data.py) script processes raw EDF files from the TUSZ dataset:

```bash
python scripts/gen_data.py \
    --raw_base /path/to/edf/files \
    --out_base /path/to/output/clips \
    --labels_csv_path /path/to/labels/labels.csv \
    --n_jobs 10
```

**Output**: `.npy` files in `clips/` directory + `labels.csv`

### 2. Split Dataset

The [`split_dataset.py`](scripts/split_dataset.py) script splits clips into train/test sets:

```bash
python scripts/split_dataset.py \
    --base_dir /path/to/new_data \
    --labels_dir /path/to/labels \
    --labels_csv_path /path/to/labels/labels.csv \
    --ratio 0.1 \
    --n_jobs 10
```

**Output**:
- `train/<band>.npy`: Concatenated training data per band 
- `test/<band>.npy`: Concatenated test data per band 
- `labels/train.npy`: Training labels 
- `labels/test.npy`: Test labels 

## Training

The [`train_vae.py`](scripts/train_vae.py) script trains VAE/DAE models: 

```bash
python scripts/train_vae.py \
    --yaml_file configs/train.yaml \
    --z_dim 50 \
    --band_name "alpha"
```
Settings such as the training mode (VAE or DAE) and checkpoint save paths can be configured in the [`train.yaml`](configs/train.yaml) file.

## Evaluation

The [`test_vae.py`](scripts/test_vae.py) script evaluates VAE/DAE models:

```bash
python scripts/test_vae.py \
    --yaml_file configs/test.yaml \
    --z_dim 50 \
    --band_name "alpha"
```

The project includes batch scripts optimized for RAVEN cluster:

1. **Data Generation**: [`cluster/data_gen.sbatch`](cluster/data_gen.sbatch)
   - 72 CPUs, no GPU

2. **Data Splitting**: [`cluster/split_data.sbatch`](cluster/split_data.sbatch)
   - 72 CPUs, no GPU

3. **Training**: [`cluster/train_vae.sbatch`](cluster/train_vae.sbatch)
   - 4x A100 GPUs

4. **Testing**: [`cluster/test_vae.sbatch`](cluster/test_vae.sbatch)
   - 2x A100 GPUs