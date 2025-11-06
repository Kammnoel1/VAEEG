#!/bin/bash -l

# Log files and working directory.
#SBATCH -o logs/%j_train_vae.out
#SBATCH -e logs/%j_train_vae.err
#SBATCH -D /raven/u/noka/VAEEG   # Set this to your project root directory

# Job name.
#SBATCH -J train_vae

# Resources.
#SBATCH --nodes=1
#SBATCH --constraint="gpu"
#SBATCH --tasks-per-node=1
#SBATCH --cpus-per-task=72
#SBATCH --mem=500000
#SBATCH --gres=gpu:a100:4

# Email notifications.
#SBATCH --mail-type=end
#SBATCH --mail-user=kammnoel1@gmail.com

# Wall clock limit.
#SBATCH --time=03:00:00

# Load required system modules.
source /etc/profile.d/modules.sh
module purge

# Load Anaconda and PyTorch (these modules should provide the correct PyTorch binaries).
module load anaconda/3/2023.03
module load pytorch/gpu-cuda-11.6/2.0.0
module load scikit-learn/1.2.2

# Unset PYTHONPATH to prevent loading issues (this is important if you have local repositories that interfere).
unset PYTHONPATH

# Activate your virtual environment which contains all the necessary Python packages.
source ${HOME}/venvs/vaeeg/bin/activate

export PYTHONPATH=${SLURM_SUBMIT_DIR}/src:$PYTHONPATH


# Set environment variables as needed (e.g., for GPU libraries).
export LD_LIBRARY_PATH=${CUDA_HOME}/extras/CUPTI/lib64:${LD_LIBRARY_PATH}

export MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n1)
export MASTER_PORT=29500


# Run your pipeline script with the provided YAML configuration and z_dim.
srun python -m src.train_vae --yaml_file train.yaml --z_dim 50 --band_name "high_beta"
