#!/bin/bash -l

# Log files and working directory.
#SBATCH -o logs/%j_latent_t-sne.out
#SBATCH -e logs/%j_latent_t-sne.err
#SBATCH -D /raven/u/noka/VAEEG   # Set this to your project root directory

# Job name.
#SBATCH -J latent-tsne

# Resources.
#SBATCH --nodes=1
#SBATCH --tasks-per-node=1
#SBATCH --cpus-per-task=72
#SBATCH --mem=0

# Email notifications.
#SBATCH --mail-type=end
#SBATCH --mail-user=kammnoel1@gmail.com

# Wall clock limit.
#SBATCH --time=01:00:00

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

# Run your pipeline script with the provided YAML configuration and z_dim.
srun python -m src.analysis.latent_t_sne --input_latent /ptmp/noka/tusz_new/analysis/alpha_z50.npy --input_labels /ptmp/noka/tusz_new/analysis/alpha_z50_labels.npy --band "alpha" --output_emb /ptmp/noka/tusz_new/analysis/ --num_workers ${SLURM_CPUS_PER_TASK}
