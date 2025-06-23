#!/bin/bash -l

# Log files and working directory.
#SBATCH -o logs/%j_get_labels.out
#SBATCH -e logs/%j_get_labels.err
#SBATCH -D /raven/u/noka/VAEEG   # Set this to your project root directory

# Job name.
#SBATCH -J get_labels

# Resources.
#SBATCH --nodes=1
#SBATCH --tasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64000

# Email notifications.
#SBATCH --mail-type=end
#SBATCH --mail-user=kammnoel1@gmail.com

# Wall clock limit.
#SBATCH --time=00:30:00

# Load required system modules.
source /etc/profile.d/modules.sh
module purge

# Load Anaconda (no GPU modules needed for this script).
module load anaconda/3/2023.03

# Unset PYTHONPATH to prevent loading issues (this is important if you have local repositories that interfere).
unset PYTHONPATH

# Activate your virtual environment which contains all the necessary Python packages.
source ${HOME}/venvs/vaeeg/bin/activate

export PYTHONPATH=${SLURM_SUBMIT_DIR}/src:$PYTHONPATH

# Run the get_labels script with the provided parameters.
# Default parameters - modify as needed when submitting the job
srun python -m src.analysis.get_labels --latent /ptmp/noka/analysis/tusz/latent/alpha_z50_train.npy --mapping_csv mapping_train.csv --out_labels /ptmp/noka/labels/tusz/train_labels.npy
