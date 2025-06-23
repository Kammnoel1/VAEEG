#!/bin/bash -l

# Log files and working directory.
#SBATCH -o logs/%j_avg_plot.out
#SBATCH -e logs/%j_avg_plot.err
#SBATCH -D /raven/u/noka/VAEEG   # Set this to your project root directory

# Job name.
#SBATCH -J avg_plot

# Resources.
#SBATCH --nodes=1
#SBATCH --tasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32000

# Email notifications.
#SBATCH --mail-type=end
#SBATCH --mail-user=kammnoel1@gmail.com

# Wall clock limit.
#SBATCH --time=00:30:00

# Load required system modules.
source /etc/profile.d/modules.sh
module purge

# Load Anaconda (matplotlib and numpy are included).
module load anaconda/3/2023.03

# Unset PYTHONPATH to prevent loading issues (this is important if you have local repositories that interfere).
unset PYTHONPATH

# Activate your virtual environment which contains all the necessary Python packages.
source ${HOME}/venvs/vaeeg/bin/activate

export PYTHONPATH=${SLURM_SUBMIT_DIR}/src:$PYTHONPATH

# Run the avg plotting script with the provided parameters.
srun python src/plotting/avg.py --data_path /ptmp/noka/new_data/tusz/train/theta.npy --labels_path /ptmp/noka/labels/tusz/train_labels.npy
