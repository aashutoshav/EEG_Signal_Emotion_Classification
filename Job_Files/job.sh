#!/bin/bash
#SBATCH -p gpu_v100_2
#SBATCH -N 1
#SBATCH -n 1
# #SBATCH -c 32
#SBATCH --mem 200000M
#SBATCH -t 0-11:59 # time (D-HH:MM)
#SBATCH --job-name="eeg_proc"
#SBATCH -o ./slurm/output%j.txt
#SBATCH -e ./slurm/error%j.txt
#SBATCH --gres=gpu:1
nvidia-smi
conda env list
spack load cuda/gypzm3r
spack load cudnn
#source activate alextf
#source $CONDA_PREFIX/etc/conda/activate.d/env_vars.sh
source activate torchpip
srun python3 ../SlitCNNModels/eeg.py