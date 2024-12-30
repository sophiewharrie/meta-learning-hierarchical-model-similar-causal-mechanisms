#!/bin/bash
#SBATCH --job-name=main_scm_job
#SBATCH --time=24:00:00
#SBATCH --gres=gpu
#SBATCH --constraint='ampere'
#SBATCH --mem=64G
#SBATCH --output=slurm/main_scm_job_%j.out
#SBATCH --error=slurm/main_scm_job_%j.err

module load scicomp-python-env

XLA_PYTHON_CLIENT_PREALLOCATE=false
XLA_PYTHON_CLIENT_MEM_FRACTION=.70

# Run Python script for SCM
python3 ./causal_distances/SCM/scm.py --outprefix /path/to/outdir/prefix