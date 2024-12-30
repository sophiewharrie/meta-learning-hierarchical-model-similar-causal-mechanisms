#!/bin/bash
#SBATCH --job-name=main_icp_job
#SBATCH --time=72:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --output=slurm/main_icp_job_%j.out
#SBATCH --error=slurm/main_icp_job_%j.err

module load r/4.3.0

# Run Rscript for ICP
Rscript causal_distances/ICP/hiddenICP.R