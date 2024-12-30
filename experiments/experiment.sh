#!/bin/bash

sbatch experiments/run_bnn_baseline.sbatch experiments/config.sh
sbatch experiments/run_two_level_hierarchical.sbatch experiments/config.sh
sbatch experiments/run_three_level_hierarchical.sbatch experiments/config.sh BASELINE
sbatch experiments/run_three_level_hierarchical.sbatch experiments/config.sh MR
sbatch experiments/run_three_level_hierarchical.sbatch experiments/config.sh SCM
sbatch experiments/run_three_level_hierarchical.sbatch experiments/config.sh ICP