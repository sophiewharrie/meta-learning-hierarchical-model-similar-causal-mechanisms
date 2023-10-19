experiment_name=ablation_experiments

# economic dataset
for trial in 1 2 3
do
    identifier=economicsabalation_trial_${trial}
    datafile=data/economics/econ_dataset${trial}.csv
    sbatch experiments/5_interventional_ablation_study/experiment_econ.sbatch ${datafile} ${identifier} ${experiment_name}
done

# medical dataset
for trial in 1 2 3
do
    identifier=medicalablation_trial_${trial}
    datafile=data/medical/data/processed/ukbb_dataset${trial}.csv
    sbatch experiments/5_interventional_ablation_study/experiment_ukbb.sbatch ${datafile} ${identifier} ${experiment_name}
done
