experiment_name=real_experiments

# economic dataset
for trial in 1 2 3
do
    identifier=economics_trial_${trial}
    datafile=data/economics/econ_dataset${trial}.csv
    sbatch experiments/4_real_data_experiment/experiment_econ.sbatch ${datafile} ${identifier} ${experiment_name}
done

# medical dataset
for trial in 1 2 3
do
    identifier=medical_trial_${trial}
    datafile=data/medical/data/processed/ukbb_dataset${trial}.csv
    sbatch experiments/4_real_data_experiment/experiment_ukbb.sbatch ${datafile} ${identifier} ${experiment_name}
done

experiment_name=real_shift_experiments

# economic dataset
for trial in 1 2 3
do
    identifier=economicsshift_trial_${trial}
    datafile=data/economics/econshift_dataset${trial}.csv
    sbatch experiments/4_real_data_experiment/experiment_econ.sbatch ${datafile} ${identifier} ${experiment_name}
done
