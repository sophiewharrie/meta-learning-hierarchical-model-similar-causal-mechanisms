experiment_name=real_experiments
skip_known=yes
num_groups="2 4"

# medical dataset
for trial in 0 1 2 3 4 5
do
    identifier=medical_trial_${trial}
    datafile=data/medical/data/processed/medical_dataset${trial}.csv
    metafile='ignore'
    intervfile=data/medical/data/processed/medical_dataset${trial}_mask.csv
    sbatch experiments/methods/run_method_comparison.sbatch ${datafile} ${metafile} ${intervfile} ${skip_known} ${experiment_name} ${identifier} ${num_groups}
done

# covid dataset
for trial in 0 1 2 3 4 5
do
    identifier=covid_trial_${trial}
    datafile=data/covid/covid_dataset${trial}.csv
    metafile='ignore'
    intervfile=data/covid/covid_dataset${trial}_mask.csv
    sbatch experiments/methods/run_method_comparison.sbatch ${datafile} ${metafile} ${intervfile} ${skip_known} ${experiment_name} ${identifier} ${num_groups}
done

# cognition dataset
for trial in 0 1 2 3 4 5
do
    identifier=cognition_trial_${trial}
    datafile=data/cognition/psych_dataset${trial}.csv
    metafile='ignore'
    intervfile=data/cognition/psych_dataset${trial}_mask.csv
    sbatch experiments/methods/run_method_comparison.sbatch ${datafile} ${metafile} ${intervfile} ${skip_known} ${experiment_name} ${identifier} ${num_groups}
done

