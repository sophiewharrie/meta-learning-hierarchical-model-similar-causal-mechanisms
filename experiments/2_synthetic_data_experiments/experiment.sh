experiment_name=synthetic_experiments
skip_known=yes 

for trial in 1 2 3 4 5 6
do
    for C in 4 5 6
    do
        num_groups=${C}
        identifier=syndata_C_${C}_trial_${trial}
        datafile=data/synthetic/output/${identifier}_data.csv
        metafile=data/synthetic/output/${identifier}_task_metadata.csv
        intervfile=data/synthetic/output/${identifier}_intervmask.csv
        sbatch experiments/methods/run_method_comparison.sbatch ${datafile} ${metafile} ${intervfile} ${skip_known} ${experiment_name} ${identifier} ${num_groups}
    done
done

for trial in 1 2 3 4 5 6
do
    for eta_group in 0.3 0.5 0.7
    do
        num_groups=6
        identifier=syndata_eta_${eta_group}_trial_${trial}
        datafile=data/synthetic/output/${identifier}_data.csv
        metafile=data/synthetic/output/${identifier}_task_metadata.csv
        intervfile=data/synthetic/output/${identifier}_intervmask.csv
        sbatch experiments/methods/run_method_comparison.sbatch ${datafile} ${metafile} ${intervfile} ${skip_known} ${experiment_name} ${identifier} ${num_groups}
    done
done

for trial in 1 2 3 4 5 6
do
    for N_train in 100 150 200
    do
        num_groups=6
        identifier=syndata_N_${N_train}_trial_${trial}
        datafile=data/synthetic/output/${identifier}_data.csv
        metafile=data/synthetic/output/${identifier}_task_metadata.csv
        intervfile=data/synthetic/output/${identifier}_intervmask.csv
        sbatch experiments/methods/run_method_comparison.sbatch ${datafile} ${metafile} ${intervfile} ${skip_known} ${experiment_name} ${identifier} ${num_groups}
    done
done

for trial in 1 2 3 4 5 6
do
    for M in 20 40 60
    do
        num_groups=6
        M_train=$M
        M_test=$M
        identifier=syndata_M_${M}_trial_${trial}
        datafile=data/synthetic/output/${identifier}_data.csv
        metafile=data/synthetic/output/${identifier}_task_metadata.csv
        intervfile=data/synthetic/output/${identifier}_intervmask.csv
        sbatch experiments/methods/run_method_comparison.sbatch ${datafile} ${metafile} ${intervfile} ${skip_known} ${experiment_name} ${identifier} ${num_groups}
    done
done

for trial in 1 2 3 4 5 6
do
    for p_interv in 0.3 0.5 0.7
    do
        num_groups=6
        identifier=syndata_intervp_${p_interv}_trial_${trial}
        datafile=data/synthetic/output/${identifier}_data.csv
        metafile=data/synthetic/output/${identifier}_task_metadata.csv
        intervfile=data/synthetic/output/${identifier}_intervmask.csv
        sbatch experiments/methods/run_method_comparison.sbatch ${datafile} ${metafile} ${intervfile} ${skip_known} ${experiment_name} ${identifier} ${num_groups}
    done
done
