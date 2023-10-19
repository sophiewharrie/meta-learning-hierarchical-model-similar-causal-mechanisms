for trial in 1 2 3
do
    for C in 2 4 6 8
    do
        identifier=datavalidation_C_${C}_trial_${trial}
        datafile=data/synthetic/output/${identifier}_data.csv
        simfile=data/synthetic/output/${identifier}_causal_sim.csv
        metafile=data/synthetic/output/${identifier}_task_metadata.csv
        sbatch experiments/1_causal_heterogeneity/experiment.sbatch ${datafile} ${simfile} ${metafile} ${identifier} ${C}
    done
done
