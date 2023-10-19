for trial in 1 2 3
do
    for C in 4
    do
        identifier=datavalidation_C_${C}_trial_${trial}
        datafile=data/synthetic/output/${identifier}_data.csv
        input_simfile=data/synthetic/output/${identifier}_causal_sim.csv
        simfile=data/synthetic/output/${identifier}_alpha_causal_sim.csv
        metafile=data/synthetic/output/${identifier}_task_metadata.csv
        sbatch experiments/2_alpha_analysis/experiment.sbatch ${datafile} ${input_simfile} ${simfile} ${metafile} ${identifier} ${C}
    done
done
