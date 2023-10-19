for trial in 1 2 3
do
    for C in 4
    do
        for N in 50 100 150 200 250
        do
            M=20
            identifier=datavalidation_C_${C}_N_${N}_trial_${trial}
            datafile=data/synthetic/output/${identifier}_data.csv
            simfile=data/synthetic/output/${identifier}_causal_sim.csv
            metafile=data/synthetic/output/${identifier}_task_metadata.csv
            sbatch experiments/3_N_m_analysis/experiment.sbatch ${datafile} ${simfile} ${metafile} ${identifier} ${trial} ${C} ${N} ${M}
        done
    done
done


for trial in 1 2 3
do
    for C in 4
    do
        for M in 4 8 16 32 64
        do
            N=200
            identifier=datavalidation_C_${C}_M_${M}_trial_${trial}
            datafile=data/synthetic/output/${identifier}_data.csv
            simfile=data/synthetic/output/${identifier}_causal_sim.csv
            metafile=data/synthetic/output/${identifier}_task_metadata.csv
            sbatch experiments/3_N_m_analysis/experiment.sbatch ${datafile} ${simfile} ${metafile} ${identifier} ${trial} ${C} ${N} ${M}
        done
    done
done
