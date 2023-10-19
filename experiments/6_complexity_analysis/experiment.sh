for trial in 1 2 3 4 5
do
    for C in 4
    do
        for N in 50 100 200
        do
            M=10
            identifier=datavalidation_C_${C}_N_${N}_trial_${trial}
            datafile=data/synthetic/output/${identifier}_data.csv
            simfile=data/synthetic/output/${identifier}_causal_sim.csv
            metafile=data/synthetic/output/${identifier}_task_metadata.csv
            sbatch experiments/6_complexity_analysis/experiment.sbatch ${datafile} ${simfile} ${metafile} ${identifier} ${trial} ${C} ${N} ${M}
        done
    done
done


for trial in 1 2 3 4 5
do
    for C in 4
    do
        for M in 5 10 20
        do
            N=200
            identifier=datavalidation_C_${C}_M_${M}_trial_${trial}
            datafile=data/synthetic/output/${identifier}_data.csv
            simfile=data/synthetic/output/${identifier}_causal_sim.csv
            metafile=data/synthetic/output/${identifier}_task_metadata.csv
            sbatch experiments/6_complexity_analysis/experiment.sbatch ${datafile} ${simfile} ${metafile} ${identifier} ${trial} ${C} ${N} ${M}
        done
    done
done
