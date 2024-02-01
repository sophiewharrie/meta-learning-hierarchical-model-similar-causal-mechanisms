# fixed across all datasets
seed=1234

# causal heterogeneity experiment 1
# vary the number of causal groups
N_train=150
N_val=30
N_test=30
M_train=40
M_test=40
sigma_ref=1.0
sigma_group=0.02
sigma_task=0.0001
sigma_noise=0.1
eta_group=0.5
eta_task=0.001
p_interv=0.3

for trial in 1 2 3 4 5 6
do
    trial_seed=${seed}${trial}
    for C in 4 5 6
    do
        identifier=syndata_C_${C}_trial_${trial}
        outprefix=data/synthetic/output/${identifier}
        sbatch experiments/methods/generate_data.sbatch ${outprefix} ${trial_seed} ${N_train} ${N_val} ${N_test} ${M_train} ${M_test} ${C} ${sigma_ref} ${sigma_group} ${sigma_task} ${sigma_noise} ${eta_group} ${eta_task} ${p_interv}
    done
done

# causal heterogeneity experiment 2
# fix group-level variation (eta_group)
C=6
N_train=150
N_val=30
N_test=30
M_train=40
M_test=40
sigma_ref=1.0
sigma_group=0.02
sigma_task=0.0001
sigma_noise=0.1
eta_task=0.001
p_interv=0.3

for trial in 1 2 3 4 5 6
do
    trial_seed=${seed}${trial}
    for eta_group in 0.3 0.5 0.7
    do
        identifier=syndata_eta_${eta_group}_trial_${trial}
        outprefix=data/synthetic/output/${identifier}
        sbatch experiments/methods/generate_data.sbatch ${outprefix} ${trial_seed} ${N_train} ${N_val} ${N_test} ${M_train} ${M_test} ${C} ${sigma_ref} ${sigma_group} ${sigma_task} ${sigma_noise} ${eta_group} ${eta_task} ${p_interv}
    done
done

# amount of data 1
# vary the number of training tasks N
C=6
N_val=30
N_test=30
M_train=40
M_test=40
sigma_ref=1.0
sigma_group=0.02
sigma_task=0.0001
sigma_noise=0.1
eta_group=0.5
eta_task=0.001
p_interv=0.3

for trial in 1 2 3 4 5 6
do
    trial_seed=${seed}${trial}
    for N_train in 100 150 200
    do
        identifier=syndata_N_${N_train}_trial_${trial}
        outprefix=data/synthetic/output/${identifier}
        sbatch experiments/methods/generate_data.sbatch ${outprefix} ${trial_seed} ${N_train} ${N_val} ${N_test} ${M_train} ${M_test} ${C} ${sigma_ref} ${sigma_group} ${sigma_task} ${sigma_noise} ${eta_group} ${eta_task} ${p_interv}
    done
done

# amount of data 2
# vary the number of samples per task M
C=6
N_train=150
N_val=30
N_test=30
sigma_ref=1.0
sigma_group=0.02
sigma_task=0.0001
sigma_noise=0.1
eta_group=0.5
eta_task=0.001
p_interv=0.3

for trial in 1 2 3 4 5 6
do
    trial_seed=${seed}${trial}
    for M in 20 40 60
    do
        M_train=$M
        M_test=$M
        identifier=syndata_M_${M}_trial_${trial}
        outprefix=data/synthetic/output/${identifier}
        sbatch experiments/methods/generate_data.sbatch ${outprefix} ${trial_seed} ${N_train} ${N_val} ${N_test} ${M_train} ${M_test} ${C} ${sigma_ref} ${sigma_group} ${sigma_task} ${sigma_noise} ${eta_group} ${eta_task} ${p_interv}
    done
done

# ratio of observational to interventional data
# higher p means more observational data
C=6
N_train=150
N_val=30
N_test=30
M_train=40
M_test=40
sigma_ref=1.0
sigma_group=0.02
sigma_task=0.0001
sigma_noise=0.1
eta_group=0.5
eta_task=0.001

for trial in 1 2 3 4 5 6
do
    trial_seed=${seed}${trial}
    for p_interv in 0.3 0.5 0.7
    do
        identifier=syndata_intervp_${p_interv}_trial_${trial}
        outprefix=data/synthetic/output/${identifier}
        sbatch experiments/methods/generate_data.sbatch ${outprefix} ${trial_seed} ${N_train} ${N_val} ${N_test} ${M_train} ${M_test} ${C} ${sigma_ref} ${sigma_group} ${sigma_task} ${sigma_noise} ${eta_group} ${eta_task} ${p_interv}
    done
done
