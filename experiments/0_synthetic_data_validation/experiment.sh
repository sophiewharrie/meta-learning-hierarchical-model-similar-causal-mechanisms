# fixed across all datasets
ref_input=data_generator/example_edgefile.txt
seed=1234
N_train=200
N_val=30
N_test=30
M_train=10
M_test=10
sigma_ref=1.0
sigma_group=0.1
sigma_task=0.0001
sigma_noise=0.1
eta_group=0.6
eta_task=0.05

for trial in 1 2 3
do
    trial_seed=${seed}${trial}
    for C in 1 2 3 4 5 6 7 8
    do
        identifier=datavalidation_C_${C}_trial_${trial}
        outprefix=data/synthetic/output/${identifier}
        sbatch experiments/0_synthetic_data_validation/experiment.sbatch ${outprefix} ${ref_input} ${trial_seed} ${N_train} ${N_val} ${N_test} ${M_train} ${M_test} ${C} ${sigma_ref} ${sigma_group} ${sigma_task} ${sigma_noise} ${eta_group} ${eta_task}
    done
done
