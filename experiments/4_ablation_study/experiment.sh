# C_6 corresponds to scenario 1 (standard causal heterogeneity setting)
# M_20 corresponds to scenario 2 (limited data setting)
# intervp_0.5 corresponds to scenario 3 (less interventional data setting)
for dataset_name in C_6 M_20 intervp_0.5
do
    for trial in 1 2 3 4 5 6
    do
        
        num_groups=6
        experiment_name=ablation_study
        
        dataprefix=data/synthetic/output/syndata_${dataset_name}_trial_${trial}
        datafile=${dataprefix}_data.csv
        metafile=${dataprefix}_task_metadata.csv
        intervfile=${dataprefix}_intervmask.csv
        identifier=syndata_ablation_${dataset_name}_trial_${trial}
        
        # setup consistent conditions across all experiments
        num_pretrain_epochs=500
        num_initial_epochs=10
        num_main_epochs=100
        meta_learning_rate_global=0.001
        meta_learning_rate_group=0.0001
        meta_learning_rate=0.001
        base_learning_rate=0.001
        lambda=0.01
        causal_prior=erdos_renyi
        causal_model=linear
        alpha=0.02
        bandwidth_z=5
        bandwidth_theta=1000 
        outprefix=experiments/output/data/${identifier}
        
        # baseline 1: run the full method
        outpath=${outprefix}_ourmethodunknowncausalmodels_${num_groups}_${num_pretrain_epochs}_${num_initial_epochs}_${num_main_epochs}_${meta_learning_rate_global}_${meta_learning_rate_group}_${base_learning_rate}_${causal_model}_${causal_prior}_${lambda}_${alpha}_${bandwidth_z}_${bandwidth_theta}
        sbatch experiments/methods/our_method_unknown_causal_model.sbatch ${experiment_name} ${datafile} ${intervfile} ${num_groups} ${outpath} ${num_pretrain_epochs} ${num_initial_epochs} ${num_main_epochs} ${meta_learning_rate_global} ${meta_learning_rate_group} ${base_learning_rate} ${alpha} ${lambda} ${causal_prior} ${causal_model} ${bandwidth_z} ${bandwidth_theta}

        # ablation 1: remove the causal similarity aspect
        # (replace with some simpler form of task similarity in parameter space)
        outpath=${outprefix}_ourmethodunknowncausalmodelsnocausal_${num_groups}_${num_pretrain_epochs}_${num_initial_epochs}_${num_main_epochs}_${meta_learning_rate_global}_${meta_learning_rate_group}_${base_learning_rate}_${causal_model}_${causal_prior}_${lambda}_${alpha}_${bandwidth_z}_${bandwidth_theta}
        sbatch experiments/methods/our_method_unknown_causal_model_diagnostic.sbatch ${experiment_name} ${datafile} ${intervfile} ${num_groups} ${outpath} ${num_pretrain_epochs} ${num_initial_epochs} ${num_main_epochs} ${meta_learning_rate_global} ${meta_learning_rate_group} ${base_learning_rate} ${alpha} ${lambda} ${causal_prior} ${causal_model} ${bandwidth_z} ${bandwidth_theta}

        # ablation 2: remove the task similarity aspect entirely
        outpath=${outprefix}_metalearner_${base_learning_rate}_${meta_learning_rate}
        sbatch experiments/methods/bayesian_metalearner_baseline.sbatch ${experiment_name} ${datafile} ${outpath} ${base_learning_rate} ${meta_learning_rate}

        # ablation 3: remove latent causal representation for comparing causal similarity
        # replace with comparison in G space
        outpath=${outprefix}_ourmethodunknowncausalmodelsnolatent_${num_groups}_${num_pretrain_epochs}_${num_initial_epochs}_${num_main_epochs}_${meta_learning_rate_global}_${meta_learning_rate_group}_${base_learning_rate}_${causal_model}_${causal_prior}_${lambda}_${alpha}_${bandwidth_z}_${bandwidth_theta}
        sbatch experiments/methods/our_method_unknown_causal_model_alt.sbatch ${experiment_name} ${datafile} ${intervfile} ${num_groups} ${outpath} ${num_pretrain_epochs} ${num_initial_epochs} ${num_main_epochs} ${meta_learning_rate_global} ${meta_learning_rate_group} ${base_learning_rate} ${alpha} ${lambda} ${causal_prior} ${causal_model} ${bandwidth_z} ${bandwidth_theta}

        # ablation 4: remove the global hierarchy
        outpath=${outprefix}_ourmethodunknowncausalmodelsnoglobal_${num_groups}_${num_pretrain_epochs}_${num_initial_epochs}_${num_main_epochs}_${meta_learning_rate_global}_${meta_learning_rate_group}_${base_learning_rate}_${causal_model}_${causal_prior}_${lambda}_${alpha}_${bandwidth_z}_${bandwidth_theta}
        sbatch experiments/methods/our_method_unknown_causal_model.sbatch ${experiment_name} ${datafile} ${intervfile} ${num_groups} ${outpath} ${num_pretrain_epochs} 0 ${num_main_epochs} ${meta_learning_rate_global} ${meta_learning_rate_group} ${base_learning_rate} ${alpha} ${lambda} ${causal_prior} ${causal_model} ${bandwidth_z} ${bandwidth_theta}

    done
done