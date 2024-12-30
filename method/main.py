import argparse
import numpy as np
import sys
import torch
import traceback
import time

sys.path.insert(0, 'utils')
sys.path.insert(0, 'metrics')
sys.path.insert(0, 'method')
from loader import DataModule
from accuracy import task_classification_metrics
from accuracy import avg_task_classification_metrics
from hierarchical_model import hierarchical_objective
from hierarchical_eval_utils import evaluate_best_model


def objective_cv(config=None):
    
    with wandb.init(config=config):
        config = wandb.config

        scores = []

        # for k-fold cross validation, we compute the mean of the scores for the validation sets across k folds

        for fold in range(args['n_kfold_splits']):

            print("Fitting model for fold {} / {}".format(fold+1, args['n_kfold_splits']))

            # run the hyperparameter tuning study
            try:
                score, task_metrics, avg_metrics, failed_run = objective(config)

            except Exception as e:
                # log as failed run
                error_msg = str(e)
                stacktrace = traceback.format_exc()
                failed_run = True
                wandb.log({"error": error_msg})
                print(f"Run failed with error: {error_msg}", file=sys.stderr)
                print(stacktrace, file=sys.stderr)

            if fold == args['n_kfold_splits']-1 or failed_run:
                # reset the iterator for the next task
                data.reset_kfold_iterator()
            else:
                # update to the next fold of train/val data
                data.train_samples, data.val_samples = data.get_next_kfold()

            if failed_run:
                print("Failed run, likely due to bad hyperparameter values (see log for error message)")
                score = -1000 if args['wandb_direction']=='maximize' else 1000 # log poor score for failed run (NOTE: don't use np.inf because it causes a bug in W&B)
                scores.append(score)
                break
            
            scores.append(score)

        score = np.mean(scores)
        wandb.log({"score": score})

    wandb.finish()


def objective(config):
    """Define the objective function for the hyperparameter tuning
    """

    # NOTE - the objective function returns predictions for each task, in a dictionary of the form {task_label : {y_actual: [], 'y_pred': []}}

    # the score for the trial is calculated by averaging the evaluation metrics across all tasks - this encourages the tuning to identify hyperparameters that perform well (on average) across all tasks
    
    val_task_preds = hierarchical_objective(config, trainloader, taskloader, data, args)
    
    if val_task_preds is None:
        # trial stopped early due to error
        return None, None, None, True

    # get classification metrics for each task, then average across tasks
    task_metrics = task_classification_metrics(val_task_preds)
    avg_metrics = avg_task_classification_metrics(task_metrics)
    score = avg_metrics[args['wandb_eval']]

    return score, task_metrics, avg_metrics, False


def get_tuning_settings():
    parameters = {
        "nn_prior_name": {"values": ['multivariate_normal_prior']},
        "hidden_layer_size_tabular": {'distribution': 'q_log_uniform_values', 'min': 20, 'max': 160, 'q': 1},
        "inner_learning_rate": {'distribution': 'log_uniform_values', 'min': 1e-6, 'max': 1e-1},
        "inner_temperature": {'distribution': 'log_uniform_values', 'min': 1e-5, 'max': 1e-1},
        "global_prior_sigma": {'distribution': 'log_uniform_values', 'min': 1e-5, 'max': 1},
        "model_init_log_sds": {'distribution': 'uniform', 'min': np.log(0.0001), 'max': 0},
        "prior_scaling": {"value": 10000} # a constant used for scaling of prior (in scale of size of dataset, adjusted accordingly or set to 1 if not needed)
    }
    
    # nn architecture specific parameters
    if args['data_type'] == 'tabular':
        parameters["nn_model_name"] = {"value": 'linear_nn_model'}
        parameters["n_layers"] = {"values": [2]}
    elif args['data_type'] == 'sequence':
        parameters["nn_model_name"] = {"value": 'sequence_nn_model'}
        parameters["hidden_layer_size_longitudinal"] = {'distribution': 'q_log_uniform_values', 'min': 20, 'max': 160, 'q': 1}

    # model specific parameters
    if args['method'] == 'bnn_baseline':
        parameters["num_inner_updates"] = {"value": args['max_num_epochs']}

    elif args['method'] in ['2_level_hierarchical']:
        parameters["num_inner_updates"] = {'distribution': 'int_uniform', 'min': 1, 'max': 9}
        parameters["outer_temperature"] = {'distribution': 'log_uniform_values', 'min': 1e-5, 'max': 1e-1}
        parameters["outer_learning_rate"] = {'distribution': 'log_uniform_values', 'min': 1e-6, 'max': 1e-1}
        
    elif args['method'] in ['3_level_hierarchical']:
        parameters["num_inner_updates"] = {'distribution': 'int_uniform', 'min': 1, 'max': 9}
        parameters["outer_temperature"] = {'distribution': 'log_uniform_values', 'min': 1e-5, 'max': 1e-1}
        parameters["outer_learning_rate"] = {'distribution': 'log_uniform_values', 'min': 1e-6, 'max': 1e-1}
        parameters["auxiliary_learning_rate"] = {'distribution': 'log_uniform_values', 'min': 1e-6, 'max': 1e-1}
        parameters["auxiliary_temperature"] = {'distribution': 'log_uniform_values', 'min': 1e-5, 'max': 1e-1}
        parameters["num_auxiliary_updates"] = {'distribution': 'int_uniform', 'min': 1, 'max': 9}
        parameters["eta_1"] = {'distribution': 'uniform', 'min': 0, 'max': 5} # modulating factor for task similarity weights in likelihood
        parameters["eta_2"] = {'distribution': 'uniform', 'min': 0, 'max': 5} # modulating factor for task similarity weights in minibatching
        parameters["avg_num_active_tasks"] = {'distribution': 'int_uniform', 'min': 1, 'max': 6}
    
    return parameters


def create_sweep(sweep_config, project_name, sweep_filename):
    # create a new Weights & Biases sweep
    sweep_id = wandb.sweep(sweep_config, project=project_name)
    
    with open(sweep_filename, 'w') as f:
        # write the sweep ID to the file
        f.write(sweep_id)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--tabular_datafile', type=str, help='path for the mailfile, which is a tabular dataset (.csv)')
    parser.add_argument('--longitudinal_datafile', type=str, default='none', help='path for the longfile, which is a longitudinal dataset (.csv). Only needed if using sequence model')
    parser.add_argument('--metafile', type=str, help='path for the metadata file for the dataset (.csv)')
    parser.add_argument('--distancefile', type=str, help='path for the distfile, which is a causal distance file used to construct task similarity kernels for the 3-level hierarchical model (.csv)')
    parser.add_argument('--outprefix', type=str, help='prefix for where to store results and other outputs')
    parser.add_argument('--data_type', type=str, default='sequence', help='`tabular` for non-sequence data or `sequence` for combined tabular and sequence (longitudinal) model')
    parser.add_argument('--learning_type', type=str, default='transductive', help='`transductive` if target tasks are also in training set or `inductive` if target tasks are separate to training set')
    parser.add_argument('--test_frac', type=float, default=0.5, help='fraction of data to withhold for testing (e.g. 0.4 means 40% of samples)')
    parser.add_argument('--query_frac', type=float, default=0.5, help='fraction of data to withhold for query set in support/query split (e.g., 0.4 means 40% of samples)')
    parser.add_argument('--case_frac', type=float, default=0.5, help='fraction of data for cases when balancing case/controls in sampling (e.g., 0.4 means 40% of samples)')
    parser.add_argument('--batch_size', type=int, default=100, help='amount of task data to use for each batch')
    parser.add_argument('--method', type=str, default='bnn_baseline', help='machine learning algorithm, choose from: bnn_baseline, 2_level_hierarchical, 3_level_hierarchical')
    parser.add_argument('--random_seed', type=int, default=42, help='random seed for reproducibility of data splits (should be unique for each trial')
    parser.add_argument('--n_kfold_splits', type=int, default=2, help='number of folds used in cross validation')
    parser.add_argument('--minibatch_size', type=int, default=5, help='number of tasks in minibatch for each global update')
    parser.add_argument('--max_num_epochs', type=int, default=30, help='maximum number of training epochs used by neural network algorithms')
    parser.add_argument('--num_mc_samples', type=int, default=5, help='number of MC samples used by Bayesian ML algorithms')
    parser.add_argument('--wandb_n_trials', type=int, default=20, help='number of trials to use in the wandb hyperparameter optimisation study')
    parser.add_argument('--wandb_eval', type=str, default='AUCPRC', help='evaluation metric used for scoring in wandb study')
    parser.add_argument('--wandb_direction', type=str, default='maximize', help='whether the objective is to minimize or maximize the evaluation metric')
    parser.add_argument('--wandb_key_file', type=str, default='credentials/wandb_api_key.txt', help='path to the file containing the Weights & Biases API key')
    parser.add_argument('--sweep_id_filename', type=str, help='where to store and read sweep ID for the Weights & Biases hyperparameter tuning')
    parser.add_argument('--wandb_project', type=str, help='name of project for Weights & Biases hyperparameter tuning')
    parser.add_argument('--wandb_entity', type=str, help='username for Weights & Biases hyperparameter tuning')
    parser.add_argument('--mode', type=str, default="regular", help='`regular` (runs hyperparameter tuning), `debug` (no hyperparameter tuning), or `init` (setup new sweep for hyperparameter tuning)')

    args = vars(parser.parse_args())
    
    args['device'] = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {args['device']}")

    print(args)

    data = DataModule(tabular_datapath=args['tabular_datafile'], longitudinal_datapath=args['longitudinal_datafile'], metapath=args['metafile'], distpath=args['distancefile'], method=args['method'], learning_type=args['learning_type'], test_frac=args['test_frac'], query_frac=args['query_frac'], case_frac=args['case_frac'], n_kfold_splits=args['n_kfold_splits'], random_seed=args['random_seed'], data_type=args['data_type'])

    # run an wandb study for multitask learning across all tasks and report the evaluation metrics for the best model (i.e. best average performance across all tasks)
    taskloader = data.get_taskloader(batchsize=1, shuffle=False, target_tasks=False) # during training, use same tasks for training and validation
    trainloader = data.get_taskloader(batchsize=args['minibatch_size'], shuffle=True, target_tasks=False)

    if args['mode']=='debug':

        # no wandb trial, used fixed set of parameters (adjust if needed)
        # NOTE requires giving the parameter values, using dictionaries similar to templates given below

        if args['method'] == 'bnn_baseline':
            # NOTE replace with your values, these are just arbitrary placeholders
            best_params = {
                'nn_prior_name': 'multivariate_normal_prior',  
                'inner_learning_rate': 0.01,  
                'inner_temperature': 0.001, 
                'global_prior_sigma': 0.05,
                'model_init_log_sds': -7.0,
                'hidden_layer_size_longitudinal': 60, 
                'hidden_layer_size_tabular': 60, 
                'num_inner_updates': 3,
                'prior_scaling': 10000,
                }
            
        elif args['method'] == '2_level_hierarchical':
             # NOTE replace with your values, these are just arbitrary placeholders
            best_params = {
                'nn_prior_name': 'multivariate_normal_prior',  
                'inner_learning_rate': 0.01,  
                'inner_temperature': 0.001, 
                'outer_temperature': 0.001, 
                'outer_learning_rate': 0.01, 
                'global_prior_sigma': 0.05,
                'model_init_log_sds': -7.0,
                'hidden_layer_size_longitudinal': 60, 
                'hidden_layer_size_tabular': 60, 
                'num_inner_updates': 3,
                'prior_scaling': 10000,
                }
            
        else:
             # NOTE replace with your values, these are just arbitrary placeholders
            best_params = {
                'nn_prior_name': 'multivariate_normal_prior',  
                'inner_learning_rate': 0.01,  
                'inner_temperature': 0.001, 
                'outer_temperature': 0.001, 
                'outer_learning_rate': 0.01, 
                'global_prior_sigma': 0.05,
                'model_init_log_sds': -7.0,
                'hidden_layer_size_longitudinal': 60, 
                'hidden_layer_size_tabular': 60, 
                'num_inner_updates': 3,
                'auxiliary_learning_rate': 0.01, 
                'auxiliary_temperature': 0.001, 
                'num_auxiliary_updates': 2,
                'eta_1': 1,
                'eta_2': 0, 
                'avg_num_active_tasks': 1,
                'prior_scaling': 10000,
                }

        if args['method'] == 'bnn_baseline':
            best_params['num_inner_updates'] = args['max_num_epochs']
        
        if args['data_type'] == 'tabular':
            best_params["nn_model_name"] = 'linear_nn_model'
        elif args['data_type'] == 'sequence':
            best_params["nn_model_name"] = 'sequence_nn_model'
    
    else:

        print("Running hyperparameter tuning study")

        import wandb

        try:
            with open(args['wandb_key_file'], 'r') as f:
                api_key = f.read().strip()

            wandb.login(key=api_key)

        except:
            print("Failed to read Weights & Biases API key from 'next' file. Exiting.")
            sys.exit(1)
        
        # config for hyperparameter tuning
        sweep_config = {
            "method": "bayes", # bayesian optimisation
            "metric": {"goal": args['wandb_direction'], "name": "score"}, # for binary classification
            "parameters": get_tuning_settings()
        }

        if args['mode'] == 'init':
            create_sweep(sweep_config, args['wandb_project'], args['sweep_id_filename'])
            time.sleep(90) # give time for config to be synced to cloud

        else:

            # run hyperparameter tuning (sweeps)    
            with open(args['sweep_id_filename'], 'r') as f:
                sweep_id = f.read().strip()
            
            print(f"Using sweep ID: {sweep_id}")

            wandb.agent(sweep_id, project=args['wandb_project'], entity=args['wandb_entity'], function=objective_cv, count=args['wandb_n_trials'])

            # retrieve the best run's configuration
            api = wandb.Api()
            sweep = api.sweep(sweep_id)
            best_run = sweep.best_run()
            best_params = best_run.config

            print(f"Best run ID: {best_run.id}")
            print("Best parameters:")
            print(best_params)

    # evaluate best model on test set and report evaluation metrics

    if args['mode'] != 'init':

        # reset the data loaders
        taskloader = data.get_taskloader(batchsize=1, shuffle=False, target_tasks=True)
        trainloader = data.get_taskloader(batchsize=args['minibatch_size'], shuffle=True, target_tasks=False)
        
        evaluate_best_model(best_params, trainloader, taskloader, data, args)