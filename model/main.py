import argparse
import torch
import mlflow
import pandas as pd
import numpy as np
from tqdm import tqdm
import time
import sys
sys.path.insert(0, 'metrics')
sys.path.insert(0, 'utils')
from loader import DataModule
from model import initial_model, main_model, predict_model
from causal_inference import predict_causal_groups_for_new_data
from accuracy import calculate_metrics


def evaluate_model(models, loader, args, metric_type, nsamples=50):
    # fine tune model on support set and predict on query set (meta-testing)
    y_pred_samples, y_true, y_task = [],[],[]
    task_number = 0
    for X_spt, Y_spt, X_qry, Y_qry, causal_group in tqdm(loader):
        y_pred_samples_task = predict_model(models[causal_group], X_spt, Y_spt, X_qry, Y_qry, args['base_learning_rate_main'], args, nsamples)
        y_pred_samples.append(y_pred_samples_task)
        y_true.append(Y_qry.squeeze())
        y_task.append(np.array([task_number]*Y_qry.shape[1])) # give task number for each prediction
        task_number += 1
        
    y_pred_samples = np.concatenate(y_pred_samples, axis=1)
    y_pred_mean = np.mean(y_pred_samples, axis=0)
    y_true = np.concatenate(y_true, axis=0)
    y_task = np.concatenate(y_task, axis=0)

    if args['regression']:
        metrics = calculate_metrics(y_pred_mean, y_true, y_task, task_type='regression', metric_type=metric_type)
    else: 
        metrics = calculate_metrics(y_pred_mean, y_true, y_task, task_type='classification', metric_type=metric_type)

    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--datafile', type=str, help='path for the dataset (.csv)')
    parser.add_argument('--simfile', type=str, help='path for the file containing causal distances of task pairs (.csv) (only needed for known CGMs)')
    parser.add_argument('--metafile', type=str, help='path for the file containing causal similarities of tasks (.csv) (only needed for ground truth causal distance)')
    parser.add_argument('--num_groups', type=int, default=4, help='number of causal groups (not needed for ground truth causal distance)')
    parser.add_argument('--causal_distance', type=str, default='ground_truth', help='one of ground_truth,SID,SHD,OD,ID,TOD_alphaxx,TID_alphaxx (where xx is a value for alpha) (only needed for known CGMs)')
    parser.add_argument('--unknown_causal_models', action='store_true', help='if true, ignores the simfile and infers the causal group assignments from the data')
    parser.add_argument('--inference_type', type=str, default='interventional', help='observational or interventional approach for causal inference (only required if causal groups are being inferred from data)')
    parser.add_argument('--interventional_option', type=int, default=4, help='which type of interventional proxy to use, choose from 0, 1, ..., 5')
    parser.add_argument('--outprefix', type=str, default=None, help='prefix for the output')
    parser.add_argument('--task', type=str, default='regression', help='either regression or classification')
    parser.add_argument('--hidden_layer_size', type=int, default=20, help='dimension of hidden layers (assumes 2 hidden layers)')
    parser.add_argument('--num_initial_epochs', type=int, default=50, help='number of training epochs for initialisation step')
    parser.add_argument('--num_main_epochs', type=int, default=50, help='(maximum) number of training epochs for main model')
    parser.add_argument('--patience', type=int, default=2, help='how many increases in validation loss to go before early stopping of training for main model (to prevent overfitting)')
    parser.add_argument('--meta_learning_rate_initial', type=float, default=0.01, help='learning rate for outer (meta) model (used in meta-training) for initial model')
    parser.add_argument('--base_learning_rate_initial', type=float, default=0.01, help='learning rate for inner (base) model (used in both meta-training and meta-testing) for initial model')
    parser.add_argument('--meta_learning_rate_global_main', type=float, default=0.01, help='learning rate for global (meta) model (used in meta-training) for main model')
    parser.add_argument('--meta_learning_rate_group_main', type=float, default=0.01, help='learning rate for group (meta) models (used in meta-training) for main model')
    parser.add_argument('--base_learning_rate_main', type=float, default=0.01, help='learning rate for inner (base) model (used in both meta-training and meta-testing) for main model')
    parser.add_argument('--num_base_steps', type=int, default=2, help='number of SGD steps for base learner')
    parser.add_argument('--prior_sigma_1', type=float, default=0.1, help='prior sigma 1 on the mixture prior distribution')
    parser.add_argument('--prior_sigma_2', type=float, default=0.4, help='prior sigma 2 on the mixture prior distribution')
    parser.add_argument('--prior_pi', type=float, default=1.0, help='pi on the scaled mixture prior')
    parser.add_argument('--lambda', type=float, default=0.01, help='cold posterior constant or complexity cost to improve variational inference')

    args = vars(parser.parse_args())
    if args['task'] == 'regression': args['regression'] = True
    else: args['regression'] = False

    # abalation studies for interventional distance
    args['weighted_average'] = False
    args['replace_missing_terms'] = False
    args['uncertainty_penalty'] = False
    if args['unknown_causal_models'] and args['inference_type']=='interventional':
        # interventional_option=0 ablation study: no replacement of missing terms
        if args['interventional_option'] == 1: 
            args['weighted_average'] = True
        elif args['interventional_option'] == 2: # ablation study: no penalty for uncertainty
            args['replace_missing_terms'] = True
        elif args['interventional_option'] == 3: 
            args['replace_missing_terms'] = True
            args['weighted_average'] = True
        elif args['interventional_option'] == 4: # best option
            args['replace_missing_terms'] = True
            args['uncertainty_penalty'] = True
        elif args['interventional_option'] == 5: 
            args['replace_missing_terms'] = True
            args['weighted_average'] = True
            args['uncertainty_penalty'] = True
    
    args['device'] = 'cpu' # TODO torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print("Training our causal similarity-based hierarchical model")

    mlflow.log_param("model", "our_method")

    print("Initialisation step")
    
    # train init_model to determine priors and causal group assignments for training data
    # NOTE shuffle=False so causal groups can be updated in order
    data = DataModule(args['datafile'], train_test_split=True, batch_size=1, shuffle=False)
    train_loader = data.train_dataloader()
    val_loader = data.val_dataloader()
    test_loader = data.test_dataloader()

    start_time = time.time()
    init_model, metatrain_causal_groups, task_meta_similarities, task_parameters = initial_model(train_loader, val_loader, args, train_loader.dataset.intervention_variables, train_loader.dataset.intervention_weights)
    end_time = time.time()
    mlflow.log_metric("runtime_initialisation", end_time-start_time)

    if args['unknown_causal_models'] or args['causal_distance']!='ground_truth':
        print("Inferring the causal groups for the validation dataset")
        metaval_causal_groups = predict_causal_groups_for_new_data(val_loader, train_loader, metatrain_causal_groups, args, train_loader.dataset.intervention_variables, train_loader.dataset.intervention_weights, task_meta_similarities, task_parameters, init_model, predict_model)
        print("Inferring the causal groups for the test dataset")
        metatest_causal_groups = predict_causal_groups_for_new_data(test_loader, train_loader, metatrain_causal_groups, args, train_loader.dataset.intervention_variables, train_loader.dataset.intervention_weights, task_meta_similarities, task_parameters, init_model, predict_model)

        # get new loaders with shuffled data
        data = DataModule(args['datafile'], train_test_split=True, batch_size=1, shuffle=True)
        train_loader = data.train_dataloader()
        val_loader = data.val_dataloader()
        test_loader = data.test_dataloader()
        train_loader.dataset.has_causal_groups = True
        train_loader.dataset.num_causal_groups = args['num_groups']
        train_loader.dataset.causal_groups = metatrain_causal_groups
        val_loader.dataset.has_causal_groups = True
        val_loader.dataset.num_causal_groups = args['num_groups']
        val_loader.dataset.causal_groups = metaval_causal_groups
        test_loader.dataset.has_causal_groups = True
        test_loader.dataset.num_causal_groups = args['num_groups']
        test_loader.dataset.causal_groups = metatest_causal_groups
        
        # save the predicted causal group assignments for evaluation
        all_assignments = np.concatenate([metatrain_causal_groups, metaval_causal_groups, metatest_causal_groups])
        pred_df = pd.DataFrame({'task':range(len(all_assignments)), 'predicted_groups':all_assignments})
        pred_df.to_csv('{}_causal_assignments.csv'.format(args['outprefix']), index=None)

    elif args['causal_distance']=='ground_truth':
        print("Using ground truth causal assignments")
        data = DataModule(args['datafile'], train_test_split=True, batch_size=1, shuffle=True, has_causal_groups=True, causalpath=args['metafile'], causaldist=args['causal_distance'])
        train_loader = data.train_dataloader()
        val_loader = data.val_dataloader()
        test_loader = data.test_dataloader()

    print("Training the main model")
    start_time = time.time()
    models, total_num_epochs = main_model(train_loader, val_loader, args, init_model)
    mlflow.log_metric("total_num_epochs", total_num_epochs)
    end_time = time.time()
    mlflow.log_metric("runtime_main_training", end_time-start_time)

    print("Validating model on meta-val set")
    metrics_val = evaluate_model(models, val_loader, args, 'val')
    
    print("Evaluating model on meta-test set")
    metrics_test = evaluate_model(models, test_loader, args, 'test')