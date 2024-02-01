import argparse
import torch
import mlflow
import pandas as pd
import numpy as np
from tqdm import tqdm
import time
import sys
#sys.path.insert(0, 'model')
sys.path.insert(0, 'utils')
sys.path.insert(0, 'metrics')
from loader import DataModule
from causal_model import init_causal_model
from prediction_model import init_prediction_model, main_prediction_model, run_prediction_model
from causal_similarity_utils import get_causally_similar_tasks, pred_causal_group_new_tasks, get_groundtruth_representative_causal_models
from accuracy import calculate_metrics

def evaluate_model(pred_models, causal_groups, loader, args, metric_type, global_pred_model):
    y_pred_samples, y_true, y_task = [],[],[]
    task_number = 0 
    if args['causal_distance']=='ground_truth': ground_truth_causal_groups = loader.dataset.causal_groups
    for X_spt, Y_spt, X_qry, Y_qry, spt_interv_mask, qry_interv_mask in tqdm(loader):
        if args['causal_distance']=='ground_truth': causal_group = ground_truth_causal_groups[task_number]
        else: causal_group = causal_groups[task_number]
        y_pred_samples_task = run_prediction_model(X_spt, Y_spt, X_qry, Y_qry, causal_group=causal_group, group_pred_models=pred_models, args=args, train=False, eval=True, global_pred_model=global_pred_model)
        y_pred_samples.append(y_pred_samples_task)
        y_true.append(Y_qry.squeeze())
        y_task.append(np.array([task_number]*Y_qry.shape[1]))
        task_number += 1
    
    y_pred_samples = np.concatenate(y_pred_samples, axis=1)
    y_pred_mean = np.mean(y_pred_samples, axis=0)
    y_true = np.concatenate(y_true, axis=0)
    y_task = np.concatenate(y_task, axis=0)

    if args['regression']:
        metrics = calculate_metrics(y_pred_mean, y_true, y_task, task_type='regression', metric_type=metric_type)
    else: 
        metrics = calculate_metrics(y_pred_mean, y_true, y_task, task_type='classification', metric_type=metric_type)

    return metrics, causal_groups


def initialisation_step(train_loader, val_loader, test_loader, args):

    print("Initial training of predictive model")
    init_pred, pred_model_for_task_train, pred_model_for_task_val, pred_model_for_task_test = init_prediction_model(train_loader, val_loader, test_loader, args)    

    if args['causal_distance']=='ground_truth': 
        causal_groups_train = train_loader.dataset.causal_groups
        causal_groups_val = val_loader.dataset.causal_groups
        causal_groups_test = test_loader.dataset.causal_groups 
        # load ground truth representative model for each group from files
        representative_models = get_groundtruth_representative_causal_models(train_loader, causal_groups_train, args)

    elif args['causal_distance']=='diagnostic':
        causal_groups_train, representative_models = get_causally_similar_tasks(pred_model_for_task_train, args=args)
        causal_groups_val = pred_causal_group_new_tasks(pred_model_for_task_val, representative_models, args)
        causal_groups_test = pred_causal_group_new_tasks(pred_model_for_task_test, representative_models, args)

    else:
        # to be inferred from data
        print("Training the causal model")
        init_task_SCM_train, init_task_SCM_val, init_task_SCM_test = init_causal_model(train_loader, val_loader, test_loader, args)
        causal_groups_train, representative_models = get_causally_similar_tasks(init_task_SCM_train, args=args)
        causal_groups_val = pred_causal_group_new_tasks(init_task_SCM_val, representative_models, args)
        causal_groups_test = pred_causal_group_new_tasks(init_task_SCM_test, representative_models, args)
    
    return init_pred, causal_groups_train, causal_groups_val, causal_groups_test, representative_models


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--datafile', type=str, default='data/synthetic/output/test_data.csv', help='path for the dataset (.csv)')
    parser.add_argument('--intervfile', type=str, default='data/synthetic/output/test_intervmask.csv', help='path for the intervention mask of the dataset (.csv)')
    parser.add_argument('--metafile', type=str, default='data/synthetic/output/test_task_metadata.csv', help='path for the file containing causal groups of tasks (.csv) (only needed if ground truth causal similarity is known)')
    parser.add_argument('--causal_distance', type=str, default='inferred', help='set to ground_truth for actual causal groups, inferred if they need to be estimated from the data (our method), diagnostic if using a simple baseline (e.g., for debugging purposes)')
    parser.add_argument('--num_groups', type=int, default=2, help='number of causal groups')
    parser.add_argument('--outprefix', type=str, default=None, help='prefix for the output')
    parser.add_argument('--task', type=str, default='regression', help='either regression or classification')
    parser.add_argument('--num_pretrain_epochs', type=int, default=0, help='(maximum) number of training epochs for pretraining step, put 0 if not using pretraining')
    parser.add_argument('--num_initial_epochs', type=int, default=10, help='(maximum) number of training epochs for initialisation step')
    parser.add_argument('--num_main_epochs', type=int, default=30, help='(maximum) number of training epochs for main model')
    parser.add_argument('--patience', type=int, default=3, help='how many increases in validation loss to go before early stopping of training for main model (to prevent overfitting)')
    parser.add_argument('--meta_learning_rate_global', type=float, default=0.001, help='learning rate for global (meta) model (used in meta-training)')
    parser.add_argument('--meta_learning_rate_group', type=float, default=0.001, help='learning rate for group (meta) model (used in meta-training)')
    parser.add_argument('--base_learning_rate', type=float, default=0.001, help='learning rate for inner (base) model (used in both meta-training and meta-testing)')
    parser.add_argument('--hidden_layer_size', type=int, default=20, help='dimension of hidden layers for prediction model (assumes 2 hidden layers)')
    parser.add_argument('--num_base_steps', type=int, default=2, help='number of SGD steps for base learner')
    parser.add_argument('--n_edges_per_node', type=int, default=2, help='prior for number of edges per node in causal DAG')
    parser.add_argument('--obs_noise', type=float, default=0.1, help='noise for causal likelihood model (gaussian SCM)')
    parser.add_argument('--alpha', type=float, default=0.01, help='initial value of inverse temperature for p(G|Z)')
    parser.add_argument('--prior_pi', type=float, default=1, help='hyperparameter value for weight prior (used in both predictive and causal models)')
    parser.add_argument('--prior_sigma_1', type=float, default=0.1, help='hyperparameter value for weight prior (used in both predictive and causal models)')
    parser.add_argument('--prior_sigma_2', type=float, default=0.001, help='hyperparameter value for weight prior (used in both predictive and causal models)')
    parser.add_argument('--lambda', type=float, default=0.01, help='complexity cost for prediction model')
    parser.add_argument('--bandwidth_z', type=float, default=5, help='bandwidth parameter for kernel of Z')
    parser.add_argument('--bandwidth_theta', type=float, default=1000, help='bandwidth parameter for kernel of Theta')
    parser.add_argument('--num_mc_samples', type=int, default=3, help='number of samples to use when Monte Carlo approximations are used')
    parser.add_argument('--complexity_cost_weight', type=float, default=1, help='weighting of KL term in ELBO of causal model')
    parser.add_argument('--causal_prior', type=str, default='erdos_renyi', help='which causal prior to use, either erdos_renyi or scale_free')
    parser.add_argument('--causal_model', type=str, default='linear', help='which causal model to use, either linear or non_linear')

    args = vars(parser.parse_args())
    args['pretraining'] = args['num_pretrain_epochs'] > 0
    if not args['pretraining']: print("Skipping the pretraining sequence")

    if args['task'] == 'regression': args['regression'] = True
    else: args['regression'] = False

    if args['intervfile']=='ignore': 
        print("Ignoring the intervention mask input")
        args['use_interv_mask'] = False
    else: args['use_interv_mask'] = True

    mlflow.log_param("model", "our_method")

    args['device'] = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print("Loading the data")

    has_causal_groups = args['causal_distance']=='ground_truth'
    data = DataModule(args['datafile'], train_test_split=True, batch_size=1, shuffle=False, has_causal_groups=has_causal_groups, causalpath=args['metafile'], causaldist='ground_truth', interv_mask=True, intervpath=args['intervfile'])
    train_loader = data.train_dataloader()
    val_loader = data.val_dataloader()
    test_loader = data.test_dataloader()
    
    args['d'] = train_loader.dataset.num_features+1
    args['k'] = args['d']

    start_time = time.time()
    init_pred, causal_groups_train, causal_groups_val, causal_groups_test, representative_models = initialisation_step(train_loader, val_loader, test_loader, args)
    end_time = time.time()
    mlflow.log_metric("runtime_initialisation", end_time-start_time)

    # save the predicted causal group assignments
    if args['causal_distance'].startswith('inferred') or args['causal_distance']=='diagnostic':
        all_assignments = np.concatenate([causal_groups_train, causal_groups_val, causal_groups_test])
        pred_df = pd.DataFrame({'task':range(len(all_assignments)), 'predicted_groups':all_assignments})
        pred_df.to_csv('{}_causal_assignments_final.csv'.format(args['outprefix']), index=None)

    print("Training the main model")
    start_time = time.time()
    pred_models, total_num_epochs, global_pred_model = main_prediction_model(init_pred, train_loader, val_loader, causal_groups_train, causal_groups_val, representative_models, args)
    end_time = time.time()
    mlflow.log_metric("total_num_epochs", total_num_epochs)
    mlflow.log_metric("runtime_main_training", end_time-start_time)

    print("Validating model on meta-val set")
    metrics_val, causal_groups_val = evaluate_model(pred_models, causal_groups_val, val_loader, args, 'val', global_pred_model)
    
    print("Evaluating model on meta-test set")
    metrics_test, causal_groups_test = evaluate_model(pred_models, causal_groups_test, test_loader, args, 'test', global_pred_model)

