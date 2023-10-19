import argparse
import torch
import mlflow
import pandas as pd
import numpy as np
from tqdm import tqdm
import sys
sys.path.insert(0, 'metrics')
sys.path.insert(0, 'utils')
from loader import DataModule
from bayesian_metalearner_model import train_bayesian_maml, predict_bayesian_maml
from accuracy import calculate_metrics

def evaluate_model(model, loader, args, metric_type, nsamples=50):
    # fine tune model on support set and predict on query set (meta-testing)
    y_pred_samples, y_true, y_task = [],[],[]
    task_number = 0
    for X_spt, Y_spt, X_qry, Y_qry in tqdm(loader):
        y_pred_samples_task = predict_bayesian_maml(model, X_spt, Y_spt, X_qry, Y_qry, args, nsamples)
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
    parser.add_argument('--outprefix', type=str, help='prefix for the output')
    parser.add_argument('--task', type=str, default='regression', help='either regression or classification')
    parser.add_argument('--hidden_layer_size', type=int, default=20, help='dimension of hidden layers (assumes 2 hidden layers)')
    parser.add_argument('--num_epochs', type=int, default=50, help='number of training epochs')
    parser.add_argument('--meta_learning_rate', type=float, default=0.01, help='learning rate for outer (meta) model (used in meta-training)')
    parser.add_argument('--base_learning_rate', type=float, default=0.01, help='learning rate for inner (base) model (used in both meta-training and meta-testing)')
    parser.add_argument('--num_base_steps', type=int, default=2, help='number of SGD steps for base learner')
    parser.add_argument('--prior_sigma_1', type=float, default=0.1, help='prior sigma 1 on the mixture prior distribution')
    parser.add_argument('--prior_sigma_2', type=float, default=0.4, help='prior sigma 2 on the mixture prior distribution')
    parser.add_argument('--prior_pi', type=float, default=1.0, help='pi on the scaled mixture prior')
    parser.add_argument('--lambda', type=float, default=0.01, help='cold posterior constant or complexity cost to improve variational inference')

    args = vars(parser.parse_args())
    if args['task'] == 'regression': args['regression'] = True
    else: args['regression'] = False

    args['device'] = 'cpu' # TODO torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print("Training a Bayesian meta-learning model")
    data = DataModule(args['datafile'], train_test_split=True, batch_size=1, shuffle=True)
    train_loader = data.train_dataloader()
    val_loader = data.val_dataloader()
    test_loader = data.test_dataloader()

    mlflow.log_param("model", "bayesian_maml_baseline")
    
    print("Training model on meta-test set")
    model = train_bayesian_maml(train_loader, val_loader, args)

    print("Validating model on meta-val set")
    metrics_val = evaluate_model(model, val_loader, args, 'val')
    
    print("Evaluating model on meta-test set")
    metrics_test = evaluate_model(model, test_loader, args, 'test')