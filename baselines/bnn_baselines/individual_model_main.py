import argparse
import mlflow
import pandas as pd
import jax.numpy as jnp
import sys
sys.path.insert(0, 'metrics')
sys.path.insert(0, 'utils')
from loader import DataModule
from bnn_model import train_bnn, predict_bnn
from accuracy import calculate_metrics

def evaluate_model(loader, args, hidden_layers, metric_type):
    y_pred_samples, y_true, y_task = [],[],[]
    task_number = 0
    for X_spt, Y_spt, X_qry, Y_qry in loader:
        X_train = jnp.array(X_spt[0])
        Y_train = jnp.array(Y_spt[0])
        X_test = jnp.array(X_qry[0])
        Y_test = jnp.array(Y_qry[0])
        task_numbers = jnp.array([task_number]*len(Y_test))
        inference_model, predictive_input = train_bnn(X_train, Y_train, args['outprefix'], args['regression'], args['inference'], args['num_warmup'], args['num_samples'], args['num_chains'], hidden_layers, args['learning_rate'], args['num_steps'])
        y_pred_mean_task, y_pred_samples_task, y_true_task, y_task_task = predict_bnn(X_test, Y_test, task_numbers, inference_model, predictive_input)
        y_pred_samples.append(y_pred_samples_task)
        y_true.append(y_true_task)
        y_task.append(y_task_task)
        task_number += 1

    y_pred_samples = jnp.concatenate(y_pred_samples, axis=1)
    y_pred_mean = jnp.mean(y_pred_samples, axis=0)
    y_true = jnp.concatenate(y_true, axis=0)
    y_task = jnp.concatenate(y_task, axis=0)
    
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
    parser.add_argument('--inference', type=str, default='SVI', help='either HMC or SVI')
    parser.add_argument('--num_samples', type=int, default=100, help='number of MC samples to generate')
    parser.add_argument('--hidden_layer_size', type=int, default=10, help='dimension of hidden layers (assumes 2 hidden layers)')
    parser.add_argument('--num_warmup', type=int, default=500, help='(for HMC) number of warmup iterations')
    parser.add_argument('--num_chains', type=int, default=1, help='(for HMC) number of mcmc chains to run')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='(for SVI) learning rate for model training (uses Adam optimiser)')
    parser.add_argument('--num_steps', type=int, default=10000, help='(for SVI) number of steps for model training')
    
    args = vars(parser.parse_args())
    if args['task'] == 'regression': args['regression'] = True
    else: args['regression'] = False

    mlflow.log_param("model", "individual_bnn_baseline")
    
    hidden_layers = [args['hidden_layer_size'],args['hidden_layer_size']] # model has 2 hidden layers

    # report the error on the meta-val and meta-test sets
    data = DataModule(args['datafile'], train_test_split=True, batch_size=1, shuffle=True)

    val_loader = data.val_dataloader()
    print("Training individual models for {} meta-val tasks".format(len(val_loader)))
    metrics_val = evaluate_model(val_loader, args, hidden_layers, 'val')

    test_loader = data.test_dataloader()
    print("Training individual models for {} meta-test tasks".format(len(test_loader)))
    metrics_test = evaluate_model(test_loader, args, hidden_layers, 'test')
    