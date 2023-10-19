import argparse
import mlflow
import pandas as pd
import sys
import numpy as np
sys.path.insert(0, 'metrics')
sys.path.insert(0, 'utils')
from loader import DataModule
from bnn_model import train_bnn, predict_bnn
from accuracy import calculate_metrics

def evaluate_model(X, Y, task_list, args, inference_model, predictive_input, metric_type):
    y_pred_mean, y_pred_samples, y_true, y_task = predict_bnn(X, Y, task_list, inference_model, predictive_input)

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

    df = pd.read_csv(args['datafile'])
    
    print("Training a global model")
    
    mlflow.log_param("model", "global_bnn_baseline")
    
    hidden_layers = [args['hidden_layer_size'],args['hidden_layer_size']] # model has 2 hidden layers

    # use all data for training (no support/query split)
    data = DataModule(args['datafile'], train_test_split=False, batch_size=-1, shuffle=True)
    train_loader = data.train_dataloader()
    X_data, Y_data = next(iter(train_loader))
    X_data = np.array(X_data.view(X_data.shape[0]*X_data.shape[1], -1))
    Y_data = np.array(Y_data.reshape(-1))
    
    inference_model, predictive_input = train_bnn(X_data, Y_data, args['outprefix'], args['regression'], args['inference'], args['num_warmup'], args['num_samples'], args['num_chains'], hidden_layers, args['learning_rate'], args['num_steps'])

    # report the error on the meta-val and meta-test sets
    # use query samples (for consistency with meta-learning methods)
    data = DataModule(args['datafile'], train_test_split=True, batch_size=-1, shuffle=True)
    val_loader = data.val_dataloader()
    test_loader = data.test_dataloader()
    X_val_spt, Y_val_spt, X_val_qry, Y_val_qry = next(iter(val_loader))
    val_tasks = np.tile(np.arange(X_val_qry.shape[0]).reshape(-1, 1), (1, X_val_qry.shape[1])).reshape(-1)
    X_val_data = np.array(X_val_qry.view(X_val_qry.shape[0]*X_val_qry.shape[1], -1))
    Y_val_data = np.array(Y_val_qry.reshape(-1))
    X_test_spt, Y_test_spt, X_test_qry, Y_test_qry = next(iter(test_loader))
    test_tasks = np.tile(np.arange(X_test_qry.shape[0]).reshape(-1, 1), (1, X_test_qry.shape[1])).reshape(-1)
    X_test_data = np.array(X_test_qry.view(X_test_qry.shape[0]*X_test_qry.shape[1], -1))
    Y_test_data = np.array(Y_test_qry.reshape(-1))
    
    print("Validating model on meta-val set")
    metrics_test = evaluate_model(X_val_data, Y_val_data, val_tasks, args, inference_model, predictive_input, 'val')
    
    print("Evaluating model on meta-test set")
    metrics_test = evaluate_model(X_test_data, Y_test_data, test_tasks, args, inference_model, predictive_input, 'test')
    