from sklearn.metrics import mean_squared_error
from sklearn.metrics import f1_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import average_precision_score
import numpy as np
import mlflow


def calculate_metrics(y_pred, y_true, y_task, task_type='regression', metric_type='train', threshold=0.5):
    """
    A function to calculate metrics for a set of tasks and log using MLflow. 
    Specify task_type as either `regression` or `classification`.

    Note that these metrics are calculated on point estimates (e.g. mean of posterior distribution)
    so the inputs are 1D numpy arrays. Note that y_pred, y_true, y_task have the same dimension

    Parameters:
    y_pred (numpy array, continuous values): gives the predicted value (for classification give probability value)
    y_true (numpy array, continuous for regression, binary for classification): gives the true values 
    y_task (numpy array): gives task number of each value in y_pred (equivalently, y_true) 
    task_type (string): calculate regression metrics if 'regression', or binary classification metrics if 'classification'
    metric_type (string): `train` if calculating metric for the meta-train set or `test` if for meta-test set (for logging purposes)
    threshold (default 0.5): binary classification threshold for probability that determines a 1 label 
    """
    metric_values = []
    for task in np.unique(y_task):
        y_pred_task = y_pred[y_task==task]
        y_true_task = y_true[y_task==task]
        if task_type=='regression':
            task_metric = regression_metrics(y_pred_task, y_true_task, metric_type)
        else:
            task_metric = classification_metrics(y_pred_task, y_true_task, threshold, metric_type)
        metric_values.append(task_metric)
    
    metric_names = list(metric_values[0].keys())
    task_avgs = {}
    for metric in metric_names:
        task_avg = np.mean([task_metric[metric] for task_metric in metric_values])
        label = f'{metric}_avg_{metric_type}'
        print(f'{label}: {task_avg}')
        mlflow.log_metric(label, task_avg)
        task_avgs[f'{metric}_avg'] = task_avg

    return task_avgs


def regression_metrics(y_pred, y_true, metric_type='train', log=True):
    """Metrics for evaluating regression tasks
    """
    # Mean squared error regression loss (smaller is better)
    rmse = mean_squared_error(y_true, y_pred, squared=False)
    if log: mlflow.log_metric(f"RMSE_task_{metric_type}",rmse)
    
    metrics = {'RMSE':rmse}

    return metrics


def classification_metrics(y_pred, y_true, threshold=0.5, metric_type='train', log=True):
    """Metrics for evaluating classifcation tasks
    """
    # F1 score (larger is better)
    y_pred_bin = (y_pred >= threshold).astype(int) # convert to binary
    f1 = f1_score(y_true, y_pred_bin)
    if log: mlflow.log_metric(f"F1_task_{metric_type}",f1)

    # AUC-ROC score (larger is better)
    aucroc = roc_auc_score(y_true, y_pred)
    if log: mlflow.log_metric(f"AUCROC_task_{metric_type}",aucroc)
    
    # AUC-PRC score (larger is better)
    aucprc = average_precision_score(y_true, y_pred)
    if log: mlflow.log_metric(f"AUCPRC_task_{metric_type}",aucprc)
    
    metrics = {'F1':f1, 
               'AUCROC':aucroc, 
               'AUCPRC':aucprc}
    
    return metrics