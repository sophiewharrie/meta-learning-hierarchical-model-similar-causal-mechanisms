import numpy as np
from sklearn.metrics import f1_score, roc_auc_score, average_precision_score, matthews_corrcoef, mean_squared_error, precision_score, recall_score, brier_score_loss, confusion_matrix
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, precision_recall_curve, auc
import pandas as pd
from scipy import stats


def task_classification_metrics(task_preds):
    """Get classification metrics for each task

    Assumes input is of form task_preds = {task_label : {y_actual: [], 'y_pred': []}}

    Returns output of form task_metrics = {task_label : {metric: value}}
    """
    task_metrics = {}

    for task_label in task_preds:
        metrics = classification_metrics(task_preds[task_label]['y_pred'], task_preds[task_label]['y_actual'], task_preds[task_label]['y_pred_bin'])
        task_metrics[task_label] = metrics
    
    return task_metrics


def avg_task_classification_metrics(task_metrics):
    """
    Get average of classification metrics across tasks

    Assumes input is of form task_metrics = {task_label : {metric: value}}

    Assumes all tasks have exactly the same metrics

    Returns output of form avg_metrics = {metric: value}
    """
    avg_metrics = {}

    task_labels = list(task_metrics.keys())
    metric_names = list(task_metrics[task_labels[0]].keys())

    for metric in metric_names:
        metric_values = [task_metrics[task_label][metric] for task_label in task_labels]
        avg_metrics[metric] = np.mean(metric_values)

    return avg_metrics


def classification_metrics(y_pred, y_true, y_pred_bin=None, threshold=0.50):
    """Metrics for evaluating (binary) classifcation tasks

    Assumes y_pred gives probabilities of 1 label and threshold specifies the cutoff for converting probabilities to binary values
    """
    if y_pred_bin is None: y_pred_bin = (y_pred >= threshold).astype(int) # convert to binary
    
    f1 = f1_score(y_true, y_pred_bin, average='binary')

    precision = precision_score(y_true, y_pred_bin, average='binary')

    recall = recall_score(y_true, y_pred_bin, average='binary')
    
    aucroc = roc_auc_score(y_true, y_pred)

    aucprc = average_precision_score(y_true, y_pred)

    mcc = matthews_corrcoef(y_true, y_pred_bin)

    brier = brier_score_loss(y_true, y_pred)

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred_bin).ravel()

    positive_rate = sum(y_pred_bin)/len(y_pred_bin) # the proportion of samples identified as positive (out of all samples, regardless of whether its correct or not)

    metrics = {'F1':f1,
               'Precision':precision,
               'Recall':recall,
               'AUCROC':aucroc, 
               'AUCPRC':aucprc, 
               'MCC':mcc,
               'Brier':brier,
               'False Negative Rate': fn / (fn + tp),
               'Specificity': tn / (tn + fp),
               'Negative Predictive Value': tn / (tn + fn),
               'True negatives':tn,
               'False positives':fp, 
               'False negatives':fn,
               'True positives':tp,
               'Positive rate':positive_rate}
    
    return metrics

def compute_curves(task_preds, args): 
    """Compute and save AUCROC and Precision-Recall curves for the task predictions. 
    
    Assumes input is of form task_preds = {task_label : {y_actual: [], 'y_pred': []}}
    Saves the curves to output_prefix. 
    """
    all_labels = []
    all_preds = []

    for task_label in task_preds:
        all_labels.extend(task_preds[task_label]['y_actual'])
        all_preds.extend(task_preds[task_label]['y_pred'])

    all_labels = np.array(all_labels)
    all_preds = np.array(all_preds)

    fpr, tpr, _ = roc_curve(all_labels, all_preds)
    precision_curve, recall_curve, _ = precision_recall_curve(all_labels, all_preds)
    roc_auc = auc(fpr, tpr)
    pr_auc = auc(recall_curve, precision_curve)

    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.4f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend(loc="lower right")
    plt.savefig('{}_roc_curve.png'.format(args['outprefix']))
    plt.close()

    plt.figure()
    plt.plot(recall_curve, precision_curve, color='blue', lw=2, label='PR curve (area = %0.4f)' % pr_auc)
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend(loc="upper right")
    plt.savefig('{}_pr_curve.png'.format(args['outprefix']))
    plt.close()

def get_pred_table(task_preds):
    all_task_preds = pd.DataFrame()

    for task in task_preds:
        task_pred = pd.DataFrame({'y_actual':task_preds[task]['y_actual'],
                                'y_pred':task_preds[task]['y_pred'],
                                'y_pred_bin':task_preds[task]['y_pred_bin'],
                                'y_pred_entropy':task_preds[task]['y_pred_entropy'], 
                                'aleatoric_uncertainty':task_preds[task]['aleatoric_uncertainty'],
                                'epistemic_uncertainty':task_preds[task]['epistemic_uncertainty']})
        
        task_pred['task'] = task
        task_pred['patient_ids'] = task_preds[task]['data_ids']

        all_task_preds = pd.concat([all_task_preds, task_pred])

    return all_task_preds