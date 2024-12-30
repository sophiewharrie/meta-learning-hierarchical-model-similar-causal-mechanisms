import sys
sys.path.insert(0, 'utils')
sys.path.insert(0, 'metrics')
sys.path.insert(0, 'method')
from accuracy import task_classification_metrics
from accuracy import avg_task_classification_metrics
from accuracy import compute_curves
from hierarchical_model import hierarchical_best_model


def evaluate_best_model(best_params, trainloader, taskloader, data, args):
    """Retrain the best model (using best parameters) on full training set.

    Evaluate using test set and log all metadata and evaluation metrics
    """
    
    test_task_preds = hierarchical_best_model(best_params, trainloader, taskloader, data, args)
    
    task_metrics = task_classification_metrics(test_task_preds)
    avg_metrics = avg_task_classification_metrics(task_metrics)

    print('Evaluation metrics:')
    for task_label in task_metrics:
        print("Task {}:".format(task_label))
        for metric in task_metrics[task_label]:
            print("\t{}: {}".format(metric, task_metrics[task_label][metric]))

    print("Average across all tasks:")
    print(avg_metrics) 

    compute_curves(test_task_preds, args)
