"""Utility functions that can be used to compute F1 scores for accuracy of causal group assignments
"""

import collections
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score


def match_ground_truth_label(labels, ground_truth_labels, num_causal_groups):
    """Match the cluster labels, according to which ground truth label is most prevalent in each (inferred) cluster

    Note: this works best when the cluster assignment is obvious (because the inferred labels are mostly correct)
    """
    # calculate how much of each ground truth label in each inferred cluster
    freq_data = []
    for pred_label in range(num_causal_groups):
        count_gt = ground_truth_labels[np.where(labels==pred_label)] 
        freq_gt = collections.Counter(count_gt)
        for gt_label in freq_gt:
            freq_data.append({'pred':pred_label, 'gt':gt_label, 'count':freq_gt[gt_label], 'freq':freq_gt[gt_label]/len(count_gt)})

    # match a ground truth label to each inferred cluster,
    # based on the most common label in a cluster
    correction = np.zeros(num_causal_groups)
    cluster_df = pd.DataFrame(freq_data).sort_values(by='freq', ascending=False)
    assigned_gt = [] 
    assigned_pred = []
    for idx, row in cluster_df.iterrows():
        # check if ground truth or inferred label have already been assigned 
        # (give priority to cluster with highest % of a ground truth label)
        if row['gt'] not in assigned_gt and row['pred'] not in assigned_pred: 
            correction[int(row['pred'])] = int(row['gt'])
            assigned_gt.append(row['gt'])
            assigned_pred.append(row['pred'])

    return correction


def get_f1_scores(causalassignment_final_path, groundtruth_path, num_train_tasks=200):
    """
    Parameters are the filepaths to the files storing the causal assignments and the ground truth labels
    
    Returns 3 scores:
    f1_init_train: corresponds to causal assignments determined for training data during pre-training/initialisation phases
    f1_final_train: corresponds to causal assignments determined for training data main training phase
    f1_final_val_test: corresponds to causal assignments predicted for new tasks not used in training
    """
    
    pred_groups = np.array(pd.read_csv(causalassignment_final_path)['predicted_groups'])
    gt_groups = np.array(pd.read_csv(groundtruth_path)['ground_truth'])
    num_groups = len(set(gt_groups))

    correction = match_ground_truth_label(pred_groups[0:num_train_tasks], gt_groups[0:num_train_tasks], num_groups)
    pred_groups = np.choose(pred_groups,correction)

    f1_final_train = f1_score(gt_groups[0:num_train_tasks], pred_groups[0:num_train_tasks], average='macro')
    f1_final_val_test = f1_score(gt_groups[num_train_tasks:], pred_groups[num_train_tasks:], average='macro')

    return f1_final_train, f1_final_val_test

