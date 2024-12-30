import pandas as pd
import numpy as np
from sklearn.feature_selection import chi2
from scipy.spatial.distance import cosine

import sys
sys.path.append('causal_distances')
from long_utils import Data

import argparse

def get_chi2(args):
    # initialize data class
    data = Data()
    task_list = data.all_tasks
    target_task_list = data.target_tasks
    exposure_list = data.exposure_list

    # chi2 results
    results_df = pd.DataFrame()

    for task in task_list:
        task_data = data.get_dataset_full(task)
        
        # run chi2 test
        X = task_data.drop(columns='outcome').values
        y = task_data['outcome']
        if sum(y)>0: print("{} Valid task with {} cases".format(task, sum(y)))
        else: print("{} Not valid task".format(task))
        chi2_stats, p_values = chi2(X, y)

        # format results
        task_df = pd.DataFrame({'exposure':exposure_list, 'outcome':task, 'value':chi2_stats})

        # if value is nan, assume that there is no effect (i.e., replace chi2 value with 0)
        task_df['value'] = task_df['value'].fillna(0)

        results_df = pd.concat([results_df, task_df])


    # exposures should include long features; outcomes include all task codes
    results_df = results_df[(results_df['exposure'].isin(exposure_list))&(results_df['outcome'].isin(task_list))]

    # exposure = outcome should have maximum value
    results_df.loc[results_df['exposure'] == results_df['outcome'], 'value'] = results_df['value'].max()*10

    # construct vectors of inferred relationships for each task
    task_vectors = {}
    for task in task_list:
        task_vector = [results_df[(results_df['outcome']==task)&(results_df['exposure']==exposure)]['value'].tolist()[0] for exposure in exposure_list]
        task_vectors[task] = np.array(task_vector)

    # compute distance between task vectors to estimate distance between each pair of tasks
    t_indices = dict(zip(exposure_list, range(len(exposure_list))))
    dist_data_cosine = []
    for t1 in task_list:
        for t2 in task_list:
            v1 = task_vectors[t1]
            v2 = task_vectors[t2]
            # exclude t1 and t2 from task vectors, because their coefficients are not comparable
            v1[t_indices[t1]] = 0
            v2[t_indices[t1]] = 0
            v1[t_indices[t2]] = 0
            v2[t_indices[t2]] = 0
            dist_cosine = cosine(v1, v2)
            dist_data_cosine.append({'task1':t1, 'task2':t2, 'value':dist_cosine})

    # # save the causal distance files
    dist_df = pd.DataFrame(dist_data_cosine)
    dist_df.to_csv('{}_causal_distance_BASELINE_method.csv'.format(args['outprefix']), index=None)
    results_df.to_csv('{}_causal_estimates_BASELINE_method.csv'.format(args['outprefix']), index=None)

    print(dist_df.head().to_string())
    print(results_df.head().to_string())


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--outprefix', type=str, help='prefix for the path to save the output files')
    args = vars(parser.parse_args())

    get_chi2(args)