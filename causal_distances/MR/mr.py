import pandas as pd
import glob
import numpy as np
import os
from scipy.spatial.distance import cosine
import argparse

def main(mr_filepattern, outprefix):
    file_list = glob.glob(mr_filepattern)

    task_list = []

    wildcard_pos_start = mr_filepattern.rfind('*')
    wildcard_pos_end = len(mr_filepattern) - wildcard_pos_start - 1
    for path in file_list:
        matching_part = path[wildcard_pos_start:-wildcard_pos_end]
        task_list.append(matching_part)

    print(task_list)

    # get causal relationships from MR
    df_list = [pd.read_csv(file).dropna() for file in file_list]
    results_df = pd.concat(df_list, ignore_index=True)[['Endpoint','Exposure','IVW']]
    results_df = results_df.rename(columns={'Endpoint':'outcome', 'Exposure':'exposure', 'IVW':'value'})

    # exposures should include all longitudinal features; outcomes include all task codes
    results_df = results_df[(results_df['outcome'].isin(task_list))]

    # exposure = outcome should have minimum value
    results_df.loc[results_df['exposure'] == results_df['outcome'], 'value'] = 0

    exposure_list = results_df['exposure'].unique().tolist()

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
            v1 = task_vectors[t1].copy()
            v2 = task_vectors[t2].copy()
            # exclude t1 and t2 from task vectors, because their coefficients are not comparable
            v1[t_indices[t1]] = 0
            v2[t_indices[t1]] = 0
            v1[t_indices[t2]] = 0
            v2[t_indices[t2]] = 0
            dist_cosine = cosine(v1, v2)
            dist_data_cosine.append({'task1':t1, 'task2':t2, 'value':dist_cosine})

    # save the causal distance file
    dist_df = pd.DataFrame(dist_data_cosine)
    dist_df.to_csv(f'{outprefix}_causal_distance_MR_method.csv', index=None)
    results_df.to_csv(f'{outprefix}_causal_estimates_MR_method.csv', index=None)
    print(results_df)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process MR files and compute causal distances.")
    parser.add_argument("--mr_filepattern", type=str, required=True, help="Glob pattern for MR files")
    parser.add_argument("--outprefix", type=str, required=True, help="Output file prefix")
    
    args = parser.parse_args()
    
    main(args.mr_filepattern, args.outprefix)