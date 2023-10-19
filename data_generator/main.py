import argparse
import pandas as pd
import numpy as np
import random
import mlflow
import networkx as nx
from tqdm import tqdm
import pickle
import sys
from sklearn.cluster import SpectralClustering
import collections

from utils import draw_graph
from functions import get_functional_form
from task import simulate_task_data
from task import Task

sys.path.insert(0, 'metrics')
from causal_similarity import structural_hamming_distance, structural_intervention_distance, observational_distance, interventional_distance

"""Synthetic data generator

Provide a reference DAG in the form of an edge file that contains 
all the X (covariate) and T (intervention) variables that you want to 
include in the model. There must be exactly one Y (label) variable.
Label the covariates using the naming convention X1, X2, X3, ... and 
label the interventions using the naming convention T1, T2, T3, ...
"""

class Metadata:
    def __init__(self, args):
        # initialise the main reference CGM and the reference CGMs for each causal group
        self.args = args
        self.G_ref = nx.read_edgelist(args['ref_input'], delimiter=' ', nodetype=str, create_using=nx.DiGraph())
        self.A_ref = nx.to_numpy_array(self.G_ref)
        self.node_list = list(self.G_ref.nodes())
        self.X_vars = [v for v in self.node_list if v.startswith('X')]
        self.num_X_var = len(self.X_vars)
        self.T_vars = [v for v in self.node_list if v.startswith('T')]
        self.num_T_var = len(self.T_vars)
        self.functional_forms = self.get_functional_forms()
        self.group_refs = self.get_group_refs()
        self.draw_ref()


    def get_functional_forms(self):
        # functional form is dictionary of form {'func', 'param'}
        functional_forms = {'X{}'.format(X+1):get_functional_form(self.args) for X in range(self.num_X_var)}
        functional_forms.update({'T{}'.format(T+1):get_functional_form(self.args) for T in range(self.num_T_var)})
        return functional_forms
    

    def draw_ref(self):
        # draw the reference DAG
        refpath = draw_graph(self.G_ref, self.args['outprefix']+'_refmain')
        mlflow.log_artifact(refpath)


    def get_group_refs(self):
        # initialise the reference CGMs for each causal group
        group_refs = []
        for group in range(self.args['C']):
            print("Creating functional form for group {}".format(group))
            group_ref = Task(group, group, self.A_ref, self.G_ref, self.args['sigma_group'], self.functional_forms, self.args['eta_group'], self.node_list, self.args)
            group_refs.append(group_ref)
            # draw the reference DAG
            refpath = draw_graph(group_ref.G, self.args['outprefix']+'_ref{}'.format(group))
            mlflow.log_artifact(refpath)

        return group_refs


def simulate_dataset(metadata):
    """Simulate the data for N tasks and split each task into train/test sets.
    Store the data and the DAGs for each task. 
    """
    # require data to have at least one sample in each causal group
    all_groups = False
    while not all_groups:
        
        print("Simulating the dataset")
        df = pd.DataFrame()
        tasks = []
        in_meta_train = 1
        task_metadata_list = []
        in_group = [0]*metadata.args['C']

        total_num_tasks = metadata.args['N_train'] + metadata.args['N_val'] + metadata.args['N_test']
        for t in tqdm(range(total_num_tasks)):
            task, task_data = simulate_task_data(t, metadata)
            task_metadata_list.append({'task':task.task_number, 'ground_truth':task.C})
            in_group[task.C] = 1
            task_df = pd.DataFrame(task_data)
            task_df['task'] = t

            # split each task into train/test sets
            task_df['task_train'] = np.concatenate((np.ones(metadata.args['M_train']), np.zeros(len(task_df)-metadata.args['M_train'])))

            # assign task to meta-train, meta-val or meta-test sets
            if t < metadata.args['N_train']: in_meta_train = 0 # train
            elif t < metadata.args['N_train'] + metadata.args['N_val']: in_meta_train = 1 # validate
            else: in_meta_train = 2 # test
            task_df['meta_train'] = [in_meta_train]*len(task_df) 

            df = pd.concat([df, task_df])
            tasks.append(task)

        if sum(in_group)==metadata.args['C']:
            all_groups = True
    
    # calculate the causal similarities
    sim_df = get_causal_similarities(tasks)
    sim_df.to_csv('{}_causal_sim.csv'.format(metadata.args['outprefix']), index=None)

    # save the dataset
    df.to_csv('{}_data.csv'.format(metadata.args['outprefix']), index=None)
    mlflow.log_metric("num_samples", len(df))
    mlflow.log_metric("avg_num_samples_per_task", df.groupby('task').count()['Y'].mean())
    mlflow.log_metric("number_of_tasks", len(df['task'].unique()))
    
    # save the task DAGs
    with open('{}_graphs.pkl'.format(metadata.args['outprefix']), 'wb') as f:
        pickle.dump([nx.to_dict_of_dicts(task.G) for task in tasks], f)

    # save the metadata for ground truth causal groups
    metadata_df = metadata_df = pd.DataFrame(task_metadata_list)  
    metadata_df.to_csv('{}_task_metadata.csv'.format(args['outprefix']), index=None)


def get_causal_similarities(tasks):
    """Computes causal similarities between pairs of task using a variety of metrics
    """
    num_tasks = len(tasks)
    sim_data = []
    print("Calculating causal similarity metrics")
    indices = np.tril_indices(num_tasks)
    # all causal distances are symmetric - only compute once
    for t1, t2 in zip(indices[0], indices[1]):
        if t1 == t2:
            # distance between same task is 0
            sim_data.append({'task1':t1,'task2':t2,'SHD':0,'SID':0,'OD':0,'ID':0})
        else:
            task1 = tasks[t1]
            task2 = tasks[t2]
            shd = structural_hamming_distance(task1.G, task2.G)
            sid = structural_intervention_distance(task1.G, task2.G)
            odist = observational_distance(task1, task2)
            idist = interventional_distance(task1, task2)
            sim_data.append({'task1':t1,'task2':t2,'SHD':shd,'SID':sid,'OD':odist,'ID':idist})
            sim_data.append({'task1':t2,'task2':t1,'SHD':shd,'SID':sid,'OD':odist,'ID':idist})

    sim_df = pd.DataFrame(sim_data)
    
    return sim_df


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--outprefix', type=str, help='prefix for the output')
    parser.add_argument('--ref_input', type=str, default='example_edgefile.txt', help='prefix for the edgefile for the reference DAG (see example_edgefile.txt for an example)')
    parser.add_argument('--seed', type=int, default=1234, help='random seed for reproducibility')
    parser.add_argument('--N_train', type=int, default=10, help='number of training tasks')
    parser.add_argument('--N_val', type=int, default=10, help='number of validation tasks')
    parser.add_argument('--N_test', type=int, default=10, help='number of test tasks')
    parser.add_argument('--M_train', type=int, default=10, help='number of training (support) samples per task')
    parser.add_argument('--M_test', type=int, default=10, help='number of test (query) samples per task')
    parser.add_argument('--C', type=int, default=2, help='number of causal groups')
    parser.add_argument('--sigma_ref', type=float, default=1, help='variance of coefficient distributions in CAM (globally)')
    parser.add_argument('--sigma_group', type=float, default=0.1, help='variance of coefficient distributions in CAM for a causal group')
    parser.add_argument('--sigma_task', type=float, default=0.0001, help='variance of coefficient distributions in CAM for a single task')
    parser.add_argument('--sigma_noise', type=float, default=0.1, help='variance of noise distributions in CAM')
    parser.add_argument('--eta_group', type=float, default=0.6, help='bound on divergence from the reference DAG across the causal groups')
    parser.add_argument('--eta_task', type=float, default=0.05, help='bound on divergence from the reference DAG within a causal group')
    parser.add_argument('--linear', action='store_true', help='add this flag for a linear CAM, otherwise the CAM with be nonlinear')
    parser.add_argument('--binary', action='store_true', help='if true will make Y a binary (0 or 1) variable instead of continuous on [0,1]')

    args = vars(parser.parse_args())
    print(args)

    metadata = Metadata(args)
    
    np.random.seed(metadata.args['seed'])
    random.seed(metadata.args['seed'])
    
    simulate_dataset(metadata)