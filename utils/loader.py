import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np

class TaskDataset(Dataset):
    def __init__(self, datapath="path/to/csv", dataset='train', train_test_split=True, has_causal_groups=False, causalpath=None, causaldist='ground_truth'):
        """
        Notes: 
        - data uses fixed train/val/test and support/query splits (to ensure consistency across experiments)
        - assumes each task has same number of samples (for simplicity) 
        - the baseline models assume the number of validation tasks = number of test tasks
        - assumes binary intervention variables
        
        Parameters - 
        datapath: path to data csv file
        dataset: one of train, val, test
        train_test_split: if true uses a support/query split
        has_causal_groups: if true retrieves the causal group assignments from the file at causalpath
        causalpath: path to csv file containing causal group assignments
        causaldist: name of causal distance to use
        """
        self.train_test_split = train_test_split
        self.has_causal_groups = has_causal_groups
        df = pd.read_csv(datapath)
        dataset_map = {'train':0, 'val':1, 'test':2}
        df = df[df['meta_train']==dataset_map[dataset]]
        self.num_tasks = len(df['task'].unique())
        features = list(df.drop(columns=['task','task_train','meta_train','Y']).keys())
        self.num_features = len(features)
        self.intervention_variables = [i for i,x in enumerate(features) if x.startswith('T')]
        self.intervention_weights, self.intervention_lists = self.get_intervention_weights(df, features)
        self.task_list = df['task'].unique()
        self.num_causal_groups = None
        if self.train_test_split:
            self.support_X = torch.tensor(np.array([df[(df['task']==t)&(df['task_train']==1)][features].values for t in self.task_list]), dtype=torch.float32)
            self.support_y = torch.tensor(np.array([df[(df['task']==t)&(df['task_train']==1)]['Y'].values for t in self.task_list]), dtype=torch.float32)
            self.query_X = torch.tensor(np.array([df[(df['task']==t)&(df['task_train']==0)][features].values for t in self.task_list]), dtype=torch.float32)
            self.query_y = torch.tensor(np.array([df[(df['task']==t)&(df['task_train']==0)]['Y'].values for t in self.task_list]), dtype=torch.float32)
        else:
            self.X = torch.tensor(np.array([df[df['task']==t][features].values for t in self.task_list]), dtype=torch.float32)
            self.y = torch.tensor(np.array([df[df['task']==t]['Y'].values for t in self.task_list]), dtype=torch.float32)
        if self.has_causal_groups:
            causal_df = pd.read_csv(causalpath)
            causal_map = dict(zip(causal_df['task'], causal_df[causaldist]))
            self.causal_groups = [int(causal_map[task]) for task in self.task_list]
            self.num_causal_groups = len(set(self.causal_groups))

    def __len__(self):
        return self.num_tasks
    
    def __getitem__(self, idx):
        if self.train_test_split and self.has_causal_groups:
            return self.support_X[idx], self.support_y[idx], self.query_X[idx], self.query_y[idx], self.causal_groups[idx]
        elif self.train_test_split:
            return self.support_X[idx], self.support_y[idx], self.query_X[idx], self.query_y[idx]
        else:
            return self.X[idx], self.y[idx]

    def get_intervention_weights(self, df, features):
        num_tasks = len(df['task'].unique())
        intervention_weights = {}
        intervention_variables = []
        intervention_lists = {}
        weight_sum = 0
        for i,v in enumerate(features):
            if v.startswith('T'):
                # proportion of tasks with intervention v
                intervention_weights[i] = (df.groupby('task').sum()[v] > 0).astype(int).sum() / num_tasks
                intervention_lists[i] = df[df[v]==1]['task'].unique().tolist() # list of tasks with v in their data
                weight_sum += intervention_weights[i]
                intervention_variables.append(v)
        # proportion of tasks with no intervention 
        intervention_weights[None] = len(df.loc[(df[intervention_variables] == 0).all(axis=1)]['task'].unique()) / num_tasks
        intervention_lists[None] = df[(df[intervention_variables] == 0).all(axis=1)]['task'].unique().tolist() # list of tasks with no intervention in their data
        weight_sum += intervention_weights[None]
        # a single task can have multiple interventions
        # therefore, rescale the values so that they sum to 1
        for key in intervention_weights: intervention_weights[key] = intervention_weights[key]/weight_sum
        return intervention_weights, intervention_lists


class DataModule():
    def __init__(self, datapath="path/to/csv", train_test_split=True, batch_size=1, shuffle=True, has_causal_groups=False, causalpath=None, causaldist='ground_truth'):
        """
        Parameters - 
        datapath: path to data csv file
        train_test_split: if true uses a support/query split
        batch_size: number of tasks to return per loaded batch (-1 for all data)
        shuffle: if true shuffles the order of the tasks
        has_causal_groups: if true retrieves the causal group assignments from the file at causalpath
        causalpath: path to csv file containing causal group assignments
        causaldist: name of causal distance to use
        """
        self.datapath = datapath
        self.train_test_split = train_test_split
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.has_causal_groups = has_causal_groups
        self.causalpath = causalpath
        self.causaldist = causaldist
        
    def train_dataloader(self):
        dataset = TaskDataset(datapath=self.datapath, dataset='train', train_test_split=self.train_test_split, has_causal_groups=self.has_causal_groups, causalpath=self.causalpath, causaldist=self.causaldist)
        if self.batch_size == -1: self.batch_size = len(dataset)
        dataloader = DataLoader(dataset=dataset, batch_size=self.batch_size, shuffle=self.shuffle)
        return dataloader

    def val_dataloader(self):
        dataset = TaskDataset(datapath=self.datapath, dataset='val', train_test_split=self.train_test_split, has_causal_groups=self.has_causal_groups, causalpath=self.causalpath, causaldist=self.causaldist)
        if self.batch_size == -1: self.batch_size = len(dataset)
        dataloader = DataLoader(dataset=dataset, batch_size=self.batch_size, shuffle=self.shuffle)
        return dataloader

    def test_dataloader(self):
        dataset = TaskDataset(datapath=self.datapath, dataset='test', train_test_split=self.train_test_split, has_causal_groups=self.has_causal_groups, causalpath=self.causalpath, causaldist=self.causaldist)
        if self.batch_size == -1: self.batch_size = len(dataset)
        dataloader = DataLoader(dataset=dataset, batch_size=self.batch_size, shuffle=self.shuffle)
        return dataloader
