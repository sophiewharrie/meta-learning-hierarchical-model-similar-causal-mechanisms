import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np

"""A general data loading utility that can be used across all methods and baselines
"""

class TaskDataset(Dataset):
    def __init__(self, datapath="path/to/csv", dataset='train', train_test_split=True, has_causal_groups=False, causalpath=None, causaldist='ground_truth', interv_mask=False, intervpath="path/to/intervmask"):
        """
        Some methods/baselines will only work if following hold: Data uses fixed train/val/test and support/query splits (to ensure consistency across experiments, setup different splits as separate experiments). Assumes each task has same number of samples and the baseline models assume the number of validation tasks = number of test tasks

        Parameters - 
        datapath: path to data csv file
        dataset: one of train, val, test
        train_test_split: if true uses a support/query split
        has_causal_groups: if true retrieves the causal group assignments from the file at causalpath
        causalpath: path to csv file containing causal group assignments
        causaldist: name of causal distance to use
        interv_mask: true if applying an interventional mask
        intervpath: path to the intervention mask data
        """
        self.train_test_split = train_test_split
        self.has_causal_groups = has_causal_groups
        df = pd.read_csv(datapath)
        dataset_map = {'train':0, 'val':1, 'test':2}
        df = df[df['meta_train']==dataset_map[dataset]]
        self.num_tasks = len(df['task'].unique())
        self.features = list(df.drop(columns=['task','task_train','meta_train','Y']).keys())
        self.num_features = len(self.features)
        self.task_list = df['task'].unique()
        self.num_causal_groups = None
        self.interv_mask = interv_mask
        if self.interv_mask and intervpath!='ignore': df_mask = pd.read_csv(intervpath)

        if self.train_test_split:
            self.support_X = torch.tensor(np.array([df[(df['task']==t)&(df['task_train']==1)][self.features].values for t in self.task_list]), dtype=torch.float32)
            self.support_y = torch.tensor(np.array([df[(df['task']==t)&(df['task_train']==1)]['Y'].values for t in self.task_list]), dtype=torch.float32)
            self.query_X = torch.tensor(np.array([df[(df['task']==t)&(df['task_train']==0)][self.features].values for t in self.task_list]), dtype=torch.float32)
            self.query_y = torch.tensor(np.array([df[(df['task']==t)&(df['task_train']==0)]['Y'].values for t in self.task_list]), dtype=torch.float32)
        else:
            self.X = torch.tensor(np.array([df[df['task']==t][self.features].values for t in self.task_list]), dtype=torch.float32)
            self.y = torch.tensor(np.array([df[df['task']==t]['Y'].values for t in self.task_list]), dtype=torch.float32)
            
        if self.interv_mask and intervpath!='ignore':
            self.interv_mask_support = torch.tensor(np.array([df_mask[(df_mask['task']==t)&(df_mask['task_train']==1)][self.features + ['Y']].values for t in self.task_list]), dtype=torch.float32)
            self.interv_mask_query = torch.tensor(np.array([df_mask[(df_mask['task']==t)&(df_mask['task_train']==0)][self.features + ['Y']].values for t in self.task_list]), dtype=torch.float32)
        elif self.interv_mask:
            # placeholder with all 0s, won't be used by model
            self.interv_mask_support = torch.zeros((self.support_X.shape[0],self.support_X.shape[1],self.support_X.shape[2]+1))
            self.interv_mask_query = torch.zeros((self.query_X.shape[0],self.query_X.shape[1],self.query_X.shape[2]+1))

        if self.has_causal_groups and causalpath is not None:
            causal_df = pd.read_csv(causalpath)
            causal_map = dict(zip(causal_df['task'], causal_df[causaldist]))
            self.causal_groups = [int(causal_map[task]) for task in self.task_list]
            self.num_causal_groups = len(set(self.causal_groups))
        elif self.has_causal_groups:
            self.causal_groups = [None for task in self.task_list]
            self.num_causal_groups = None

    def __len__(self):
        return self.num_tasks
    
    def __getitem__(self, idx):
        if self.train_test_split and self.interv_mask:
            return self.support_X[idx], self.support_y[idx], self.query_X[idx], self.query_y[idx], self.interv_mask_support[idx], self.interv_mask_query[idx]
        elif self.train_test_split:
            return self.support_X[idx], self.support_y[idx], self.query_X[idx], self.query_y[idx]
        else:
            return self.X[idx], self.y[idx]


class DataModule():
    def __init__(self, datapath="path/to/csv", train_test_split=True, batch_size=1, shuffle=True, has_causal_groups=False, causalpath=None, causaldist='ground_truth', interv_mask=False, intervpath="path/to/intervmask"):
        """
        Parameters - 
        datapath: path to data csv file
        dataset: one of train, val, test
        train_test_split: if true uses a support/query split
        has_causal_groups: if true retrieves the causal group assignments from the file at causalpath
        causalpath: path to csv file containing causal group assignments
        causaldist: name of causal distance to use
        interv_mask: true if applying an interventional mask
        intervpath: path to the intervention mask data
        """
        self.datapath = datapath
        self.train_test_split = train_test_split
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.has_causal_groups = has_causal_groups
        self.causalpath = causalpath
        self.causaldist = causaldist
        self.interv_mask = interv_mask
        self.intervpath = intervpath
        
    def train_dataloader(self):
        dataset = TaskDataset(datapath=self.datapath, dataset='train', train_test_split=self.train_test_split, has_causal_groups=self.has_causal_groups, causalpath=self.causalpath, causaldist=self.causaldist, interv_mask=self.interv_mask, intervpath=self.intervpath)
        if self.batch_size == -1: self.batch_size = len(dataset)
        dataloader = DataLoader(dataset=dataset, batch_size=self.batch_size, shuffle=self.shuffle)
        return dataloader

    def val_dataloader(self):
        dataset = TaskDataset(datapath=self.datapath, dataset='val', train_test_split=self.train_test_split, has_causal_groups=self.has_causal_groups, causalpath=self.causalpath, causaldist=self.causaldist, interv_mask=self.interv_mask, intervpath=self.intervpath)
        if self.batch_size == -1: self.batch_size = len(dataset)
        dataloader = DataLoader(dataset=dataset, batch_size=self.batch_size, shuffle=self.shuffle)
        return dataloader

    def test_dataloader(self):
        dataset = TaskDataset(datapath=self.datapath, dataset='test', train_test_split=self.train_test_split, has_causal_groups=self.has_causal_groups, causalpath=self.causalpath, causaldist=self.causaldist, interv_mask=self.interv_mask, intervpath=self.intervpath)
        if self.batch_size == -1: self.batch_size = len(dataset)
        dataloader = DataLoader(dataset=dataset, batch_size=self.batch_size, shuffle=self.shuffle)
        return dataloader


def get_task_data(loader, task_index=None, support_only=False):
    """Get task data for the task at the given index
    Returns a torch tensor where support/query and X/y data are combined together
    """
    if task_index is not None:
        if support_only:
            X_data = loader.dataset.support_X[task_index]
            Y_data = loader.dataset.support_y[task_index]
        else:
            X_data = torch.concat([loader.dataset.support_X[task_index], loader.dataset.query_X[task_index]])
            Y_data = torch.concat([loader.dataset.support_y[task_index], loader.dataset.query_y[task_index]])
    else:
        if support_only:
            X_data = loader.dataset.support_X
            X_data = X_data.reshape(-1,X_data.shape[2])
            Y_data = loader.dataset.support_y
            Y_data = Y_data.reshape(-1,)
        else:
            X_data = torch.concat([loader.dataset.support_X.reshape(-1,loader.dataset.support_X.shape[2]), loader.dataset.query_X.reshape(-1,loader.dataset.query_X.shape[2])])
            Y_data = torch.concat([loader.dataset.support_y.reshape(-1), loader.dataset.query_y.reshape(-1)])   

    combined_data = torch.column_stack([X_data, Y_data]).detach().numpy()

    if loader.dataset.interv_mask:
        combined_interv_mask = torch.concat([torch.concat([loader.dataset.interv_mask_support[i] for i in range(loader.dataset.interv_mask_support.shape[0])], dim=0), torch.concat([loader.dataset.interv_mask_query[i] for i in range(loader.dataset.interv_mask_query.shape[0])], dim=0)]).numpy()
        assert combined_data.shape == combined_interv_mask.shape
        return combined_data, combined_interv_mask
    else:
        return combined_data