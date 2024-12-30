from torch.utils.data import DataLoader
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
import torch
import numpy as np
from sklearn.utils import resample
from tqdm import tqdm
from torch.utils.data.sampler import WeightedRandomSampler

class DataModule():
    def __init__(self, tabular_datapath="path/to/csv", longitudinal_datapath="path/to/csv", metapath="path/to/csv", distpath="path/to/csv", method="bnn_baseline", learning_type='transductive', test_frac=0.5, query_frac=0.5, case_frac=0.25, n_kfold_splits=4, random_seed=42, data_type='sequence'):
        """
        Provide a dataset in CSV format and a metadata file that maps each column name in the dataset to their types.

        The data is split into test and non-test patient samples.
        The test data is withheld until the end for reporting the final evaluation metrics.
        The non-test samples are further split into training and validation sets using the k-fold strategy.
        Furthermore, the samples are also assigned to support and query sets.
        """
        # read the dataset and metadata
        self.tabular_df = pd.read_csv(tabular_datapath).sample(frac=1, random_state=random_seed) # shuffle the data
        self.data_type = data_type
        self.distpath = distpath
        if self.data_type == 'sequence':
            self.longitudinal_df = pd.read_csv(longitudinal_datapath, dtype={'ENDPOINT': str})
            self.longitudinal_df, self.long_years, self.long_endpoints = self.map_longitudinal_data(self.longitudinal_df)
            self.total_num_years = len(self.long_years)
            self.total_num_endpoints = len(self.long_endpoints)
        else:
            self.total_num_years = 0
            self.total_num_endpoints = 0

        print("Loaded {} longitudinal predictors for {} years".format(self.total_num_endpoints, self.total_num_years))

        metadata = pd.read_csv(metapath)
        
        # get column names for patient id, predictors and tasks
        self.patient_id = metadata[metadata['column_type']=='patient_id']['column_name'].tolist()[0] # assume there's exactly one patient_id column in the dataset
        self.predictors = metadata[metadata['column_type']=='predictor']['column_name'].tolist()
        print("Loaded {} tabular predictors".format(len(self.predictors)))

        if learning_type == 'transductive':
            self.task_names = metadata[metadata['column_type'].isin(['task_label','target_task'])]['column_name'].tolist()
        elif learning_type == 'inductive':
            self.task_names = metadata[metadata['column_type']=='task_label']['column_name'].tolist()

        self.task_name_map = dict(zip(range(len(self.task_names)), self.task_names))

        self.target_task_names = metadata[metadata['column_type']=='target_task']['column_name'].tolist()
        print("Loaded {} tasks (including {} target tasks)".format(len(self.task_names), len(self.target_task_names)))
        self.target_task_name_map = dict(zip(range(len(self.target_task_names)), self.target_task_names))
    
        # get cohorts for each task
        self.cohorts = {}
        for task in self.task_names + self.target_task_names:
            cohort_name = metadata[(metadata['task_cohort']==task)&(metadata['column_type']=='cohort')]['column_name'].tolist()[0] # uses the first cohort column found for the task
            self.cohorts[task] = self.tabular_df[self.tabular_df[cohort_name]==1][self.patient_id].tolist()

        self.avg_num_data = np.mean([len(self.cohorts[task])for task in self.cohorts.keys()])
        
        # randomly assign samples to (1) non-test and test sets and (2) support and query sets
        self.non_test_samples, self.test_samples = train_test_split(self.tabular_df[self.patient_id].to_numpy(), train_size=1-test_frac, random_state=random_seed)
        self.support_samples, self.query_samples = train_test_split(self.tabular_df[self.patient_id].to_numpy(), train_size=1-query_frac, random_state=random_seed+1)
        print("Using {} training/validation samples and {} test samples".format(len(self.non_test_samples), len(self.test_samples)))
        
        # create an iterator for k folds of non-test data into train/val sets
        self.kfold_iterator = self.k_fold(n_kfold_splits, random_seed)
        self.train_samples, self.val_samples = self.get_next_kfold()
        self.random_seed = random_seed
        self.n_kfold_splits= n_kfold_splits
        self.case_frac = case_frac
        self.query_frac = query_frac
        self.method = method

        # apply standard scalar to predictors
        scaler = StandardScaler().fit(self.tabular_df[self.tabular_df[self.patient_id].isin(self.non_test_samples)][self.predictors]) # fit to non-test samples
        self.tabular_df[self.predictors] = scaler.transform(self.tabular_df[self.predictors]) # transform all samples

        self.task_weights_target = None
        self.task_weights_all = None

    def map_longitudinal_data(self, long_df):
        """Assign indices to years and endpoints,
        to make it easier to construct 3D tensors
        """
        endpoints = long_df['ENDPOINT'].unique()
        endpoints.sort() # sort alphabetically for consistency (adjust if need different approach)
        start_year = long_df['EVENT_YEAR'].min()
        end_year = long_df['EVENT_YEAR'].max()
        years = np.arange(start_year, end_year+1)
        endpoint_map = dict(zip(endpoints, range(len(endpoints))))
        year_map = dict(zip(years, range(len(years))))
        long_df['ENDPOINT'] = long_df['ENDPOINT'].map(endpoint_map)
        long_df['EVENT_YEAR'] = long_df['EVENT_YEAR'].map(year_map)
        return long_df, years, endpoints

    def k_fold(self, n_kfold_splits, random_seed):
        """Get k-fold iterator for train/val splits of non-test data
        """
        kf = KFold(n_splits=n_kfold_splits, shuffle=True, random_state=random_seed)
        kfold_iterator = enumerate(kf.split(self.non_test_samples))
        return kfold_iterator
    
    def get_next_kfold(self):
        """Get the next train/val split from the k-fold iterator
        """
        i, (train_index, val_index) = next(self.kfold_iterator)
        train_samples = self.non_test_samples[train_index]
        val_samples = self.non_test_samples[val_index]
        return train_samples, val_samples

    def reset_kfold_iterator(self):
        """Reset the kfold iterator
        """
        # reset the k-fold iterator
        self.kfold_iterator = self.k_fold(self.n_kfold_splits, self.random_seed)
    
    def get_task_weights(self, eta_2):
        # load task distance scores
        df = pd.read_csv(self.distpath)
        df = df[(df['task1'].isin(self.task_names))&(df['task2'].isin(self.task_names))]
        # normalize between 0 and 1 and convert to similarity
        df['value'] = 1 - ((df['value']- df['value'].min()) / (df['value'].max() - df['value'].min()))
        # get task similarity scores for each of the tasks, averaged across the target tasks
        df = df[(df['task1'].isin(self.target_task_names))][['task2','value']].groupby('task2').mean().sort_values(by='value', ascending=False).fillna(0).reset_index()
        # apply modulating factor and convert to probabilities (sum to 1)
        df['value'] = df['value']**eta_2
        df['value'] = df['value'] / df['value'].sum()
        weights = dict(zip(df['task2'],df['value']))
        self.task_weights_all = [weights[t] for t in self.task_names]
        self.task_weights_target = [weights[t] for t in self.target_task_names]

    def get_taskloader(self, batchsize=1, shuffle=False, target_tasks=False, weighted=False):
        """Create a data loader that returns batchsize tasks per batch
        """
        if target_tasks: 
            task_list = self.target_task_names
            weights = self.task_weights_target
        else: 
            task_list = self.task_names
            weights = self.task_weights_all

        if weighted:
            num_task_samples = len(task_list)
            loader_tasks = list(WeightedRandomSampler(weights, num_task_samples, replacement=True))
            taskloader = DataLoader(loader_tasks, batch_size=batchsize, shuffle=shuffle)
        else:
            task_name_tensor = torch.tensor(range(len(task_list)), dtype=torch.long)
            taskloader = DataLoader(task_name_tensor, batch_size=batchsize, shuffle=shuffle)
        
        return taskloader

    def get_taskdata(self, task_label, eval=False, as_tensor=False, return_ids=False, device='cpu'):
        """Return the train / val / test data for the task with name given by task_label

        If eval is True, then only return train (non-test) / test data (i.e., no validation set needed)
        """
        if task_label is None:
            cohort = self.tabular_df[self.patient_id].tolist() # all patients
        else:
            cohort = self.cohorts[task_label]

        if eval:
            print("\t Getting patient ids")
            train_ids = self.tabular_df[(self.tabular_df[self.patient_id].isin(cohort))&(self.tabular_df[self.patient_id].isin(self.non_test_samples))][self.patient_id].tolist()
            test_ids = self.tabular_df[(self.tabular_df[self.patient_id].isin(cohort))&(self.tabular_df[self.patient_id].isin(self.test_samples))][self.patient_id].tolist()
            print("\t Getting tabular data")
            X_train = self.tabular_df[(self.tabular_df[self.patient_id].isin(train_ids))][self.predictors].values
            X_test = self.tabular_df[(self.tabular_df[self.patient_id].isin(test_ids))][self.predictors].values

            print("\t Getting task labels")
            if task_label is None:
                y_train = np.zeros(len(self.tabular_df[(self.tabular_df[self.patient_id].isin(train_ids))]))
                y_test = np.zeros(len(self.tabular_df[(self.tabular_df[self.patient_id].isin(test_ids))]))
            else:
                y_train = self.tabular_df[(self.tabular_df[self.patient_id].isin(train_ids))][task_label].to_numpy()
                y_test = self.tabular_df[(self.tabular_df[self.patient_id].isin(test_ids))][task_label].to_numpy()

            print("\t Getting sequence data")
            if self.data_type == 'sequence':
                X_long_train = self.get_longdata(train_ids)
                X_long_test = self.get_longdata(test_ids)
            else:
                # put placeholder if not using long data
                X_long_train = np.zeros_like(y_train)
                X_long_test = np.zeros_like(y_test)

            print("\t Converting to tensor format")
            if as_tensor:
                X_train = torch.Tensor(X_train).to(device)
                X_long_train = torch.Tensor(X_long_train).to(device)
                y_train = torch.Tensor(y_train).long().to(device)
                X_test = torch.Tensor(X_test).to(device)
                X_long_test = torch.Tensor(X_long_test).to(device)
                y_test = torch.Tensor(y_test).long().to(device)

            if return_ids:
                return X_train, X_long_train, y_train, train_ids, X_test, X_long_test, y_test, test_ids
            else:
                return X_train, X_long_train, y_train, X_test, X_long_test, y_test

        else:
            train_ids = self.tabular_df[(self.tabular_df[self.patient_id].isin(cohort))&(self.tabular_df[self.patient_id].isin(self.train_samples))][self.patient_id].tolist()
            val_ids = self.tabular_df[(self.tabular_df[self.patient_id].isin(cohort))&(self.tabular_df[self.patient_id].isin(self.val_samples))][self.patient_id].tolist()
            test_ids = self.tabular_df[(self.tabular_df[self.patient_id].isin(cohort))&(self.tabular_df[self.patient_id].isin(self.test_samples))][self.patient_id].tolist()
            X_train = self.tabular_df[(self.tabular_df[self.patient_id].isin(train_ids))][self.predictors].values
            X_val = self.tabular_df[(self.tabular_df[self.patient_id].isin(val_ids))][self.predictors].values
            X_test = self.tabular_df[(self.tabular_df[self.patient_id].isin(test_ids))][self.predictors].values

            if task_label is None:
                y_train = np.zeros(len(self.tabular_df[(self.tabular_df[self.patient_id].isin(train_ids))]))
                y_val = np.zeros(len(self.tabular_df[(self.tabular_df[self.patient_id].isin(val_ids))]))
                y_test = np.zeros(len(self.tabular_df[(self.tabular_df[self.patient_id].isin(test_ids))]))
            else:
                y_train = self.tabular_df[(self.tabular_df[self.patient_id].isin(train_ids))][task_label].to_numpy()
                y_val = self.tabular_df[(self.tabular_df[self.patient_id].isin(val_ids))][task_label].to_numpy()
                y_test = self.tabular_df[(self.tabular_df[self.patient_id].isin(test_ids))][task_label].to_numpy()

                if sum(y_train)==0 or sum(y_val)==0 or sum(y_test)==0:
                    print(f"WARNING: at least one of train/val/test sets in {task_label} contains no labelled samples")

            if self.data_type == 'sequence':
                X_long_train = self.get_longdata(train_ids)
                X_long_val = self.get_longdata(val_ids)
                X_long_test = self.get_longdata(test_ids)
            else:
                # put placeholder if not using long data
                X_long_train = np.zeros_like(y_train)
                X_long_val = self.get_longdata(y_val)
                X_long_test = np.zeros_like(y_test)

            if as_tensor:
                X_train = torch.Tensor(X_train).to(device)
                X_long_train = torch.Tensor(X_long_train).to(device)
                y_train = torch.Tensor(y_train).long().to(device)
                X_val = torch.Tensor(X_val).to(device)
                X_long_val = torch.Tensor(X_long_val).to(device)
                y_val = torch.Tensor(y_val).long().to(device)
                X_test = torch.Tensor(X_test).to(device)
                X_long_test = torch.Tensor(X_long_test).to(device)
                y_test = torch.Tensor(y_test).long().to(device)

            return X_train, X_long_train, y_train, X_val, X_long_val, y_val, X_test, X_long_test, y_test


    def get_longdata(self, patients):
        """Get longitudinal data in dense format for given patients. 

        Returns 3D tensor of shape (num_patients, num_years, num_endpoints)
        """
        has_repeated_patients = len(patients) != len(set(patients))

        if has_repeated_patients:
            
            # slower approach but accounts for same patient appearing multiple times in list

            long_data = self.longitudinal_df[(self.longitudinal_df['PATIENT_ID'].isin(patients))]

            data = pd.DataFrame()
            # map patient ids to indices (in order of batch)
            # construct the data patient by patient so we can account for the duplicates (due to resampling) that need unique indices
            for i in range(len(patients)):
                tmp_data = long_data[long_data['PATIENT_ID']==patients[i]]
                tmp_data.loc[:, 'PATIENT_ID'] = i
                data = pd.concat([data, tmp_data])
            
            # construct a 3D tensor from a sparse tensor torch.sparse_coo_tensor(indices, values, ...) where non-zero values are specified by indices and values
            indices = [data['PATIENT_ID'].tolist(), data['EVENT_YEAR'].tolist(), data['ENDPOINT'].tolist()]
            values = [1]*len(data)
            x_longitudinal = torch.sparse_coo_tensor(indices, values, (len(patients), self.total_num_years, self.total_num_endpoints), dtype=torch.float32)
            x_longitudinal = x_longitudinal.to_dense()
            return x_longitudinal

        else:

            # significantly faster but doesn't allow for same patient appearing multiple times in list

            # filter the longitudinal data for the given patients
            long_data = self.longitudinal_df[self.longitudinal_df['PATIENT_ID'].isin(patients)]

            # create a dictionary to map patient IDs to their index in the output tensor
            patient_to_index = {patient: index for index, patient in enumerate(patients)}

            # create the sparse tensor indices and values
            patient_indices = long_data['PATIENT_ID'].map(patient_to_index).values
            year_indices = long_data['EVENT_YEAR'].values
            endpoint_indices = long_data['ENDPOINT'].values
            
            indices = np.stack([patient_indices, year_indices, endpoint_indices])
            indices = torch.from_numpy(indices)
            values = torch.ones(len(long_data), dtype=torch.float32)

            # create the sparse tensor
            x_longitudinal = torch.sparse_coo_tensor(
                indices, 
                values, 
                size=(len(patients), self.total_num_years, self.total_num_endpoints),
                dtype=torch.float32
            )

            # convert to dense tensor
            x_longitudinal = x_longitudinal.to_dense()
            return x_longitudinal
    
    
    def sample_supportquery_data(self, task_label, num_samples_task, sample_seed=None, eval=False, as_tensor=False, device='cpu', dataset='train'):
        """Samples support and query sets for task with task_label

        Give a unique sample_seed for each training epoch to get different samples of task
        """
        # get training data for task

        num_support_samples_task = int(num_samples_task*(1-self.query_frac))
        num_query_samples_task = num_samples_task - num_support_samples_task
        
        cohort = self.cohorts[task_label]

        if eval: 
            # no validation set needed
            train_samples = self.non_test_samples 
        else: 
            if dataset == 'train':
                train_samples = self.train_samples
            elif dataset == 'val':
                train_samples = self.val_samples
            elif dataset == 'test':
                train_samples = self.test_samples
        
        random_state = self.random_seed+sample_seed

        # select patient ids for support/query sets

        if dataset == 'val':

            # no case/control balancing is applied

            potential_support_cases = self.tabular_df[(self.tabular_df[self.patient_id].isin(cohort)) 
                                        &(self.tabular_df[self.patient_id].isin(train_samples))
                                        &(self.tabular_df[self.patient_id].isin(self.support_samples))
                                        &(self.tabular_df[task_label]==1)]
            
            if len(potential_support_cases) < int(num_support_samples_task*self.case_frac):
                support_case_ids = potential_support_cases[self.patient_id].tolist()
            else:
                support_case_ids = resample(potential_support_cases[self.patient_id], 
                                        n_samples=int(num_support_samples_task*self.case_frac), 
                                        replace=True, random_state=random_state).tolist()

            potential_query_cases = self.tabular_df[(self.tabular_df[self.patient_id].isin(cohort)) 
                                        &(self.tabular_df[self.patient_id].isin(train_samples))
                                        &(self.tabular_df[self.patient_id].isin(self.query_samples))
                                        &(self.tabular_df[task_label]==1)]

            if len(potential_query_cases) < int(num_query_samples_task*self.case_frac):
                query_case_ids = potential_query_cases[self.patient_id].tolist()
            else:
                query_case_ids = resample(potential_query_cases[self.patient_id], 
                                            n_samples=int(num_query_samples_task*self.case_frac), 
                                            replace=True, random_state=random_state).tolist()               

            support_control_ids = resample(self.tabular_df[(self.tabular_df[self.patient_id].isin(cohort)) 
                                        &(self.tabular_df[self.patient_id].isin(train_samples))
                                        &(self.tabular_df[self.patient_id].isin(self.support_samples))
                                        &(self.tabular_df[task_label]==0)][self.patient_id], 
                                        n_samples=num_support_samples_task-len(support_case_ids), 
                                        replace=True, random_state=random_state).tolist()
            
            query_control_ids = resample(self.tabular_df[(self.tabular_df[self.patient_id].isin(cohort)) 
                                        &(self.tabular_df[self.patient_id].isin(train_samples))
                                        &(self.tabular_df[self.patient_id].isin(self.query_samples))
                                        &(self.tabular_df[task_label]==0)][self.patient_id], 
                                        n_samples=num_query_samples_task-len(query_case_ids), 
                                        replace=True, random_state=random_state).tolist()
        
        else:

            # case/control balancing is applied in this sampling because class imbalance is a problem for ML with medical datasets

            # allows for sampling with replacement (in the event there are fewer cases than needed it will sample the same data multiple times)

            support_case_ids = resample(self.tabular_df[(self.tabular_df[self.patient_id].isin(cohort)) 
                                        &(self.tabular_df[self.patient_id].isin(train_samples))
                                        &(self.tabular_df[self.patient_id].isin(self.support_samples))
                                        &(self.tabular_df[task_label]==1)][self.patient_id], 
                                        n_samples=int(num_support_samples_task*self.case_frac), 
                                        replace=True, random_state=random_state).tolist()

            query_case_ids = resample(self.tabular_df[(self.tabular_df[self.patient_id].isin(cohort)) 
                                        &(self.tabular_df[self.patient_id].isin(train_samples))
                                        &(self.tabular_df[self.patient_id].isin(self.query_samples))
                                        &(self.tabular_df[task_label]==1)][self.patient_id], 
                                        n_samples=int(num_query_samples_task*self.case_frac), 
                                        replace=True, random_state=random_state).tolist()               

            support_control_ids = resample(self.tabular_df[(self.tabular_df[self.patient_id].isin(cohort)) 
                                        &(self.tabular_df[self.patient_id].isin(train_samples))
                                        &(self.tabular_df[self.patient_id].isin(self.support_samples))
                                        &(self.tabular_df[task_label]==0)][self.patient_id], 
                                        n_samples=num_support_samples_task-len(support_case_ids), 
                                        replace=True, random_state=random_state).tolist()
            
            query_control_ids = resample(self.tabular_df[(self.tabular_df[self.patient_id].isin(cohort)) 
                                        &(self.tabular_df[self.patient_id].isin(train_samples))
                                        &(self.tabular_df[self.patient_id].isin(self.query_samples))
                                        &(self.tabular_df[task_label]==0)][self.patient_id], 
                                        n_samples=num_query_samples_task-len(query_case_ids), 
                                        replace=True, random_state=random_state).tolist()

        # combine the control/case samples and shuffle the order
        support_ids = support_case_ids + support_control_ids
        query_ids = query_case_ids + query_control_ids
        np.random.shuffle(support_ids)
        np.random.shuffle(query_ids)
        
        X_tab_spt = self.tabular_df.set_index(self.patient_id).loc[support_ids][self.predictors].values
        y_spt = self.tabular_df.set_index(self.patient_id).loc[support_ids][task_label].to_numpy()
        X_tab_qry = self.tabular_df.set_index(self.patient_id).loc[query_ids][self.predictors].values
        y_qry = self.tabular_df.set_index(self.patient_id).loc[query_ids][task_label].to_numpy()
        
        # can be used to adjust relative importance (1 if not using)
        imp_spt = np.ones_like(y_spt)
        imp_qry = np.ones_like(y_qry)

        if self.data_type == 'sequence':
            X_long_spt = self.get_longdata(support_ids)
            X_long_qry = self.get_longdata(query_ids)
        else:
            # put placeholder if not using long data
            X_long_spt = np.zeros_like(y_spt)
            X_long_qry = np.zeros_like(y_qry)
        
        if sum(y_spt)==0 or sum(y_qry)==0:
            print(f"WARNING: at least one of spt/qry sets in {task_label} contains no labelled samples")

        if as_tensor:
            return torch.Tensor(X_tab_spt).to(device), torch.Tensor(X_tab_qry).to(device), torch.Tensor(X_long_spt).to(device), torch.Tensor(X_long_qry).to(device), torch.Tensor(y_spt).long().to(device), torch.Tensor(y_qry).long().to(device), torch.Tensor(imp_spt).long().to(device), torch.Tensor(imp_qry).long().to(device)

        return X_tab_spt, X_tab_qry, X_long_spt, X_long_qry, y_spt, y_qry, imp_spt, imp_qry