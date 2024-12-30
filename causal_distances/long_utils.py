"""Utility functions for reading the longitudinal data input, used by several causal distance methods
"""

import pandas as pd
import numpy as np

# GLOBAL PARAMETERS
# update these for your data!
MAINDATA_FILEPATH = '/path/to/your/mainfile.csv'
LONGDATA_FILEPATH = '/path/to/your/longfile.csv'
METADATA_FILEPATH = '/path/to/your/metafile.csv '
YEAR_SPLIT = 0.5 # how to split the longitudinal data into exposures and outcomes, i.e., for Y years, the first Y*YEAR_SPLIT years are the exposure period and the rest are the outcome period

class Data:
    def __init__(self):
        # get training and target tasks from metafile
        meta_df = pd.read_csv(METADATA_FILEPATH)
        self.patient_col = meta_df[meta_df['column_type']=='patient_id']['column_name'].iloc[0]
        self.target_tasks = meta_df[meta_df['column_type']=='target_task']['column_name'].tolist()
        self.training_tasks = meta_df[meta_df['column_type']=='task_label']['column_name'].tolist()
        self.all_tasks = self.target_tasks + self.training_tasks
        self.covariate_predictors = meta_df[meta_df['column_type']=='predictor']['column_name'].tolist()

        # load longitudinal data and get years for exposure and outcome
        self.long_df = pd.read_csv(LONGDATA_FILEPATH, dtype={'ENDPOINT': str})
        self.all_years = sorted(self.long_df['EVENT_YEAR'].unique())
        self.exposure_list = self.long_df['ENDPOINT'].unique().tolist()
        print(self.exposure_list)
        num_exposure_years = int(len(self.all_years)*YEAR_SPLIT)
        self.exposure_years = self.all_years[0:num_exposure_years]
        self.outcome_years = self.all_years[num_exposure_years:]

        # get a list of patients in each task cohort (defined as task not observed in exposure years)
        self.main_df = pd.read_csv(MAINDATA_FILEPATH)
        all_patients = self.main_df[self.patient_col].tolist()
        self.main_df['PATIENT_ID'] = self.main_df[self.patient_col].tolist()
        self.cohorts = {} 
        for task in self.all_tasks:
            exclude_patients = self.long_df[(self.long_df['ENDPOINT']==task)&(self.long_df['EVENT_YEAR'].isin(self.exposure_years))]['PATIENT_ID'].tolist()
            self.cohorts[task] = [patient for patient in all_patients if patient not in exclude_patients]


    def aggregate_long_data(self, df, patient_list, with_covariates=False):
        # aggregate given longitudinal data df into a tabular format
        # using count of how many times a code is reported
        df = df.groupby(['PATIENT_ID','ENDPOINT']).count().reset_index()
        df = df.pivot(index='PATIENT_ID', columns='ENDPOINT', values='EVENT_YEAR')
        # merge with patient_list, to ensure all patients are included in order of patient_list
        if with_covariates:  patient_df = pd.merge(pd.DataFrame({'PATIENT_ID':patient_list}), self.main_df[[self.patient_col]+self.covariate_predictors].rename(columns={self.patient_col:'PATIENT_ID'}), how='left')
        else: patient_df = pd.DataFrame({'PATIENT_ID':patient_list})
        df = pd.merge(df, patient_df, on='PATIENT_ID', how='right')
        # assume patients with no longitudinal data is because no data was recorded for the selected endpoints and fill with zeros
        df = df.fillna(0)
        df.set_index('PATIENT_ID', inplace=True)
        return df
        

    def get_dataset(self, outcome_endpoint, with_covariates=False):
        # USED BY DAG (SCM) approach
        # creates a dataset in the format used for causal model fitting:
        # - each row corresponds to a patient observation (using the cohort for the outcome_endpoint)
        # - the first N columns correspond to each of the N tasks (during the observation period)
        # - the final (N+1)th column corresponds to the outcome endpoint (during the prediction period)
        # - this dataset can be used to test for relationships between the N tasks (exposures) and an outcome
        #
        # NOTE: outcome_endpoint can be either a string or list of strings
        # - if outcome_endpoint is a string: it corresponds to one specific task (e.g., a specific stroke endpoint) for the outcome
        # - if outcome_endpoint is a list: it corresponds to a list of tasks (e.g., a list of endpoints for stroke) and the outcome is constructed by combining these tasks

        if isinstance(outcome_endpoint, list):
            # multiple tasks in outcome
            # cohort corresponds to intersection of cohorts for each task
            task_cohorts = [self.cohorts[task] for task in outcome_endpoint]
            task_cohort = list(set.intersection(*map(set, task_cohorts)))
            outcome_endpoint_list = outcome_endpoint
        else:
            # single task in outcome
            task_cohort = self.cohorts[outcome_endpoint]
            outcome_endpoint_list = [outcome_endpoint]
        
        # get exposure data for patients in task cohort - note uses only the tasks as predictors
        data = self.long_df[(self.long_df['ENDPOINT'].isin(self.all_tasks))&(self.long_df['EVENT_YEAR'].isin(self.exposure_years))]
        data = self.aggregate_long_data(data, task_cohort, with_covariates)
        
        # check all tasks are in exposures (if a column is missing it means there is no longitudinal data available, raise warning and fill with zeros)
        for task in self.all_tasks:
            if task not in data:
                print(f"Warning: there is no longitudinal data available for exposure {task}, filling column with zeros")
                data[task] = 0

        if with_covariates: data = data[self.all_tasks + self.covariate_predictors]
        else: data = data[self.all_tasks] # make sure columns are in correct order

        # construct outcome variable (binary indicator)
        outcome_data = self.long_df[(self.long_df['ENDPOINT'].isin(outcome_endpoint_list))&(self.long_df['EVENT_YEAR'].isin(self.outcome_years))]
        outcome_data = self.aggregate_long_data(outcome_data, task_cohort)
        # if outcome constructed from multiple tasks, combine into one indicator variable
        outcome_variable = outcome_data.sum(axis=1)
        outcome_variable = (outcome_variable>0).astype(int) # make binary

        # merge exposures and outcomes into one dataset
        # first N columns are exposures and (N+1)th column is the outcome
        data.columns = ['exposure_'+col for col in data.columns]
        data['outcome'] = outcome_variable
        #assert len(data.keys())==len(self.all_tasks)+1, "The number of columns in the data should correspond to the number of exposures (tasks) + 1 for the outcome. This error is being raised because a column is missing from the data (e.g., perhaps because there is no longitudinal data available for an exposure or outcome)"
        return data

    def get_dataset_full(self, outcome_endpoint):
        # USED BY CHI2 APPROACH
        # similar to get_dataset() but doesn't re-split longitudinal data (uses labels from mainfile)

        if isinstance(outcome_endpoint, list):
            # multiple tasks in outcome
            # cohort corresponds to intersection of cohorts for each task
            task_cohorts = [self.cohorts[task] for task in outcome_endpoint]
            task_cohort = list(set.intersection(*map(set, task_cohorts)))
            outcome_endpoint_list = outcome_endpoint
        else:
            # single task in outcome
            task_cohort = self.cohorts[outcome_endpoint]
            outcome_endpoint_list = [outcome_endpoint]
        
        # get exposure data for patients in task cohort
        # NOTE: this variant includes all long endpoints
        data = self.long_df[self.long_df['EVENT_YEAR'].isin(self.exposure_years)]
        data = self.aggregate_long_data(data, task_cohort)
        
        # check all tasks are in exposures (if a column is missing it means there is no longitudinal data available, raise warning and fill with zeros)
        for task in self.all_tasks:
            if task not in data:
                print(f"Warning: check there is an ENDPOINT in the longitudinal file with the name matching task {task}, filling missing data column with zeros")
                data[task] = 0
        data = data[self.exposure_list] # make sure columns are in correct order

        # construct outcome variable (binary indicator)
        outcome_data = self.main_df[['PATIENT_ID'] + outcome_endpoint_list]
        outcome_data = pd.merge(outcome_data, pd.DataFrame({'PATIENT_ID':task_cohort}), on='PATIENT_ID', how='right')
        outcome_data = outcome_data[outcome_endpoint_list]
        outcome_data = outcome_data.fillna(0)
        # if outcome constructed from multiple tasks, combine into one indicator variable
        outcome_variable = outcome_data.sum(axis=1)
        outcome_variable = (outcome_variable>0).astype(int).tolist() # make binary

        # merge exposures and outcomes into one dataset
        # first N columns are exposures and (N+1)th column is the outcome
        data.columns = ['exposure_'+col for col in data.columns]
        data['outcome'] = outcome_variable
        return data