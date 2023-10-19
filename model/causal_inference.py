import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
from sklearn.cluster import SpectralClustering
from pyemd import emd_samples
from scipy.stats import mode
import pandas as pd
import sys
from numpy.random import multivariate_normal
sys.path.insert(0, 'metrics')


def proxy_similarity_obs(task1_data, task2_data, task_index1=None, task_index2=None, intervention_variables=None, intervention_weights=None, args=None, task_meta_similarities=None, train_loader=None):
    """Proxy for observational similarity score
    """ 
    return emd_samples(task1_data, task2_data)


def proxy_similarity_int(task1_data, task2_data, task_index1, task_index2, intervention_variables, intervention_weights, args, task_meta_similarities=None, train_loader=None):
    """Proxy for interventional similarity score

    Notes:
    intervention_variables: a list that gives the column indices of the intervention variables in the data
    intervention_weights: a dictionary that specifies a weight for each intervention (should also include the None intervention)
    replace_missing_terms: if true, finds substitute tasks for calculating interventional distances, and otherwise excludes the these terms
    weighted_average: if true, calculates a weighted average using the provided intervention_weights, and otherwise uses non-weighted average
    uncertainty_penalty: if true, adds uncertainty estimates for where substitute tasks were used, and otherwise does not use uncertainty weights
    """ 
    # calculate observational distance for each intervention and take the average
    # (assumes interventions are sampled uniformly)
    obs_distances = []
    included_interventions = []
    uncertainty_adjustments = []
    for intervention in intervention_variables:
        samples1 = task1_data[np.where(task1_data[:,intervention]==1)[0]]
        samples2 = task2_data[np.where(task2_data[:,intervention]==1)[0]]
        if len(samples1) > 0 and len(samples2) > 0: 
            obs_distances.append(emd_samples(samples1, samples2))
            uncertainty_adjustments.append(1)
            included_interventions.append(None)
        elif args['replace_missing_terms']:
            # calculate obs_distance by substituting some other task/s
            new_task1_data, new_task2_data, uncertainty_weight = get_substitution_tasks(samples1, samples2, task1_data, task2_data, task_index1, task_index2, task_meta_similarities, intervention, train_loader)
            new_samples1 = new_task1_data[np.where(new_task1_data[:,intervention]==1)[0]]
            new_samples2 = new_task2_data[np.where(new_task2_data[:,intervention]==1)[0]]
            obs_distances.append(emd_samples(new_samples1, new_samples2)) 
            uncertainty_adjustments.append(uncertainty_weight)
            included_interventions.append(intervention)
    # also append a distance for the no intervention case
    samples1 = task1_data[(task1_data[:,intervention_variables]==0).all(axis=1)]
    samples2 = task2_data[(task2_data[:,intervention_variables]==0).all(axis=1)]
    if len(samples1) > 0 and len(samples2) > 0: 
        obs_distances.append(emd_samples(samples1, samples2))
        uncertainty_adjustments.append(1)
        included_interventions.append(None)
    elif args['replace_missing_terms']:
        intervention = None
        new_task1_data, new_task2_data, uncertainty_weight = get_substitution_tasks(samples1, samples2, task1_data, task2_data, task_index1, task_index2, task_meta_similarities, intervention, train_loader)
        new_samples1 = new_task1_data[(new_task1_data[:,intervention_variables]==0).all(axis=1)]
        new_samples2 = new_task2_data[(new_task2_data[:,intervention_variables]==0).all(axis=1)]
        obs_distances.append(emd_samples(new_samples1, new_samples2)) 
        uncertainty_adjustments.append(uncertainty_weight)
        included_interventions.append(None)
    # calculate (weighted?) average of all terms
    if len(obs_distances) > 0:
        weights = [1]*len(obs_distances) # default: equal weight for all terms
        if args['weighted_average']:
            weights = [intervention_weights[intervention] for intervention in included_interventions]
        if args['replace_missing_terms'] and args['uncertainty_penalty']: 
            # NOTE: can only use uncertainty estimates when replace_missing_terms=True
            weights = np.array(weights) * np.array(uncertainty_adjustments)
        return np.average(obs_distances, weights=weights)
    else:
        # fallback on observational distance if there are no interventional terms (e.g. because of no common interventions)
        return proxy_similarity_obs(task1_data, task2_data)


def get_substitution_tasks(samples1, samples2, task1_data, task2_data, task_index1, task_index2, task_meta_similarities, intervention, train_loader):
    num_tasks_to_substitute = 0
    
    if len(samples1)==0:
        num_tasks_to_substitute += 1
        # find a similar task to task_index1 with given intervention
        if intervention is not None: new_task1_index = task_meta_similarities[(task_meta_similarities[f'task_index1']==task_index1)&(task_meta_similarities[f'task_index2']!=task_index2)&(task_meta_similarities[f'task_index2_intervention{intervention}']==1)].sort_values(by='distance')['task_index2'].tolist()[0]
        # special case for none intervention
        else: new_task1_index = task_meta_similarities[(task_meta_similarities[f'task_index1']==task_index1)&(task_meta_similarities[f'task_index2']!=task_index2)&(task_meta_similarities[f'task_index2_interventionnone']==1)].sort_values(by='distance')['task_index2'].tolist()[0]
        new_task1_data = get_task_data(train_loader, new_task1_index)
    else:
        new_task1_index = task_index1
        new_task1_data = task1_data

    if len(samples2)==0: 
        num_tasks_to_substitute += 1
        # find a similar task to task_index2 with given intervention
        if intervention is not None: new_task2_index = task_meta_similarities[(task_meta_similarities[f'task_index2']==task_index2)&(task_meta_similarities[f'task_index1']!=new_task1_index)&(task_meta_similarities[f'task_index1_intervention{intervention}']==1)].sort_values(by='distance')['task_index1'].tolist()[0]
        # special case for none intervention
        else: new_task2_index = task_meta_similarities[(task_meta_similarities[f'task_index2']==task_index2)&(task_meta_similarities[f'task_index1']!=new_task1_index)&(task_meta_similarities[f'task_index1_interventionnone']==1)].sort_values(by='distance')['task_index1'].tolist()[0]
        new_task2_data = get_task_data(train_loader, new_task2_index)
    else:
        new_task2_data = task2_data

    uncertainty_weight_map = {0:1, 1:0.5, 2:0.25}
    uncertainty_weight = uncertainty_weight_map[num_tasks_to_substitute]

    return new_task1_data, new_task2_data, uncertainty_weight


def predict_causal_groups_for_new_data_unknown_CGMS(new_loader, train_loader, train_causal_groups, args, intervention_variables, intervention_weights, task_meta_similarities, task_parameters, init_model, predict_model):
    causal_groups = []

    train_task_numbers = train_loader.dataset.task_list
    new_task_numbers = new_loader.dataset.task_list

    if args['inference_type'] == 'interventional': 
        print("\t calculating interventional distance")
        proxy_similarity = proxy_similarity_int
    else:
        print("\t calculating observational distance")
        proxy_similarity = proxy_similarity_obs

    new_task = 0
    for X_spt, Y_spt, X_qry, Y_qry in new_loader:
        sim_data = []
        new_task_meta_similarities = [] # add new task to the similarities for training tasks
        new_task_id = new_loader.dataset.task_list[new_task] # the actual task id 

        if args['inference_type'] == 'interventional' and args['replace_missing_terms']:
            
            task_model, weight_mu, weight_rho, bias_mu, bias_rho = predict_model(init_model, X_spt, Y_spt, X_qry, Y_qry, args['base_learning_rate_initial'], args, return_parameters=True)
            mu1 = torch.concat([weight_mu, bias_mu]).numpy()
            rho1 = torch.concat([weight_rho, bias_rho]).numpy()
            sigma1 = np.log1p(np.exp(rho1))*np.identity(len(rho1))
            samples1 = multivariate_normal(mu1, sigma1, size=100)

            for train_task in range(len(train_task_numbers)):
                # calculate probabilistic distance
                mu2 = task_parameters['mu'][train_task]
                sigma2 = np.log1p(np.exp(task_parameters['rho'][train_task]))*np.identity(len(task_parameters['rho'][train_task])) # convert rho to sigma
                samples2 = multivariate_normal(mu2, sigma2, size=100)
                d = emd_samples(samples1, samples2)
                new_task_meta_similarities.append({'task_index1':new_task_id, 'task_index2':train_task, 'distance':d})
                new_task_meta_similarities.append({'task_index1':train_task, 'task_index2':new_task_id, 'distance':d})
            
            new_task_meta_similarities = pd.concat([task_meta_similarities, pd.DataFrame(new_task_meta_similarities)])

            for intervention in train_loader.dataset.intervention_lists:
                intervention_list = train_loader.dataset.intervention_lists[intervention] + new_loader.dataset.intervention_lists[intervention]
                if intervention is not None:
                    new_task_meta_similarities[f'task_index1_intervention{intervention}'] = new_task_meta_similarities['task_index1'].isin(intervention_list).astype(int)
                    new_task_meta_similarities[f'task_index2_intervention{intervention}'] = new_task_meta_similarities['task_index2'].isin(intervention_list).astype(int)
                else:
                    new_task_meta_similarities[f'task_index1_interventionnone'] = new_task_meta_similarities['task_index1'].isin(intervention_list).astype(int)
                    new_task_meta_similarities[f'task_index2_interventionnone'] = new_task_meta_similarities['task_index2'].isin(intervention_list).astype(int)
            
        for train_task in range(len(train_task_numbers)):
            new_task_data = get_task_data(new_loader, new_task)
            train_task_data = get_task_data(train_loader, train_task)
            proxy = proxy_similarity(new_task_data, train_task_data, new_task_id, train_task, intervention_variables, intervention_weights, args, new_task_meta_similarities, train_loader)
            sim_data.append({'task1':new_task, 'task2':train_task, 'causal_distance':proxy})
        # get all train tasks with min distance to new task
        sim_df = pd.DataFrame(sim_data)
        min_dist = sim_df.sort_values(by='causal_distance')['causal_distance'].min()
        min_tasks = sim_df[sim_df['causal_distance']==min_dist]['task2'].tolist()
        # get the clusters of these tasks and assign the one that comes up most frequently in the list
        # (if there are multiple values that tie, select one of the top options at random)
        groups_for_min_tasks = [train_causal_groups[t] for t in min_tasks]
        causal_assignment = mode(groups_for_min_tasks).mode
        causal_groups.append(causal_assignment)

        new_task += 1

    return causal_groups


def predict_causal_groups_for_new_data_known_CGMS(new_loader, train_loader, train_causal_groups, args):
    causal_groups = []

    print("\t calculating {} distance".format(args['causal_distance']))

    train_task_numbers = train_loader.dataset.task_list
    new_task_numbers = new_loader.dataset.task_list
    train_task_assignments = dict(zip(train_task_numbers, train_causal_groups))

    # use simfile for known causal models
    sim_df = pd.read_csv(args['simfile'])
    causal_distance = args['causal_distance']
    for new_task_number in new_task_numbers:
        # get all train tasks with min distance to new task
        min_dist = sim_df[(sim_df['task1']==new_task_number)&(sim_df['task2'].isin(train_task_numbers))].sort_values(by=causal_distance)[causal_distance].min()
        min_tasks = sim_df[(sim_df['task1']==new_task_number)&(sim_df['task2'].isin(train_task_numbers))&(sim_df[causal_distance]==min_dist)]['task2'].tolist()
        # get the clusters of these tasks and assign the one that comes up most frequently in the list
        # (if there are multiple values that tie, select one of the top options at random)
        groups_for_min_tasks = [train_task_assignments[t] for t in min_tasks]
        causal_assignment = mode(groups_for_min_tasks).mode
        causal_groups.append(causal_assignment)

    return causal_groups


def predict_causal_groups_for_new_data(new_loader, train_loader, train_causal_groups, args, intervention_variables, intervention_weights, task_meta_similarities, task_parameters, init_model, predict_model):
    """
    Predicts the causal groups for N' new tasks

    Returns the causal group assignments c1, c2, ..., cN'
    (in the order that the data is stored in the new_loader)
    """
    if args['unknown_causal_models']:
        causal_groups = predict_causal_groups_for_new_data_unknown_CGMS(new_loader, train_loader, train_causal_groups, args, intervention_variables, intervention_weights, task_meta_similarities, task_parameters, init_model, predict_model)
    else:
        causal_groups = predict_causal_groups_for_new_data_known_CGMS(new_loader, train_loader, train_causal_groups, args)

    return causal_groups


def get_task_data(loader, task_index, support_only=False):
    """Get task data for the task at the given index
    Returns a torch tensor where support/query and X/y data are combined together
    """
    if support_only:
        X_data = loader.dataset.support_X[task_index]
        Y_data = loader.dataset.support_y[task_index]
    else:
        X_data = torch.concat([loader.dataset.support_X[task_index], loader.dataset.query_X[task_index]])
        Y_data = torch.concat([loader.dataset.support_y[task_index], loader.dataset.query_y[task_index]])

    combined_data = torch.column_stack([X_data, Y_data]).detach().numpy()

    return combined_data


def predict_causal_distances(loader, proxy_similarity, intervention_variables, intervention_weights, task_meta_similarities, args):
    """
    Get the causal distances between the tasks in the loader

    Predicts the causal distances using the proxy measures

    Returns an NxN matrix of causal distances, where N is number of tasks in loader
    """
    num_tasks = len(loader)
    # build a distance matrix
    D = np.zeros((num_tasks, num_tasks))
    # distances are symmetric, so we only need to compute them once for each unordered pair
    indices = np.tril_indices(num_tasks)
    for t1, t2 in zip(indices[0], indices[1]):
        if t1 == t2: D[t1,t2] = 0 # distance between same task is 0
        else:
            task1_data = get_task_data(loader, t1)
            task2_data = get_task_data(loader, t2)
            proxy = proxy_similarity(task1_data, task2_data, t1, t2, intervention_variables, intervention_weights, args, task_meta_similarities, loader)
            D[t1,t2] = proxy
            D[t2,t1] = proxy

    # replace 0 valued distances with a small numerical value to avoid division by zero
    min_value = D[D>0].min()/10
    D[D==0] = min_value
    return D


def retrieve_causal_distances(simfile, loader, causal_distance):
    """
    Get the causal distances between the tasks in the loader

    NOTE assumes the simfile has been constructed so that the distances are **symmetric** and present for **all** ordered pairs

    Returns an NxN matrix of causal distances, where N is number of tasks in loader
    """
    sim_df = pd.read_csv(simfile)
    task_numbers = loader.dataset.task_list
    num_tasks = len(task_numbers)
    sim_df = sim_df[(sim_df['task1'].isin(task_numbers))&(sim_df['task2'].isin(task_numbers))][['task1','task2',causal_distance]]
    # build a distance matrix
    D = np.zeros((num_tasks, num_tasks))
    for idx, row in sim_df.iterrows(): D[int(row['task1']),int(row['task2'])] = float(row[causal_distance])
    # replace 0 valued distances with a small numerical value to avoid division by zero
    min_value = D[D>0].min()/10
    D[D==0] = min_value
    return D


def predict_causal_groups_for_training_data(D, num_causal_groups):
    """Applies spectral clustering to affinity matrix to predict causal groups

    NOTE: when evaluating against ground truth need to first align the group labels
    """
    A = 1/D # take the reciprocal of distance to get similarity
    spectral = SpectralClustering(num_causal_groups, affinity='precomputed')
    labels = spectral.fit_predict(A)
    return labels


def get_causal_groups(train_loader, args, intervention_variables, intervention_weights, task_meta_similarities):
    """
    Predicts the causal groups for the training dataset

    Returns the causal group assignments c1, c2, ..., cN for N tasks
    (in the order that the data is stored in the train_loader)
    """

    if args['unknown_causal_models']:
        if args['inference_type'] == 'interventional': 
            print("\t calculating interventional distance")
            proxy_similarity = proxy_similarity_int
        else: 
            print("\t calculating observational distance")
            proxy_similarity = proxy_similarity_obs

        # for unknown CGMs, predict D using proxy measures
        D = predict_causal_distances(train_loader, proxy_similarity, intervention_variables, intervention_weights, task_meta_similarities, args)

    else:
        # for known CGMs, retrieve D from precomputed causal distance files
        print("\t calculating {} distance".format(args['causal_distance']))
        D = retrieve_causal_distances(args['simfile'], train_loader, args['causal_distance'])
    
    print("\t applying spectral clustering")
    causal_groups = predict_causal_groups_for_training_data(D, args['num_groups'])

    return causal_groups
