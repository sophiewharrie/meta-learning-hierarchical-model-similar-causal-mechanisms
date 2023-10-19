import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.nn import Sequential
from collections import Counter
import copy
import pandas as pd
from numpy.random import multivariate_normal
from pyemd import emd_samples

from blitz.modules import BayesianLinear
from blitz.utils import variational_estimator
from prior import PriorWeightDistributionMVN
from causal_inference import get_causal_groups

import higher

# adds decorator to nn.Module with utilities for variational inference
Sequential = variational_estimator(Sequential)

def initial_model(train_loader, val_loader, args, intervention_variables, intervention_weights):
    """
    Causal groups is a list of length num_tasks that gives the causal group assignment for each task
    """
    # train initial model to learn priors for each cluster
    init_model = train_2level_model(train_loader, val_loader, args['num_initial_epochs'], args['meta_learning_rate_initial'], args['base_learning_rate_initial'], args)
    
    causal_groups = None
    task_meta_similarities = None
    task_parameters=None
    if args['unknown_causal_models'] or args['causal_distance']!='ground_truth':

        if args['unknown_causal_models'] and args['replace_missing_terms']:
            print("Computing task similarities in the meta-parameter space")
            # get parameters of base learners for each task
            task_parameters = {'mu':[], 'rho':[]}
            for X_spt, Y_spt, X_qry, Y_qry in train_loader:
                task_model, weight_mu, weight_rho, bias_mu, bias_rho = predict_model(init_model, X_spt, Y_spt, X_qry, Y_qry, args['base_learning_rate_initial'], args, return_parameters=True)
                task_parameters['mu'].append(torch.concat([weight_mu, bias_mu]).numpy())
                task_parameters['rho'].append(torch.concat([weight_rho, bias_rho]).numpy())
            
            # compute similarities between pairs of tasks in the meta-parameter space
            task_meta_similarities = []
            num_tasks = len(train_loader)
            indices = np.tril_indices(num_tasks)
            for task_index1, task_index2 in zip(indices[0], indices[1]):
                if task_index1==task_index2: task_meta_similarities.append({'task_index1':task_index1, 'task_index2':task_index2, 'distance':0})
                else:
                    # calculate probabilistic distance
                    mu1 = task_parameters['mu'][task_index1]
                    sigma1 = np.log1p(np.exp(task_parameters['rho'][task_index1]))*np.identity(len(task_parameters['rho'][task_index1])) # convert rho to sigma
                    samples1 = multivariate_normal(mu1, sigma1, size=100)
                    mu2 = task_parameters['mu'][task_index2]
                    sigma2 = np.log1p(np.exp(task_parameters['rho'][task_index2]))*np.identity(len(task_parameters['rho'][task_index2])) # convert rho to sigma
                    samples2 = multivariate_normal(mu2, sigma2, size=100)
                    d = emd_samples(samples1, samples2)
                    task_meta_similarities.append({'task_index1':task_index1, 'task_index2':task_index2, 'distance':d})
                    task_meta_similarities.append({'task_index1':task_index2, 'task_index2':task_index1, 'distance':d})
            
            # transform into a dataframe for easy querying
            task_meta_similarities = pd.DataFrame(task_meta_similarities)
            for intervention in train_loader.dataset.intervention_lists:
                if intervention is not None:
                    task_meta_similarities[f'task_index1_intervention{intervention}'] = task_meta_similarities['task_index1'].isin(train_loader.dataset.intervention_lists[intervention]).astype(int)
                    task_meta_similarities[f'task_index2_intervention{intervention}'] = task_meta_similarities['task_index2'].isin(train_loader.dataset.intervention_lists[intervention]).astype(int)
                else:
                    task_meta_similarities[f'task_index1_interventionnone'] = task_meta_similarities['task_index1'].isin(train_loader.dataset.intervention_lists[intervention]).astype(int)
                    task_meta_similarities[f'task_index2_interventionnone'] = task_meta_similarities['task_index2'].isin(train_loader.dataset.intervention_lists[intervention]).astype(int)
            
        print("Inferring the causal groups for the training dataset")
        causal_groups = get_causal_groups(train_loader, args, intervention_variables, intervention_weights, task_meta_similarities)
    
    return init_model, causal_groups, task_meta_similarities, task_parameters


def main_model(train_loader, val_loader, args, init_model=None):
    models, total_num_epochs = train_3level_model(init_model, train_loader, val_loader, args['num_main_epochs'], args['meta_learning_rate_global_main'], args['meta_learning_rate_group_main'], args['base_learning_rate_main'], args)
    return models, total_num_epochs


def train_3level_model(init_model, train_loader, val_loader, num_epochs, meta_learning_rate_global, meta_learning_rate_group, base_learning_rate, args):
    print(f"Training model for {num_epochs} epochs with base learning rate {base_learning_rate} and meta learning rates {meta_learning_rate_global} (global) and {meta_learning_rate_group} (causal group)")
    num_tasks = len(train_loader)
    num_tasks_per_group = Counter(train_loader.dataset.causal_groups)

    early_stopping = EarlyStopping(patience=args['patience'])
    
    global_model = copy.deepcopy(init_model)
    group_models = [copy.deepcopy(init_model) for c in range(args['num_groups'])]

    # update the prior of the group models with the variational distribution from the global model
    for model_idx in range(len(group_models)):
        group_model = group_models[model_idx]
        group_models[model_idx] = update_model_priors(group_model, global_model)

    global_model.train()
    for group_model in group_models: group_model.train()
    criterion = torch.nn.MSELoss()
    meta_opt_global = optim.Adam(global_model.parameters(), lr=meta_learning_rate_global)
    # assumes same learning rate for all causal groups
    meta_opt_groups = [optim.Adam(group_model.parameters(), lr=meta_learning_rate_group) for group_model in group_models]

    for epoch in range(num_epochs):
        qry_train_losses = []
        qry_val_losses = []
        group_qry_train_losses = [[] for  c in range(args['num_groups'])]

        # update models with training data
        for X_spt, Y_spt, X_qry, Y_qry, causal_group in train_loader:
            # sample the support and query data for the task
            X_spt_task = X_spt.to(args['device'])
            Y_spt_task = Y_spt.to(args['device']).unsqueeze(2)
            X_qry_task = X_qry.to(args['device'])
            Y_qry_task = Y_qry.to(args['device']).unsqueeze(2)

            group_model = group_models[causal_group.item()]
            
            # initialize the inner optimizer to adapt the parameters to the support set.
            inner_opt = torch.optim.SGD(group_model.parameters(), lr=base_learning_rate) 

            meta_opt_global.zero_grad()
            for optimiser in meta_opt_groups: optimiser.zero_grad()

            with higher.innerloop_ctx(
                group_model, inner_opt, copy_initial_weights=False
            ) as (fnet, diffopt):
                # update prior of local model using variational distribution of group model
                fnet = update_model_priors(fnet, fnet)

                # adapt the base model's parameters to the task
                for _ in range(args['num_base_steps']):
                    spt_loss = fnet.sample_elbo(inputs=X_spt_task, 
                                                labels=Y_spt_task,
                                                criterion=criterion, # criterion for likelihood term of loss
                                                sample_nbr=3, # number of MC samples
                                                complexity_cost_weight=args['lambda']) # weight on KL part of ELBO (complexity cost)
                    diffopt.step(spt_loss)

                # causal group model
                num_tasks_group = num_tasks_per_group[causal_group.item()]
                qry_loss_group = fnet.sample_elbo(inputs=X_qry_task, 
                                            labels=Y_qry_task,
                                            criterion=criterion, 
                                            sample_nbr=3,
                                            complexity_cost_weight=args['lambda']/num_tasks_group)
                
                qry_loss_group += args['lambda']/num_tasks_group * group_model.nn_kl_divergence()
                group_qry_train_losses[causal_group.item()].append(qry_loss_group.detach())
                qry_loss_group.backward(retain_graph=True)

                # global model
                qry_loss_global = fnet.sample_elbo(inputs=X_qry_task, 
                                            labels=Y_qry_task,
                                            criterion=criterion, 
                                            sample_nbr=3, 
                                            complexity_cost_weight=args['lambda']/num_tasks) 
                
                qry_loss_global += args['lambda'] * ((1/num_tasks * group_model.nn_kl_divergence()) + (1/num_tasks_group * global_model.nn_kl_divergence()))
                qry_train_losses.append(qry_loss_global.detach())
                qry_loss_global.backward()
            
            meta_opt_global.step()
            optimiser = meta_opt_groups[causal_group.item()]
            optimiser.step()
            
            # update the prior of the group models with the variational distribution from the global model
            for model_idx in range(len(group_models)):
                group_model = group_models[model_idx]
                group_models[model_idx] = update_model_priors(group_model, global_model)

        # validate models with validation data
        # check criterion for early stopping (due to overfitting)
        for X_spt, Y_spt, X_qry, Y_qry, causal_group in val_loader:
            # sample the support and query data for the task
            X_spt_task = X_spt.to(args['device'])
            Y_spt_task = Y_spt.to(args['device']).unsqueeze(2)
            X_qry_task = X_qry.to(args['device'])
            Y_qry_task = Y_qry.to(args['device']).unsqueeze(2)

            group_model = group_models[causal_group.item()]
            
            # initialize the inner optimizer to adapt the parameters to the support set.
            inner_opt = torch.optim.SGD(group_model.parameters(), lr=base_learning_rate) 

            meta_opt_global.zero_grad()
            for optimiser in meta_opt_groups: optimiser.zero_grad()

            with higher.innerloop_ctx(
                group_model, inner_opt, copy_initial_weights=False
            ) as (fnet, diffopt):
                # update prior of local model using variational distribution of group model
                fnet = update_model_priors(fnet, fnet)

                # adapt the base model's parameters to the task
                for _ in range(args['num_base_steps']):
                    spt_loss = fnet.sample_elbo(inputs=X_spt_task, 
                                                labels=Y_spt_task,
                                                criterion=criterion, # criterion for likelihood term of loss
                                                sample_nbr=3, # number of MC samples
                                                complexity_cost_weight=args['lambda']) # weight on KL part of ELBO (complexity cost)
                    diffopt.step(spt_loss)

                # global model
                qry_loss_global = fnet.sample_elbo(inputs=X_qry_task, 
                                            labels=Y_qry_task,
                                            criterion=criterion, 
                                            sample_nbr=3, 
                                            complexity_cost_weight=args['lambda']/num_tasks) 
                
                qry_loss_global += args['lambda'] * ((1/num_tasks * group_model.nn_kl_divergence()) + (1/num_tasks_group * global_model.nn_kl_divergence()))
                qry_val_losses.append(qry_loss_global.detach())


        epoch_train_loss = sum(qry_train_losses) / len(qry_train_losses)
        epoch_val_loss = sum(qry_val_losses) / len(qry_val_losses)
        epoch_group_train_losses = [sum(group_qry_loss) / len(group_qry_loss) for group_qry_loss in group_qry_train_losses]
        
        print(
            f'[Epoch {epoch}] Train Loss: {epoch_train_loss}\n',
            f'Val Loss: {epoch_val_loss}\n',
            f'Group Train Losses: {epoch_group_train_losses}'
        )

        early_stopping(epoch_val_loss)
        if early_stopping.early_stop:
            print("Early stopping")
            break

    return group_models, epoch


def update_model_priors(model_to_update, model_with_priors):
    """Updates the priors of model using the variational distribution learned by model_with_priors
    """
    weight_dim = model_with_priors[0].weight_mu.flatten().shape[0]
    bias_dim = model_with_priors[0].bias_mu.shape[0]
    # N(mu, sigma*I) where covariance matrix is a diagonal matrix and sigma is obtained by converting from rho
    weight_cov_matrix = torch.eye(weight_dim)*torch.log1p(torch.exp(model_with_priors[0].weight_rho.flatten())) # convert rho to sigma and make diagonal matrix by * Identity
    model_to_update[0].weight_prior_dist = PriorWeightDistributionMVN(mean=model_with_priors[0].weight_mu.flatten(), covariance_matrix=weight_cov_matrix, name='mean')
    bias_cov_matrix = torch.eye(bias_dim)*torch.log1p(torch.exp(model_with_priors[0].bias_rho))
    model_to_update[0].bias_prior_dist = PriorWeightDistributionMVN(mean=model_with_priors[0].bias_mu, covariance_matrix=bias_cov_matrix, name='bias')
    return model_to_update


def train_2level_model(train_loader, val_loader, num_epochs, meta_learning_rate, base_learning_rate, args):
    print(f"Training model for {num_epochs} epochs with base learning rate {base_learning_rate} and meta learning rate {meta_learning_rate}")
    num_tasks = len(train_loader)
    num_features = train_loader.dataset.num_features
    
    model = nn.Sequential(
        BayesianLinear(num_features, args['hidden_layer_size'], prior_sigma_1=args['prior_sigma_1'], prior_sigma_2=args['prior_sigma_2'], prior_pi=args['prior_pi']),
        nn.ReLU(),
        BayesianLinear(args['hidden_layer_size'], args['hidden_layer_size'], prior_sigma_1=args['prior_sigma_1'], prior_sigma_2=args['prior_sigma_2'], prior_pi=args['prior_pi']),
        nn.ReLU(),
        BayesianLinear(args['hidden_layer_size'], 1, prior_sigma_1=args['prior_sigma_1'], prior_sigma_2=args['prior_sigma_2'], prior_pi=args['prior_pi'])
    ).to(args['device'])

    model.train()
    meta_opt = optim.Adam(model.parameters(), lr=meta_learning_rate)
    criterion = torch.nn.MSELoss()

    for epoch in range(num_epochs):
        qry_losses = []

        for X_spt, Y_spt, X_qry, Y_qry in train_loader:
            # sample the support and query data for the task
            X_spt_task = X_spt.to(args['device'])
            Y_spt_task = Y_spt.to(args['device']).unsqueeze(2)
            X_qry_task = X_qry.to(args['device'])
            Y_qry_task = Y_qry.to(args['device']).unsqueeze(2)
            
            # initialize the inner optimizer to adapt the parameters to the support set.
            inner_opt = torch.optim.SGD(model.parameters(), lr=base_learning_rate) 

            meta_opt.zero_grad()

            with higher.innerloop_ctx(
                model, inner_opt, copy_initial_weights=False
            ) as (fnet, diffopt):
                # update prior of local model using variational distribution of global model
                fnet = update_model_priors(fnet, fnet)

                # Optimize the likelihood of the support set by taking
                # gradient steps w.r.t. the model's parameters.
                # This adapts the model's meta-parameters to the task.
                # higher is able to automatically keep copies of
                # your network's parameters as they are being updated.
                for _ in range(args['num_base_steps']):
                    spt_loss = fnet.sample_elbo(inputs=X_spt_task, 
                                                labels=Y_spt_task,
                                                criterion=criterion, # criterion for likelihood term of loss
                                                sample_nbr=3, # number of MC samples
                                                complexity_cost_weight=args['lambda']) # weight on KL part of ELBO (complexity cost)
                    diffopt.step(spt_loss)
                
                # The final set of adapted parameters will induce some
                # final loss and accuracy on the query dataset.
                # These will be used to update the model's meta-parameters.
                qry_loss = fnet.sample_elbo(inputs=X_qry_task, 
                                            labels=Y_qry_task,
                                            criterion=criterion, 
                                            sample_nbr=3, 
                                            complexity_cost_weight=args['lambda']/num_tasks) # weight on KL part of ELBO (complexity cost)
                # NOTE this is the final part of the ELBO incorporating the meta-prior: KL[ q(theta) || p(theta) ]
                complexity_cost_weight = args['lambda']/num_tasks
                qry_loss += model.nn_kl_divergence() * complexity_cost_weight
                qry_losses.append(qry_loss.detach())

                # Update the model's meta-parameters to optimize the query
                # losses across all of the tasks sampled in this batch.
                # This unrolls through the gradient steps.
                qry_loss.backward()
            
            meta_opt.step()

        epoch_loss = sum(qry_losses) / len(qry_losses)

        print(
            f'[Epoch {epoch}] Train Loss: {epoch_loss}'
        )

    return model


def predict_model(model, X_spt, Y_spt, X_qry, Y_qry, base_learning_rate, args, nsamples=50, return_parameters=False):
    """
    Returns the predictions for the given task data
    If return_parameters=True, the inferred task-specific variational parameters will be returned instead
    """
    X_spt_task = torch.Tensor(X_spt).to(args['device'])
    Y_spt_task = torch.Tensor(Y_spt).to(args['device']).unsqueeze(2)
    X_qry_task = torch.Tensor(X_qry).to(args['device'])
    Y_qry_task = torch.Tensor(Y_qry).to(args['device']).unsqueeze(2)
    
    # initialize the inner optimizer to adapt the parameters to the support set.
    inner_opt = torch.optim.SGD(model.parameters(), lr=base_learning_rate)
    criterion = torch.nn.MSELoss()
    
    with higher.innerloop_ctx(
        model, inner_opt, copy_initial_weights=False
    ) as (fnet, diffopt):
        # update prior of local model using variational distribution of global model
        fnet = update_model_priors(fnet, fnet)

        # Optimize the likelihood of the support set by taking
        # gradient steps w.r.t. the model's parameters.
        # This adapts the model's meta-parameters to the task.
        for _ in range(args['num_base_steps']):
            spt_loss = fnet.sample_elbo(inputs=X_spt_task, 
                                        labels=Y_spt_task,
                                        criterion=criterion, 
                                        sample_nbr=3, 
                                        complexity_cost_weight=args['lambda'])
            diffopt.step(spt_loss)
        
        if return_parameters:
            fnet.eval()
            return fnet, fnet[0].weight_mu.flatten().detach(), fnet[0].weight_rho.flatten().detach(), fnet[0].bias_mu.detach(), fnet[0].bias_rho.detach()
        else:
            # The query loss and acc induced by these parameters.
            y_pred = np.squeeze(np.array([fnet(X_qry_task).detach().numpy() for _ in range(nsamples)]))
            return y_pred


class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience.

    Ref. https://github.com/Bjarten/early-stopping-pytorch/blob/master/pytorchtools.py
    """
    def __init__(self, patience=7, verbose=False, delta=0, path='checkpoint.pt', trace_func=print):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement. 
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
            path (str): Path for the checkpoint to be saved to.
                            Default: 'checkpoint.pt'
            trace_func (function): trace print function.
                            Default: print            
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.path = path
        self.trace_func = trace_func
    def __call__(self, val_loss):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
        elif score < self.best_score + self.delta:
            self.counter += 1
            self.trace_func(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.counter = 0
