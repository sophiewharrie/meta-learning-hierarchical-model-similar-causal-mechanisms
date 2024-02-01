import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.nn import Sequential
import torch.nn.utils.prune as prune
from collections import Counter
import copy
import pandas as pd
from numpy.random import multivariate_normal

from blitz.modules import BayesianLinear
from blitz.utils import variational_estimator
from priors import update_prediction_model_priors
from utils import EarlyStopping

import higher

# adds decorator to nn.Module with utilities for variational inference
Sequential = variational_estimator(Sequential)

def inner_loop_init_prediction_model(data_loader, model, outer_opt, criterion, args, train=True):
    num_tasks = len(data_loader)
    qry_losses = []
    pred_model_for_task = []

    for X_spt, Y_spt, X_qry, Y_qry, _, _ in data_loader:
        # sample the support and query data for the task
        X_spt_task = X_spt.to(args['device'])
        Y_spt_task = Y_spt.to(args['device']).unsqueeze(2)
        X_qry_task = X_qry.to(args['device'])
        Y_qry_task = Y_qry.to(args['device']).unsqueeze(2)
        
        inner_opt = torch.optim.SGD(model.parameters(), lr=args['base_learning_rate']) 
        outer_opt.zero_grad()

        with higher.innerloop_ctx(
            model, inner_opt, copy_initial_weights=False
        ) as (fnet, diffopt):
            # transfer prior from global model
            fnet = update_prediction_model_priors(fnet, model, args)

            # update the local parameters
            for _ in range(args['num_base_steps']):
                spt_loss = fnet.sample_elbo(inputs=X_spt_task, 
                                            labels=Y_spt_task,
                                            criterion=criterion, 
                                            sample_nbr=args['num_mc_samples'],
                                            complexity_cost_weight=args['lambda'])
                diffopt.step(spt_loss)

            if args['causal_distance']=='diagnostic':
                pred_model_for_task.append({'weights':fnet[0].weight_mu.flatten().detach(), 'biases':fnet[0].bias_mu.detach()})

            # update the global parameters
            qry_loss = fnet.sample_elbo(inputs=X_qry_task, 
                                        labels=Y_qry_task,
                                        criterion=criterion, 
                                        sample_nbr=args['num_mc_samples'],
                                        complexity_cost_weight=args['lambda']/num_tasks)
            
            complexity_cost_weight = args['lambda']/num_tasks
            qry_loss += model.nn_kl_divergence() * complexity_cost_weight
            qry_losses.append(qry_loss.detach())

            if train:
                qry_loss.backward()
        
        if train:
            outer_opt.step()

    epoch_loss = sum(qry_losses) / len(qry_losses)
    return model, epoch_loss, pred_model_for_task


def init_prediction_model(train_loader, val_loader, test_loader, args):
    """Pre-training step to learn priors for global parameters
    """
    num_features = train_loader.dataset.num_features
    
    pred_model = nn.Sequential(
        BayesianLinear(num_features, args['hidden_layer_size'], prior_sigma_1=args['prior_sigma_1'], prior_sigma_2=args['prior_sigma_2'], prior_pi=args['prior_pi']),
        nn.ReLU(),
        BayesianLinear(args['hidden_layer_size'], args['hidden_layer_size'], prior_sigma_1=args['prior_sigma_1'], prior_sigma_2=args['prior_sigma_2'], prior_pi=args['prior_pi']),
        nn.ReLU(),
        BayesianLinear(args['hidden_layer_size'], 1, prior_sigma_1=args['prior_sigma_1'], prior_sigma_2=args['prior_sigma_2'], prior_pi=args['prior_pi'])
    ).to(args['device'])

    pred_model.train()
    meta_opt = optim.Adam(pred_model.parameters(), lr=args['meta_learning_rate_global'])
    criterion = torch.nn.MSELoss()

    early_stopping = EarlyStopping(patience=args['patience'])
    
    pred_model_for_task_train, pred_model_for_task_val, pred_model_for_task_test = None, None, None

    for epoch in range(args['num_initial_epochs']):
        pred_model, epoch_train_loss, pred_model_for_task_train = inner_loop_init_prediction_model(train_loader, pred_model, meta_opt, criterion, args, train=True)
        _, epoch_val_loss, pred_model_for_task_val = inner_loop_init_prediction_model(val_loader, pred_model, meta_opt, criterion, args, train=False)
        _, _, pred_model_for_task_test = inner_loop_init_prediction_model(test_loader, pred_model, meta_opt, criterion, args, train=False)

        print(
            f'[Epoch {epoch}] Train Loss: {epoch_train_loss} Val Loss: {epoch_val_loss}'
        )

        early_stopping(epoch_val_loss)
        if early_stopping.early_stop:
            print("Early stopping")
            break

    return pred_model, pred_model_for_task_train, pred_model_for_task_val, pred_model_for_task_test


def run_prediction_model(X_spt, Y_spt, X_qry, Y_qry, causal_group, group_pred_models, args, train=True, eval=False, global_pred_model=None, meta_opt_global_pred=None, meta_opt_groups_pred=None, num_tasks=None, num_tasks_per_group=None):
    """
    During training, use train=True, eval=False for training data, and train=False, eval=False for val data. During evaluation (for val and test data), use train=False, eval=True
    """
    X_spt_task = X_spt.to(args['device'])
    Y_spt_task = Y_spt.to(args['device']).unsqueeze(2)
    X_qry_task = X_qry.to(args['device'])
    Y_qry_task = Y_qry.to(args['device']).unsqueeze(2)

    losses_to_return = {'qry_loss_group_pred':None,
                        'qry_loss_global_pred':None}

    group_pred_model = group_pred_models[causal_group]
    inner_opt_pred = torch.optim.SGD(group_pred_model.parameters(), lr=args['base_learning_rate']) 

    criterion = torch.nn.MSELoss()

    if not eval:
        meta_opt_global_pred.zero_grad()
        for optimiser_pred in meta_opt_groups_pred: optimiser_pred.zero_grad()

    with higher.innerloop_ctx(
        group_pred_model, inner_opt_pred, copy_initial_weights=False
    ) as (fnet_pred, diffopt_pred):
        # update prior of local model using variational distribution of group model
        fnet_pred = update_prediction_model_priors(fnet_pred, fnet_pred, args)

        # adapt the base model's parameters to the task
        for _ in range(args['num_base_steps']):
            spt_loss_pred = fnet_pred.sample_elbo(inputs=X_spt_task, 
                                        labels=Y_spt_task,
                                        criterion=criterion,
                                        sample_nbr=args['num_mc_samples'],
                                        complexity_cost_weight=args['lambda'])
            diffopt_pred.step(spt_loss_pred)

        if not eval:
            # causal group model
            num_tasks_group = num_tasks_per_group[causal_group]
            if train:
                qry_loss_group_pred = fnet_pred.sample_elbo(inputs=X_qry_task, 
                                            labels=Y_qry_task,
                                            criterion=criterion, 
                                            sample_nbr=args['num_mc_samples'],
                                            complexity_cost_weight=args['lambda']/num_tasks_group)
                
                qry_loss_group_pred += args['lambda']/num_tasks_group * group_pred_model.nn_kl_divergence()
                losses_to_return['qry_loss_group_pred'] = qry_loss_group_pred.detach()
                qry_loss_group_pred.backward(retain_graph=True)

            # global model
            qry_loss_global_pred = fnet_pred.sample_elbo(inputs=X_qry_task, 
                                        labels=Y_qry_task,
                                        criterion=criterion, 
                                        sample_nbr=args['num_mc_samples'], 
                                        complexity_cost_weight=args['lambda']/num_tasks) 
            
            qry_loss_global_pred += args['lambda'] * ((1/num_tasks * group_pred_model.nn_kl_divergence()) + (1/num_tasks_group * global_pred_model.nn_kl_divergence()))
            losses_to_return['qry_loss_global_pred'] = qry_loss_global_pred.detach()

            if train:
                qry_loss_global_pred.backward()

        else:
            y_pred = np.squeeze(np.array([fnet_pred(X_qry_task).detach().cpu().numpy() for _ in range(args['num_mc_samples'])]))
        
    if not eval and train:
        meta_opt_global_pred.step()
        optimiser_pred = meta_opt_groups_pred[causal_group]
        optimiser_pred.step()
        
        # update the prior of the group models with the variational distribution from the global model
        for model_idx in range(len(group_pred_models)):
            group_pred_model = group_pred_models[model_idx]
            group_pred_models[model_idx] = update_prediction_model_priors(group_pred_model, global_pred_model, args)

    # for eval, return the predictions from the model
    if eval:
        return y_pred
    # otherwise, return the models and losses for further training
    else:
        return losses_to_return, group_pred_models, global_pred_model, meta_opt_global_pred, meta_opt_groups_pred


def inner_loop_main_prediction_model(data_loader, causal_groups, global_pred_model, group_pred_models, meta_opt_global_pred, meta_opt_groups_pred, args, train=True):
    num_tasks = len(data_loader)
    num_tasks_per_group = Counter(causal_groups)
    qry_losses_pred = []
    group_qry_losses_pred = [[] for  c in range(args['num_groups'])]

    # update models with training data
    task_number = 0
    for X_spt, Y_spt, X_qry, Y_qry, spt_interv_mask, qry_interv_mask in data_loader:
        causal_group = causal_groups[task_number] 
        task_losses, group_pred_models, global_pred_model, meta_opt_global_pred, meta_opt_groups_pred = run_prediction_model(X_spt, Y_spt, X_qry, Y_qry, causal_group, group_pred_models, args, train, eval=False, global_pred_model=global_pred_model, meta_opt_global_pred=meta_opt_global_pred, meta_opt_groups_pred=meta_opt_groups_pred, num_tasks=num_tasks, num_tasks_per_group=num_tasks_per_group)
        group_qry_losses_pred[causal_group].append(task_losses['qry_loss_group_pred'])
        qry_losses_pred.append(task_losses['qry_loss_global_pred'])
        task_number += 1
    
    epoch_loss_pred = sum(qry_losses_pred) / len(qry_losses_pred)

    if train:
        epoch_group_losses_pred = [sum(group_qry_loss) / len(group_qry_loss) for group_qry_loss in group_qry_losses_pred]
    else:
        epoch_group_losses_pred = None

    return global_pred_model, group_pred_models, epoch_loss_pred, epoch_group_losses_pred


def main_prediction_model(init_pred_model, train_loader, val_loader, causal_groups_train, causal_groups_val, representative_models, args):
    early_stopping = EarlyStopping(patience=args['patience'])
    
    global_pred_model = copy.deepcopy(init_pred_model) 
    group_pred_models = [copy.deepcopy(init_pred_model) for c in range(args['num_groups'])] 

    # update the prior of the group models with the variational distribution from the global model
    for model_idx in range(len(group_pred_models)):
        group_pred_model = group_pred_models[model_idx]
        group_pred_models[model_idx] = update_prediction_model_priors(group_pred_model, global_pred_model, args)

    global_pred_model.train()
    for group_pred_model in group_pred_models: group_pred_model.train()
    meta_opt_global_pred = optim.Adam(global_pred_model.parameters(), lr=args['meta_learning_rate_global'])
    meta_opt_groups_pred = [optim.Adam(group_model.parameters(), lr=args['meta_learning_rate_group']) for group_model in group_pred_models]
        
    for epoch in range(args['num_main_epochs']):
        global_pred_model, group_pred_models, epoch_train_loss_pred, epoch_group_train_losses_pred = inner_loop_main_prediction_model(train_loader, causal_groups_train, global_pred_model, group_pred_models, meta_opt_global_pred, meta_opt_groups_pred, args, train=True)
        _, _, epoch_val_loss_pred, epoch_group_val_losses_pred = inner_loop_main_prediction_model(val_loader, causal_groups_val, global_pred_model, group_pred_models, meta_opt_global_pred, meta_opt_groups_pred, args, train=False)
        
        print(
            f'[Epoch {epoch}] Train Loss Predictive Model: {epoch_train_loss_pred}\n',
            f'Val Loss Predictive Model: {epoch_val_loss_pred}\n'
        )

        early_stopping(epoch_val_loss_pred)
        if early_stopping.early_stop:
            print("Early stopping")
            break

    return group_pred_models, epoch, global_pred_model
