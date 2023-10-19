import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from tqdm import tqdm
from torch.nn import Sequential

from blitz.modules import BayesianLinear
from blitz.utils import variational_estimator
from prior import PriorWeightDistributionMVN

import higher

# adds decorator to nn.Module with utilities for variational inference
Sequential = variational_estimator(Sequential)


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


def train_bayesian_maml(train_loader, val_loader, args):
    num_features = train_loader.dataset.num_features
    num_train_tasks = len(train_loader)

    # declare Bayesian neural network with two hidden layers
    model = nn.Sequential(
        BayesianLinear(num_features, args['hidden_layer_size'], prior_sigma_1=args['prior_sigma_1'], prior_sigma_2=args['prior_sigma_2'], prior_pi=args['prior_pi']),
        nn.ReLU(),
        BayesianLinear(args['hidden_layer_size'], args['hidden_layer_size'], prior_sigma_1=args['prior_sigma_1'], prior_sigma_2=args['prior_sigma_2'], prior_pi=args['prior_pi']),
        nn.ReLU(),
        BayesianLinear(args['hidden_layer_size'], 1, prior_sigma_1=args['prior_sigma_1'], prior_sigma_2=args['prior_sigma_2'], prior_pi=args['prior_pi'])
    ).to(args['device'])

    model.train()
    meta_opt = optim.Adam(model.parameters(), lr=args['meta_learning_rate'])
    criterion = torch.nn.MSELoss()

    for epoch in range(args['num_epochs']):
        qry_losses = []
        
        for X_spt, Y_spt, X_qry, Y_qry in tqdm(train_loader):
            # sample the support and query data for the task
            X_spt_task = X_spt.to(args['device'])
            Y_spt_task = Y_spt.to(args['device']).unsqueeze(2)
            X_qry_task = X_qry.to(args['device'])
            Y_qry_task = Y_qry.to(args['device']).unsqueeze(2)

            # initialize the inner optimizer to adapt the parameters to the support set.
            inner_opt = torch.optim.SGD(model.parameters(), lr=args['base_learning_rate']) 

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
                                            complexity_cost_weight=args['lambda']/num_train_tasks) # weight on KL part of ELBO (complexity cost)
                # NOTE this is the final part of the ELBO incorporating the meta-prior: KL[ q(theta) || p(theta) ]
                complexity_cost_weight = args['lambda']/num_train_tasks
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


def predict_bayesian_maml(model, X_spt, Y_spt, X_qry, Y_qry, args, nsamples=50):
    X_spt_task = X_spt.to(args['device'])
    Y_spt_task = Y_spt.to(args['device']).unsqueeze(2)
    X_qry_task = X_qry.to(args['device'])
    Y_qry_task = Y_qry.to(args['device']).unsqueeze(2)

    # initialize the inner optimizer to adapt the parameters to the support set.
    inner_opt = torch.optim.SGD(model.parameters(), lr=args['base_learning_rate'])
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
        
        # The query loss and acc induced by these parameters.
        y_pred = np.squeeze(np.array([fnet(X_qry_task).detach().numpy() for _ in range(nsamples)]))
        return y_pred