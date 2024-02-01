import torch
import numpy as np
from tqdm import tqdm

import sys
sys.path.append('../dibs') # NOTE uses modified version of the package
sys.path.insert(0, 'utils')
from dibs.inference import JointDiBS
from dibs.models import ErdosReniDAGDistribution, ScaleFreeDAGDistribution, LinearGaussian, DenseNonlinearGaussian
from loader import get_task_data
import copy
import os

# use following code to address issues with JAX/PyTorch memory conflicts, see https://github.com/google/jax/issues/1222
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false" 
os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"

import jax.random as random

def get_graph_and_likelihood_models(args):
    if args['causal_prior']=='erdos_renyi':
        graph_model = ErdosReniDAGDistribution(args['d'], n_edges_per_node=args['n_edges_per_node'])
    elif args['causal_prior']=='scale_free':
        graph_model = ScaleFreeDAGDistribution(args['d'], n_edges_per_node=args['n_edges_per_node'])
    
    if args['causal_model']=='linear':
        likelihood_model = LinearGaussian(n_vars=args['d'], obs_noise=args['obs_noise'])
    elif args['causal_model']=='non_linear':
        # 2-layer neural network with five hidden nodes
        likelihood_model = DenseNonlinearGaussian(n_vars=args['d'], hidden_layers=[5, 5], obs_noise=args['obs_noise'])
    
    return graph_model, likelihood_model


def init_causal_model(train_loader, val_loader, test_loader, args):
    key = random.PRNGKey(0)
    key, subk = random.split(key)

    # train the global model
    causal_model, opt_state_z, opt_state_theta, sf_baseline = global_causal_model(train_loader, subk, args)

    # derive the task-specific models
    init_task_SCM_train = get_causal_models_for_tasks(train_loader, causal_model, subk, opt_state_z, opt_state_theta, sf_baseline, args)
    init_task_SCM_val = get_causal_models_for_tasks(val_loader, causal_model, subk, opt_state_z, opt_state_theta, sf_baseline, args)
    init_task_SCM_test = get_causal_models_for_tasks(test_loader, causal_model, subk, opt_state_z, opt_state_theta, sf_baseline, args)

    return init_task_SCM_train, init_task_SCM_val, init_task_SCM_test


def get_causal_models_for_tasks(data_loader, causal_model, subk, opt_state_z, opt_state_theta, sf_baseline, args):
    causal_models = []
    for X_spt, Y_spt, X_qry, Y_qry, spt_interv_mask, qry_interv_mask in tqdm(data_loader):
        spt_task = torch.cat([X_spt, Y_spt.unsqueeze(dim=2)],dim=2).to(args['device'])
        qry_task = torch.cat([X_qry, Y_qry.unsqueeze(dim=2)],dim=2).to(args['device'])
        spt_interv_mask = spt_interv_mask.to(args['device'])
        qry_interv_mask = qry_interv_mask.to(args['device'])

        causal_model_for_task = local_causal_model(spt_task, qry_task, spt_interv_mask, qry_interv_mask, causal_model, subk, opt_state_z, opt_state_theta, sf_baseline, args)
        causal_models.append(causal_model_for_task)

    return causal_models


def global_causal_model(data_loader, subk, args):
    data, interv_mask = get_task_data(data_loader, task_index=None, support_only=False)
    graph_model, likelihood_model = get_graph_and_likelihood_models(args)

    if args['use_interv_mask']:
        causal_model = JointDiBS(x=data, interv_mask=interv_mask, graph_model=graph_model, likelihood_model=likelihood_model, alpha_linear=args['alpha'], kernel_param={'h_latent':args['bandwidth_z'], 'h_theta': args['bandwidth_theta']})
    else:
        causal_model = JointDiBS(x=data, graph_model=graph_model, likelihood_model=likelihood_model, alpha_linear=args['alpha'], kernel_param={'h_latent':args['bandwidth_z'], 'h_theta': args['bandwidth_theta']})
    
    g_final_global, theta_final_global, z_final_global, opt_state_z, opt_state_theta, model_key, sf_baseline = causal_model.sample(key=subk, n_particles=args['num_mc_samples'], steps=args['num_pretrain_epochs'], n_dim_particles=args['k'], init_model=True) 

    return causal_model, opt_state_z, opt_state_theta, sf_baseline


def local_causal_model(spt_task, qry_task, spt_interv_mask, qry_interv_mask, global_model, subk, opt_state_z, opt_state_theta, sf_baseline, args):
    causal_model_local = copy.deepcopy(global_model)
    data = torch.concat([spt_task[0], qry_task[0]]).detach().cpu().numpy()
    interv_mask = torch.concat([spt_interv_mask[0], qry_interv_mask[0]]).detach().cpu().numpy()
    graph_model, likelihood_model = get_graph_and_likelihood_models(args)
    
    causal_model_local.x = data
    causal_model_local.interv_mask = interv_mask
    causal_model_local.log_graph_prior  = graph_model.unnormalized_log_prob_soft
    causal_model_local.log_joint_prob = likelihood_model.interventional_log_joint_prob
    causal_model_local.likelihood_model = likelihood_model
    causal_model_local.graph_model = graph_model

    # run optimisation with non-random initialisation, determined by global training output
    g_final_local, theta_final_local, z_final_local = causal_model_local.sample(key=subk, n_particles=args['num_mc_samples'], steps=5, init_model=False, opt_state_z=opt_state_z, opt_state_theta=opt_state_theta, sf_baseline=sf_baseline)

    G = (np.array(g_final_local.__array__().mean(axis=0))>=0.5).astype(int)
    Z = z_final_local.__array__().mean(axis=0)
    Z_scores = np.einsum('...ik,...jk->...ij', Z[:,:,0], Z[:,:,1])
    causal_model_for_task = {'G':G, 'Z':Z_scores}
    
    return causal_model_for_task
