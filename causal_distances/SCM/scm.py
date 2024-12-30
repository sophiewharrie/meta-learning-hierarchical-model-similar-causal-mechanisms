import pandas as pd
import numpy as np
import sys
sys.path.append('../dibs') # NOTE uses modified version of the package, https://github.com/sophiewharrie/dibs
sys.path.append('causal_distances')
from dibs.inference import JointDiBS
from dibs.models import ErdosReniDAGDistribution, ScaleFreeDAGDistribution, LinearGaussian, DenseNonlinearGaussian
import jax.random as random
import copy
from long_utils import Data
import argparse
from scipy.spatial.distance import cosine

"""
Estimating "task similarity distance" using continuous structure learning:
- DiBS (Differentiable Bayesian Structure Learning) by Lorch et al. (2021) allows for inferring the structure of Bayesian networks (G, Theta) from data. It operates in a continuous space of a latent probabilistic graph representation Z, and can handle Bayesian network models with nonlinear dependencies encoded by neural networks
- Stein Variational Gradient Descent (SVGD) is used to iteratively transport a set of particles until they approximate the posterior distribution p(G, Theta | D). In particular, Z corresponds to the latent graph structure, and each Z particle represents a different candidate graph structure
- Our algorithm builds on top of DiBS for estimating the distance between inferred graph structures:
    1. Initialize a "global" DiBS model for the data from all tasks, i.e., as a pre-training approach to initialize a set of particles for (Z, Theta)
    2. Fine-tune the global model to each of the tasks. This results is a "local" set of particles (Z_i, Theta_i) for each task i
        2.1. The particles from the global model represent different candidate graph structures. Compute the log likelihood p(D | G, Theta) for each global particle and select the particle/s that best explain the local data
        2.2. Fine-tune the selected particles using the local data. If using > 1 local particle, take the average across all local particles to get a final estimate for Z_i 
    3. Compute distance between pairs of tasks (i, j) by measuring (cosine) distance between (Z_i, Z_j)
- This algorithm assumes that all tasks use the same variables in X (but can differ in the "target" Y), and that there are sufficient similarities in X --> Y relationships across tasks to benefit from the pre-training approach
- Important note: Without further assumptions, structure learning methods such as DiBS that learn from observational data do not guarantee that the inferred conditional dependencies between variables correspond to actual causal relationships. To improve on this, the interventional mask input allows for incorporating information about hard interventions in the data.
"""

def get_graph_and_likelihood_models(args):
    if args['causal_prior']=='erdos_renyi':
        graph_model = ErdosReniDAGDistribution(args['d'], n_edges_per_node=args['n_edges_per_node'])
    elif args['causal_prior']=='scale_free':
        graph_model = ScaleFreeDAGDistribution(args['d'], n_edges_per_node=args['n_edges_per_node'])
    
    if args['causal_model']=='linear':
        # linear gaussian model
        likelihood_model = LinearGaussian(n_vars=args['d'], obs_noise=args['obs_noise'])
    elif args['causal_model']=='non_linear':
        # non-linear gaussian model
        # 2-layer neural network with fully connected layers, number of hidden nodes given by hidden_layer_size argument
        likelihood_model = DenseNonlinearGaussian(n_vars=args['d'], hidden_layers=[args['hidden_layer_size'], args['hidden_layer_size']], obs_noise=args['obs_noise'])
    
    return graph_model, likelihood_model


def local_causal_model(local_data, local_interv_mask, global_model, subk, g_final_global, theta_final_global, z_final_global, sf_baseline, graph_model, likelihood_model, args):
    causal_model_local = copy.deepcopy(global_model)

    if not args['use_interv_mask']:
        local_interv_mask = np.zeros_like(local_data)
    
    causal_model_local.x = local_data
    causal_model_local.interv_mask = local_interv_mask
    causal_model_local.log_graph_prior  = graph_model.unnormalized_log_prob_soft
    causal_model_local.log_joint_prob = likelihood_model.interventional_log_joint_prob
    causal_model_local.likelihood_model = likelihood_model
    causal_model_local.graph_model = graph_model

    particle_log_likelihoods = []
    for particle in range(args['num_particles_global']):
        # the particles from the global model represent different candidate graph structures
        # compute log likelihood p(D | G, Theta) for each particle
        # based on this, get particle/s that best explains the local data
        log_likelihood = likelihood_model.log_likelihood(x=local_data, theta=theta_final_global[particle], g=g_final_global[particle], interv_targets=local_interv_mask)
        print(log_likelihood)
        particle_log_likelihoods.append(log_likelihood)

    use_particles = np.argsort(-np.array(particle_log_likelihoods))[:args['num_particles_local']]
    print("Selected particle/s for task:", use_particles)

    # run optimisation with non-random initialisation, i.e., based on most likely particles from global model
    g_final_local, theta_final_local, z_final_local, _ = causal_model_local.sample(key=subk, n_particles=args['num_particles_local'], steps=args['num_steps_local'], init_model=False, init_z=z_final_global[use_particles], init_theta=theta_final_global[use_particles], sf_baseline=sf_baseline[use_particles])

    # z has shape (num particles, d, k latent dim, 2 for u,v), where z = u.T * v
    z_particles = [np.einsum('...ik,...jk->...ij', particle[:,:,0], particle[:,:,1]) for particle in z_final_local]
    z_particles_mean = np.mean(z_particles, axis=0) # take average over local particles

    return z_particles_mean


def global_causal_model(data, interv_mask, subk, graph_model, likelihood_model, args):

    if args['use_interv_mask']:
        causal_model = JointDiBS(x=data, interv_mask=interv_mask, graph_model=graph_model, likelihood_model=likelihood_model, alpha_linear=args['alpha'], kernel_param={'h_latent':args['bandwidth_z'], 'h_theta': args['bandwidth_theta']})
    else:
        causal_model = JointDiBS(x=data, graph_model=graph_model, likelihood_model=likelihood_model, alpha_linear=args['alpha'], kernel_param={'h_latent':args['bandwidth_z'], 'h_theta': args['bandwidth_theta']})
    
    g_final_global, theta_final_global, z_final_global, sf_baseline = causal_model.sample(key=subk, n_particles=args['num_particles_global'], steps=args['num_steps_global'], n_dim_particles=args['latent_dim_k'], init_model=True) 

    return causal_model, g_final_global, theta_final_global, z_final_global, sf_baseline


def get_global_data(data):
    if args['use_loader']:
        # for global data use all target tasks as outcome endpoint
        outcome_endpoint = data.target_tasks
        tasks = data.all_tasks
        global_data = data.get_dataset(outcome_endpoint, with_covariates=True)
        args['d'] = len(global_data.keys())
        args['latent_dim_k'] = args['d']
        global_data = global_data.values
        global_interv_mask = None
    else:
        meta_df = pd.read_csv(args['metadata_filepath'])
        predictors = meta_df[meta_df['column_type']=='predictor']['column_name'].tolist()
        tasks = meta_df[meta_df['column_type']=='task_label']['column_name'].tolist()
        data = pd.read_csv(args['maindata_filepath'])
        global_data = data[predictors]
        global_data['Y'] = data[tasks].sum(axis=1)
        global_data = global_data.values
        global_interv_mask = None
        args['d'] = len(predictors)+1
        args['latent_dim_k'] = args['d']
    return global_data, global_interv_mask, tasks


def get_local_data(data, task):
    if args['use_loader']:
        local_data = data.get_dataset(task, with_covariates=True)
        local_data = local_data.values
        local_interv_mask = None
    else:
        meta_df = pd.read_csv(args['metadata_filepath'])
        predictors = meta_df[meta_df['column_type']=='predictor']['column_name'].tolist()
        cohort_name = meta_df[(meta_df['column_type']=='cohort')&(meta_df['task_cohort']==task)]['column_name'].tolist()[0]
        data = pd.read_csv(args['maindata_filepath'])
        local_data = data[data[cohort_name]==1][predictors]
        local_data['Y'] = data[data[cohort_name]==1][task]
        local_data = local_data.values
        local_interv_mask = None
    return local_data, local_interv_mask


def get_causal_models(args, data):

    key = random.PRNGKey(0)
    key, subk = random.split(key)

    # get the global data
    global_data, global_interv_mask, tasks = get_global_data(data)
    
    # get the graph and likelihood models
    graph_model, likelihood_model = get_graph_and_likelihood_models(args)

    # train the global model (initialisation)
    print("Training global model (may take a long time if there's a large number of variables and particles)")
    global_model, g_final_global, theta_final_global, z_final_global, sf_baseline = global_causal_model(global_data, global_interv_mask, subk, graph_model, likelihood_model, args)
    print(z_final_global)

    # derive the task-specific models
    local_z_list = []
    for task in tasks:
        
        # get the local data
        local_data, local_interv_mask = get_local_data(data, task)

        # fine-tune the local model
        print("Fine-tuning local model for task:", task)
        local_z_particles = local_causal_model(local_data, local_interv_mask, global_model, subk, g_final_global, theta_final_global, z_final_global, sf_baseline, graph_model, likelihood_model, args)
        print(local_z_particles)
        local_z_list.append(local_z_particles)

    # return latent variable z for tasks
    return local_z_list, tasks


# def euclidean_distance(a, b):
#     return np.linalg.norm(a - b)


def get_distance_matrix(z_list, task_list):
    # get distances between tasks
    dist_data = []
    for t1 in range(len(task_list)):
        for t2 in range(len(task_list)):
            z1 = z_list[t1]
            z2 = z_list[t2]
            dist = cosine(z1.flatten(), z2.flatten())
            dist_data.append({'task1':task_list[t1], 'task2':task_list[t2], 'value':dist})

    dist_df = pd.DataFrame(dist_data)
    return dist_df


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--outprefix', type=str, help='prefix for the path to save the output files')
    args = vars(parser.parse_args())

    # additional arguments that can be adjusted to your needs:

    # if True uses Data class from long_utils.py or if False uses data from own files specified here
    args['use_loader'] = True
    # only needed if use_loader is False
    args['maindata_filepath'] = 'path/to/mainfile.csv'
    # only needed is use_loader is False
    args['metadata_filepath'] = 'path/to/metafile.csv'
    # prior for expected number of edges per node in DAG
    args['n_edges_per_node'] = 3
    # type of graph prior (erdos_renyi or scale_free)
    args['causal_prior'] = 'erdos_renyi'
    # type of causal model (linear or non_linear)
    args['causal_model'] = 'linear'
    # variance of error in structural equations
    args['obs_noise'] = 0.1
    # set to True if also providing intervention mask information
    args['use_interv_mask'] = False
    # number of SVGD particles to sample for global model (how many causal model components you expect there to be in the data)
    args['num_particles_global'] = 1
    # number of SVGD particles to sample for local models
    args['num_particles_local'] = 3
    # number of SVGD steps for global model
    args['num_steps_global'] = 200
    # number of SVGD steps for each local model
    args['num_steps_local'] = 20
    # SVGD kernel parameter
    args['alpha'] = 0.2
    # SVGD kernel parameter
    args['bandwidth_z'] = 5
    # SVGD kernel parameter
    args['bandwidth_theta'] = 500

    # initialize data class
    if args['use_loader']:
        print("loading data")
        data = Data()
    else:
        data = None

    # apply DiBS (globally) and fine-tune DiBS (locally) to get latent factors z for each task
    print("getting causal models")
    z_list, task_list = get_causal_models(args, data)

    # compute causal distance between all pairs of tasks
    dist_df = get_distance_matrix(z_list, task_list)
    outpath = '{}_causal_distance_SCM_method.csv'.format(args['outprefix'])
    print(dist_df)
    dist_df.to_csv(outpath, index=None)