import torch
import numpy as np
from torch import nn, func
import torchopt
import posteriors
import sys
from functools import partial
from tqdm import tqdm
from optree import tree_map
from torch.func import grad_and_value
import gc
import matplotlib.pyplot as plt
from torch.utils.data import TensorDataset, DataLoader

sys.path.insert(0, 'utils')
sys.path.insert(0, 'metrics')
sys.path.insert(0, 'method/hierarchical_model')
from bnn_models import linear_nn_model, sequence_nn_model
from bnn_priors import multivariate_normal_prior
from causal_similarity_utils import CausalSimilarity
from accuracy import get_pred_table
from explainability import save_embeddings


def bnn_model(train_task_loader, pred_task_loader, data, hyper_params, args, pred_for='val'):
    """Trains the Bayesian neural network model using train_task_loader
    and returns probabilities for predictions on pred_task_loader

    Returns None if the trial needed to stop early due to failure
    """

    # eval setting for whether to include validation set or not
    if pred_for == 'val': 
        eval = False
    elif pred_for == 'test': 
        eval = True
    
    num_data = hyper_params['prior_scaling']
    num_tasks = len(data.task_names)
    
    if 'inner_temperature' in args: args['inner_temperature'] = args['inner_temperature']/num_data
    if 'outer_temperature' in args: args['outer_temperature'] = args['outer_temperature']/num_tasks
    if 'auxiliary_temperature' in args: args['auxiliary_temperature'] = args['auxiliary_temperature']/num_tasks
    
    if hyper_params['nn_model_name'] == 'linear_nn_model':
        num_features = len(data.predictors)
        hidden_layer_sizes = [hyper_params['hidden_layer_size_tabular']]*hyper_params['n_layers']
        model = linear_nn_model(in_features=num_features, n_layers=hyper_params['n_layers'], out_features_list=hidden_layer_sizes)
    elif hyper_params['nn_model_name'] == 'sequence_nn_model':
        num_features_tabular = len(data.predictors)
        num_features_longitudinal = data.total_num_endpoints
        model = sequence_nn_model(num_features_longitudinal, num_features_tabular, hyper_params['hidden_layer_size_longitudinal'], hyper_params['hidden_layer_size_tabular'])

    model.to(args['device'])

    params = dict(model.named_parameters())

    def log_posterior_support_update(prior_mean, prior_log_sd_diag, params, task):
        # inner update uses support set
        X_spt, X_qry, X_long_spt, X_long_qry, y_spt, y_qry, imp_spt, imp_qry = task
        logits = func.functional_call(model, params, (X_spt, X_long_spt))
        
        sd_diag = tree_map(torch.exp, prior_log_sd_diag) # need to take exponential of log(sd_diag) to get just sd_diag
        log_prior = multivariate_normal_prior(params, mean=prior_mean, sd_diag=sd_diag, normalize=False)
        
        # focal loss - turned off for now (adjust if using)
        alpha = 0.5
        gamma = 0
        # convert logits to probs
        probs = torch.nn.functional.softmax(logits, dim=-1)
        probs = torch.where(probs < 1e-5, 1e-5, probs) # replaces probs smaller than 1e-5 with 1e-5 for numerical stability
        probs /= probs.sum(dim=-1, keepdim=True) # renormalize the probabilities to ensure they sum to 1
        probs = probs[:,1] # only need probability for positive class
        loss = nn.functional.cross_entropy(logits, y_spt, reduction="none")
        p_t = probs * y_spt + (1 - probs) * (1- y_spt) # measures how hard an example is to classify; lower value means more difficult to classify so use (1-p_t)
        alpha_t = alpha * y_spt + (1 - alpha) * (1 - y_spt) # rebalancing term
        focal_score = (1 - p_t) * 1
        focal_term = focal_score ** gamma # focus on hard to classify examples; gamma=0 sets all to weight 1; gamma=1 uses weights as is; gamma > 1 puts more even more weight on higher weight examples
        log_likelihood = (alpha_t * focal_term * loss).mean()
        #log_likelihood = torch.nn.functional.cross_entropy(logits, y_spt)
        
        log_likelihood = -log_likelihood
        log_prior = log_prior / num_data
        log_posterior = (
                    log_likelihood
                    + log_prior
                )
        return log_posterior, (logits, log_likelihood, log_prior) 
    
    def log_posterior_query_update(prior_mean, prior_log_sd_diag, params, task):
        """
        Same as log_posterior_support_update(), but for the query data
        """
        # inner update uses support set
        X_spt, X_qry, X_long_spt, X_long_qry, y_spt, y_qry, imp_spt, imp_qry = task
        logits = func.functional_call(model, params, (X_qry, X_long_qry))
        
        sd_diag = tree_map(torch.exp, prior_log_sd_diag) # need to take exponential of log(sd_diag) to get just sd_diag
        log_prior = multivariate_normal_prior(params, mean=prior_mean, sd_diag=sd_diag, normalize=False)

        # focal loss - turned off for now (adjust if using)
        alpha = 0.5
        gamma = 0
        # convert logits to probs
        probs = torch.nn.functional.softmax(logits, dim=-1)
        probs = torch.where(probs < 1e-5, 1e-5, probs) # replaces probs smaller than 1e-5 with 1e-5 for numerical stability
        probs /= probs.sum(dim=-1, keepdim=True) # renormalize the probabilities to ensure they sum to 1
        probs = probs[:,1] # only need probability for positive class
        loss = nn.functional.cross_entropy(logits, y_qry, reduction="none")
        p_t = probs * y_qry + (1 - probs) * (1- y_qry) # measures how hard an example is to classify; lower value means more difficult to classify so use (1-p_t)
        alpha_t = alpha * y_qry + (1 - alpha) * (1 - y_qry) # rebalancing term
        focal_score = (1 - p_t) * 1
        focal_term = focal_score ** gamma # focus on hard to classify examples; gamma=0 sets all to weight 1; gamma=1 uses weights as is; gamma > 1 puts more even more weight on higher weight examples
        log_likelihood = (alpha_t * focal_term * loss).mean()
        #log_likelihood = torch.nn.functional.cross_entropy(logits, y_qry)
        
        log_likelihood = -log_likelihood
        log_prior = log_prior / num_data
        log_posterior = (
                    log_likelihood
                    + log_prior
                )
        return log_posterior, (logits, log_likelihood, log_prior) 

    def nelbo(m, lsd, task, log_posterior, temperature, n_samples=1, stl=True):
        # compute the nelbo (negative evidence lower bound)
        sd_diag = tree_map(torch.exp, lsd)
        nelbo, aux = posteriors.vi.diag.nelbo(m, sd_diag, task, log_posterior, temperature, n_samples, stl)
        return nelbo

    def compute_nelbo_for_task(task, prior_params, prior_log_sd_diag, auxiliary_level_num=None, return_local_model=False):
        # detach parameters used as priors from gradient updates
        prior_params_detached = {p:prior_params[p].detach() for p in prior_params}
        prior_log_sd_diag_detached = {p:prior_log_sd_diag[p].detach() for p in prior_log_sd_diag}

        inner_opt = torchopt.sgd(lr=hyper_params['inner_learning_rate']) 
        partial_log_posterior_support_update = partial(log_posterior_support_update, prior_params_detached, prior_log_sd_diag_detached)
        inner_transform = posteriors.vi.diag.build(partial_log_posterior_support_update, inner_opt, temperature=hyper_params['inner_temperature'], init_log_sds=prior_log_sd_diag)
        
        # initialise local model by copying state from global model
        # initialise variational params (mean, sd) from global model
        inner_state = inner_transform.init(prior_params, init_log_sds=prior_log_sd_diag)

        X_tab_spt, X_tab_qry, X_long_spt, X_long_qry, y_spt, y_qry, imp_spt, imp_qry = task

        # for regular models, the start of batch is 0
        # for 3-level model, the start of batch is auxiliary_level_num * num_inner_updates * batch_size
        # this ensures that each auxiliary level gets a new batch of data
        start_of_batch = auxiliary_level_num*hyper_params['num_inner_updates']*args['batch_size'] if auxiliary_level_num is not None else 0

        for k in tqdm(range(hyper_params['num_inner_updates'])):
            
            start_idx = start_of_batch + k*args['batch_size'] 
            batch_indices = torch.arange(start_idx, start_idx+args['batch_size']).long()

            # draw a new batch of data for updating local model parameters
            task_batch = (
                X_tab_spt[batch_indices],
                X_tab_qry[batch_indices],
                X_long_spt[batch_indices],
                X_long_qry[batch_indices],
                y_spt[batch_indices],
                y_qry[batch_indices],
                imp_spt[batch_indices],
                imp_qry[batch_indices]
            )

            inner_state = inner_transform.update(inner_state, task_batch, inplace=False)
        
        if return_local_model: 
            # return the local model (e.g., to use later for predictions on new data)
            return inner_state 
        else:
            # compute nelbo for query data (used for updating global parameters)
            partial_log_posterior_query_update = partial(log_posterior_query_update, prior_params_detached, prior_log_sd_diag_detached)
            task_nelbo = nelbo(inner_state.params, inner_state.log_sd_diag, task_batch, partial_log_posterior_query_update, hyper_params['inner_temperature'])
            return task_nelbo
    

    def compute_auxiliary_nelbo(auxiliary_params, auxiliary_log_sd_diag, batch, kernel_weights, prior_params, prior_log_sd_diag, auxiliary_level_num, return_task_nelbos=False):
        # compute NELBOs for all tasks in the batch using vmap
        vmapped_task_models = func.vmap(compute_nelbo_for_task, in_dims=(0, None, None, None), randomness='different')

        task_nelbos = vmapped_task_models(batch, auxiliary_params, auxiliary_log_sd_diag, auxiliary_level_num)
        
        # add prior after taking (weighted) means of inner NELBOS for batch
        num_tasks_batch = len(kernel_weights)
        kernel_weights = kernel_weights / kernel_weights.mean() # average of weights should be 1 to maintain scale of likelihood
        batch_nelbo = torch.mul(task_nelbos, kernel_weights).sum()
        
        # add KL term: log p(gamma | theta) - outer_temperature * log q(gamma)

        # detach parameters used in prior N(global_mu, global_sigma)
        prior_params_detached = {p:prior_params[p].detach() for p in prior_params}
        prior_log_sd_diag_detached = {p:prior_log_sd_diag[p].detach() for p in prior_log_sd_diag}

        sd_diag_global = tree_map(torch.exp, prior_log_sd_diag_detached)
        log_p = multivariate_normal_prior(params, mean=prior_params_detached, sd_diag=sd_diag_global) / num_tasks_batch
        sd_diag_auxiliary = tree_map(torch.exp, auxiliary_log_sd_diag)
        sampled_params = posteriors.diag_normal_sample(auxiliary_params, sd_diag_auxiliary, sample_shape=(args['num_mc_samples'],))
        log_q = func.vmap(posteriors.diag_normal_log_prob, (0, None, None))(sampled_params, auxiliary_params, sd_diag_auxiliary)
        kl_term = -(log_p - log_q * hyper_params['auxiliary_temperature']).mean() 
        batch_nelbo += kl_term

        if return_task_nelbos:
            return batch_nelbo, task_nelbos.detach()

        return batch_nelbo


    def compute_nelbo_for_auxiliary_task(task, activated_tasks, kernel_weights, all_task_batch, outer_params, outer_log_sd_diag, return_local_model=False):
        auxiliary_opt = torchopt.sgd(lr=hyper_params['auxiliary_learning_rate'])
        partial_log_posterior_support_update = partial(log_posterior_support_update, outer_params, outer_log_sd_diag)
        auxiliary_transform = posteriors.vi.diag.build(partial_log_posterior_support_update, auxiliary_opt, temperature=hyper_params['auxiliary_temperature'], init_log_sds=outer_log_sd_diag) 
        
        # initialise auxiliary model by copying state from global model
        # initialise variational params (mean, sd) from global model
        auxiliary_state = auxiliary_transform.init(outer_params, init_log_sds=outer_log_sd_diag)
        
        # get batch of activated tasks (subset of all_task_batch)
        auxiliary_task_batch = [all_task_batch[i][(activated_tasks==1)] for i in range(len(all_task_batch))]
        auxiliary_task_kernel_weights = kernel_weights[(activated_tasks==1)]

        # update auxiliary model parameters using support data
        for auxiliary_level_num in range(hyper_params['num_auxiliary_updates']):
            with torch.no_grad():
                grads, nelbo_val = grad_and_value(compute_auxiliary_nelbo, argnums=(0, 1))(auxiliary_state.params, auxiliary_state.log_sd_diag, auxiliary_task_batch, auxiliary_task_kernel_weights, outer_params, outer_log_sd_diag, auxiliary_level_num)
            auxiliary_state = update_model(auxiliary_state, auxiliary_opt, grads, nelbo_val, inplace=False)
        
        if return_local_model:
            # return the local model (i.e. inner model of corresponding task for auxiliary task)
            inner_state = compute_nelbo_for_task(task, auxiliary_state.params, auxiliary_state.log_sd_diag, return_local_model=True)
            return inner_state, auxiliary_state
        else:
            # compute nelbo for query data (used for updating global parameters)
            auxiliary_nelbo, task_nelbos = compute_auxiliary_nelbo(auxiliary_state.params, auxiliary_state.log_sd_diag, auxiliary_task_batch, auxiliary_task_kernel_weights, outer_params, outer_log_sd_diag, auxiliary_level_num, return_task_nelbos=True)

            return auxiliary_nelbo

    
    def compute_batch_nelbo(outer_params, outer_log_sd_diag, batch_data, activated_tasks=None, kernel_weights=None, all_task_batch=None):
        # compute NELBOs for all tasks in the batch using vmap
        if args['method'] in ['2_level_hierarchical']:
            # for two level model, calculate task NELBOs for all tasks in minibatch
            vmapped_task_models = func.vmap(compute_nelbo_for_task, in_dims=(0, None, None), randomness='different')
            task_nelbos = vmapped_task_models(batch_data, outer_params, outer_log_sd_diag)
        elif args['method'] in ['3_level_hierarchical']:
            # for three level model, calculate auxiliary task NELBOs for all tasks in minibatch
            # don't use vmap here because it doesn't allow for dynamically shaped data
            task_nelbos = []
            for task_index, task in enumerate(batch_data):
                print("Computing NELBO update for task {}".format(task_index))
                task_nelbo = compute_nelbo_for_auxiliary_task(task, activated_tasks[task_index], kernel_weights[task_index], all_task_batch, outer_params, outer_log_sd_diag)
                task_nelbos.append(task_nelbo)
            
            task_nelbos = torch.stack(task_nelbos)

        # add global prior after taking means of inner NELBOS for batch
        batch_nelbo = torch.mean(task_nelbos)
        
        # add KL term: log p(theta) - outer_temperature * log q(theta)
        if hyper_params['nn_prior_name'] == 'multivariate_normal_prior':
            # uses prior distribution N(0, 1) for global parameters
            log_p = multivariate_normal_prior(params, mean=0, sd_diag=hyper_params["global_prior_sigma"]) / num_tasks
            # variational distribution is N(mean, sigma^2) from global parameters
            sd_diag = tree_map(torch.exp, outer_log_sd_diag) # need to take exponential of log(sd_diag) to get just sd_diag
            sampled_params = posteriors.diag_normal_sample(outer_params, sd_diag, sample_shape=(args['num_mc_samples'],))
            log_q = func.vmap(posteriors.diag_normal_log_prob, (0, None, None))(sampled_params, outer_params, sd_diag)
            kl_term = -(log_p - log_q * hyper_params['outer_temperature']).mean()
            batch_nelbo += kl_term

        return batch_nelbo

    def update_model(state, opt, grads, nelbo_val, inplace=False):
        # update the parameters for the global model,
        # use gradients accumulated over all local tasks
        updates, opt_state = opt.update(grads, state.opt_state, params=[state.params, state.log_sd_diag], inplace=inplace)
        mean, log_sd_diag = torchopt.apply_updates((state.params, state.log_sd_diag), updates, inplace=inplace)
        state = posteriors.vi.diag.VIDiagState(mean, log_sd_diag, opt_state, nelbo_val.detach())
        return state

    def get_task_data(mini_batch_tasks, epoch, task_label_map):
        # [X_spt, X_qry, X_long_spt, X_long_qry, y_spt, y_qry] = [(tasks, num_samples, num_predictors), (tasks, num_samples, num_predictors), (tasks, num_samples, sequence_length, num_features), (tasks, num_samples, sequence_length, features), (tasks, num_samples), (tasks, num_samples)]
        X_tab_spts, X_tab_qrys, X_long_spts, X_long_qrys, y_spts, y_qrys, imp_spts, imp_qrys = [], [], [], [], [], [], [], []
        
        # regular models need 2 * num_inner_updates * batch_size data per batch
        # 3-level models need 2 * num_inner_updates * num_auxiliary_updates * batch_size data per batch
        # note: x2 is to ensure there's enough data for support/query split
        num_samples_per_task = hyper_params['num_inner_updates'] * hyper_params['num_auxiliary_updates'] * args['batch_size'] * 2 if args['method'] in ['3_level_hierarchical'] else hyper_params['num_inner_updates'] * args['batch_size'] * 2

        for task in mini_batch_tasks:
            X_tab_spt, X_tab_qry, X_long_spt, X_long_qry, y_spt, y_qry, imp_spt, imp_qry = data.sample_supportquery_data(task_label_map[task.item()], num_samples_per_task, sample_seed=epoch, eval=eval)
            X_tab_spts.append(X_tab_spt)
            X_long_spts.append(X_long_spt)
            X_tab_qrys.append(X_tab_qry)
            X_long_qrys.append(X_long_qry)
            y_spts.append(y_spt)
            y_qrys.append(y_qry)
            imp_spts.append(imp_spt)
            imp_qrys.append(imp_qry)
        
        return [torch.Tensor(np.array(X_tab_spts)).to(args['device']), torch.Tensor(np.array(X_tab_qrys)).to(args['device']), torch.Tensor(np.array(X_long_spts)).to(args['device']), torch.Tensor(np.array(X_long_qrys)).to(args['device']), torch.Tensor(np.array(y_spts)).long().to(args['device']), torch.Tensor(np.array(y_qrys)).long().to(args['device']), torch.Tensor(np.array(imp_spts)).long().to(args['device']), torch.Tensor(np.array(imp_qrys)).long().to(args['device'])]

    def meta_learning_two_levels(data_loader, data_task_label_map, pred_task_loader, pred_task_label_map, outer_state, outer_opt, epoch, train=True, kernels=None):
        nelbos = []
        local_models = {}

        if train:
            
            for mini_batch_tasks in data_loader:
                
                task_data = get_task_data(mini_batch_tasks, epoch, data_task_label_map)
                
                # accumulate gradients for batch of tasks
                # NOTE these are second-order gradients, i.e., support data is used to derive local models for each task, then query data is used to derive task NELBO for updating global model
                with torch.no_grad():
                    grads, nelbo_val = grad_and_value(compute_batch_nelbo, argnums=(0, 1))(outer_state.params, outer_state.log_sd_diag, task_data) # computes gradient of NELBO for a batch of tasks wrt global parameters
                nelbos.append(nelbo_val.cpu().detach())

                # use accumulated gradients to update the global model
                outer_state = update_model(outer_state, outer_opt, grads, nelbo_val, inplace=False)

            # if training, return the updated global model
            epoch_nelbo = np.mean(nelbos)
            return outer_state, kernels, epoch_nelbo 
        
        else:
            # get the local models (states) for making predictions, do not need to update global model
            for task in pred_task_loader:
                task_label = pred_task_label_map[task.item()]
                print("Getting the local model for task {}".format(task_label))
                print("\t Fetching the task data")
                task_data = data.sample_supportquery_data(task_label, hyper_params['num_inner_updates'] * args['batch_size'] *2, sample_seed=epoch, eval=eval, as_tensor=True, device=args['device'])
                print("\t Computing the NELBO")
                local_model = compute_nelbo_for_task(task_data, outer_state.params, outer_state.log_sd_diag, return_local_model=True)
                local_models[task_label] = local_model
            
            models = {'local':local_models}
            return models
    

    def meta_learning_three_levels(data_loader, data_task_label_map, pred_task_loader, pred_task_label_map, outer_state, outer_opt, epoch, train=True, kernels=None):
        nelbos = []
        local_models = {}
        auxiliary_models = {}

        print("Preparing a new batch of data for training")

        all_task_data = get_task_data(torch.tensor(range(len(data.task_names))), epoch, data_task_label_map) # get batch of data for all training tasks, to use for updating auxiliary task

        if train:
            
            for mini_batch_tasks in data_loader:
                
                # for each task in the mini-batch, get the "activated" tasks and kernel weights (to use for the auxiliary task updates)
                activated_tasks, kernel_weights = kernels.get_activated_tasks_and_weights([data_task_label_map[task.item()] for task in mini_batch_tasks])
                
                if train:
                    # accumulate gradients for batch of auxiliary tasks
                    # NOTE these are third-order gradients, i.e., support data is used to derive local models for each task, then query data is used to derive task NELBO for updating auxiliary task and global models
                    with torch.no_grad():
                        grads, nelbo_val = grad_and_value(compute_batch_nelbo, argnums=(0, 1))(outer_state.params, outer_state.log_sd_diag, mini_batch_tasks, activated_tasks, kernel_weights, all_task_data) # computes gradient of NELBO for a batch of tasks wrt global parameters
                    nelbos.append(nelbo_val.cpu().detach())

                    # use accumulated gradients to update the global model
                    outer_state = update_model(outer_state, outer_opt, grads, nelbo_val, inplace=False)
            
            # if training, return the updated global model
            epoch_nelbo = np.mean(nelbos)
            return outer_state, kernels, epoch_nelbo
        
        else:
            # get the local models (states) for making predictions, do not need to update global model
            for task in pred_task_loader:
                task_label = pred_task_label_map[task.item()]
                print("Getting the local model for task {}".format(task_label))
                # get the "activated" tasks and kernel weights (to use for the auxiliary task updates)
                activated_tasks, kernel_weights  = kernels.get_activated_tasks_and_weights([task_label])
                num_samples_per_task = hyper_params['num_inner_updates'] * hyper_params['num_auxiliary_updates'] * args['batch_size'] * 2
                task_data = data.sample_supportquery_data(task_label, num_samples_per_task, sample_seed=epoch, eval=eval, as_tensor=True, device=args['device'])
                local_model, auxiliary_model = compute_nelbo_for_auxiliary_task(task_data, activated_tasks[0], kernel_weights[0], all_task_data, outer_state.params, outer_state.log_sd_diag, return_local_model=True)
                local_models[task_label] = local_model
                auxiliary_models[task_label] = auxiliary_model
            
            models = {'local':local_models, 'auxiliary':auxiliary_models}
            return models

    # initialise the global model
    if args['method'] == 'bnn_baseline':
        # doesn't need a separate outer learning rate (uses inner learning rate)
        hyper_params['outer_learning_rate'] = hyper_params['inner_learning_rate']
        hyper_params['outer_temperature'] = hyper_params['inner_temperature']

    outer_opt = torchopt.adam(lr=hyper_params['outer_learning_rate'])
    outer_transform = posteriors.vi.diag.build(log_posterior_query_update, outer_opt, temperature=hyper_params['outer_temperature'], n_samples=args['num_mc_samples']) 
    outer_state = outer_transform.init(params, init_log_sds=hyper_params["model_init_log_sds"])

    if args['method'] == 'bnn_baseline':
        meta_learning = meta_learning_two_levels # just use the inference part to fit local models, skip the main training
        kernels = None
    if args['method'] in ['2_level_hierarchical']:
        meta_learning = meta_learning_two_levels
        kernels = None
    elif args['method'] in ['3_level_hierarchical']:
        # NOTE for now only supports transductive setting
        kernels = CausalSimilarity(args['distancefile'], task_labels=data.task_names, eta=hyper_params['eta_1'], avg_num_active_tasks=hyper_params['avg_num_active_tasks'], device=args['device']) 
        data.get_task_weights(hyper_params['eta_2'])
        meta_learning = meta_learning_three_levels

    if args['method'] in ['2_level_hierarchical', '3_level_hierarchical']:
        print("Training the hierarchical model")
        epoch_nelbos = []
        
        # fit model using training set
        for epoch in range(args['max_num_epochs']):
            if args['method'] in ['3_level_hierarchical']:
                train_task_loader = data.get_taskloader(batchsize=args['minibatch_size'], shuffle=True, target_tasks=False, weighted=True)
            outer_state, kernels, epoch_nelbo = meta_learning(train_task_loader, data.task_name_map, pred_task_loader, data.target_task_name_map, outer_state, outer_opt, epoch, train=True, kernels=kernels)

            epoch_nelbos.append(epoch_nelbo)
            print(f"Epoch: {epoch}\t  NELBO (negative evidence lower bound): {epoch_nelbo}")
            if np.isnan(epoch_nelbo): # stop trial early if bad config leading to nan values
                return None

            # free up unused memory
            torch.cuda.empty_cache()
            gc.collect()

        # save the loss decay plot for global model
        plt.plot(epoch_nelbos)
        plt.title('ELBO (global model)')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.grid(True)
        plt.savefig('{}_global_elbo.png'.format(args['outprefix']))
        plt.clf()

    # get the final local models to use for making predictions on validation set
    if pred_for == 'val':
        print("Getting the local models for validation")
        local_models = meta_learning(train_task_loader, data.task_name_map, pred_task_loader, data.target_task_name_map, outer_state, outer_opt, 0, train=False, kernels=kernels)
        print("Getting the predictions for performance validation")
        val_task_preds = get_predictions(pred_task_loader, data.target_task_name_map, data, model, local_models, hyper_params, args, pred_for)
    elif pred_for == 'test':
        print("Getting the local models for testing")
        local_models = meta_learning(train_task_loader, data.task_name_map, pred_task_loader, data.target_task_name_map, outer_state, outer_opt, 0, train=False, kernels=kernels)
        print("Getting the predictions for performance evaluation")
        val_task_preds = get_predictions(pred_task_loader, data.target_task_name_map, data, model, local_models, hyper_params, args, pred_for)

        # save predictions for each task
        print("Saving the predictions for each task")
        pred_df = get_pred_table(val_task_preds)
        pred_df.to_csv('{}_pred_uncertainties.csv'.format(args['outprefix']), index=None)

        for task in pred_df['task'].unique():
            plt.hist(pred_df[pred_df['task']==task]['y_pred'])
            plt.title("Distribution of expected probabilities: task {}".format(task))
            plt.savefig('{}_pred_hist_{}.png'.format(args['outprefix'], task))
            plt.clf()

        # get embeddings for all patients in the entire dataset
        X_train, X_long_train, Y_train, train_ids, X_test, X_long_test, Y_test, test_ids = data.get_taskdata(task_label=None, eval=True, as_tensor=True, return_ids=True, device=args['device'])
        X_data = torch.cat((X_train, X_test))
        X_long_data = torch.cat((X_long_train, X_long_test))

        data_ids = train_ids + test_ids
        
        if args['method'] in ['2_level_hierarchical', '3_level_hierarchical']:
            # global model embeddings
            save_embeddings(model, outer_state, "global", X_data, X_long_data, data_ids, args)

            print("Saving the hierarchical model weights")
            
            model_to_save = {
                'params': outer_state.params,
                'log_sd_diag': outer_state.log_sd_diag,
                'opt_state': outer_state.opt_state,
                'nn_model_name': hyper_params['nn_model_name'],
                'num_features_tabular': len(data.predictors),
                'num_features_longitudinal': data.total_num_endpoints,
                'hidden_layer_size_tabular': hyper_params['hidden_layer_size_tabular'],
                'num_years_longitudinal': data.total_num_endpoints,
            }

            if args['data_type'] == 'sequence':
                model_to_save['hidden_layer_size_longitudinal'] = hyper_params['hidden_layer_size_longitudinal']
            elif args['data_type'] == 'tabular':
                model_to_save['n_layers'] = hyper_params['n_layers']

            torch.save(model_to_save,'{}_model_global.pth'.format(args['outprefix']))

        for batch in pred_task_loader:
            for task in batch:
                
                task_label = data.target_task_name_map[task.item()]
                local_state = local_models['local'][task_label]

                print("Saving the model weights for task {}".format(task_label))

                # local model embeddings for each task
                save_embeddings(model, local_state, "local_task_{}".format(task_label), X_data, X_long_data, data_ids, args)

                model_to_save = {
                    'params': local_state.params,
                    'log_sd_diag': local_state.log_sd_diag,
                    'opt_state': local_state.opt_state,
                    'nn_model_name': hyper_params['nn_model_name'],
                    'num_features_tabular': len(data.predictors),
                    'num_features_longitudinal': data.total_num_endpoints,
                    'hidden_layer_size_tabular': hyper_params['hidden_layer_size_tabular'],
                    'num_years_longitudinal': data.total_num_endpoints,
                }

                if args['data_type'] == 'sequence':
                    model_to_save['hidden_layer_size_longitudinal'] = hyper_params['hidden_layer_size_longitudinal']
                elif args['data_type'] == 'tabular':
                    model_to_save['n_layers'] = hyper_params['n_layers']

                torch.save(model_to_save,'{}_model_local_task_{}.pth'.format(args['outprefix'], task_label))

                if args['method'] in ['3_level_hierarchical']:
                    # auxiliary model embeddings and feature importance for each task
                    auxiliary_state = local_models['auxiliary'][task_label]
                    save_embeddings(model, auxiliary_state, "auxiliary_task_{}".format(task_label), X_data, X_long_data, data_ids, args)

                    model_to_save = {
                        'params': auxiliary_state.params,
                        'log_sd_diag': auxiliary_state.log_sd_diag,
                        'opt_state': auxiliary_state.opt_state,
                        'nn_model_name': hyper_params['nn_model_name'],
                        'num_features_tabular': len(data.predictors),
                        'num_features_longitudinal': data.total_num_endpoints,
                        'hidden_layer_size_tabular': hyper_params['hidden_layer_size_tabular'],
                        'num_years_longitudinal': data.total_num_endpoints,
                    }

                    if args['data_type'] == 'sequence':
                        model_to_save['hidden_layer_size_longitudinal'] = hyper_params['hidden_layer_size_longitudinal']
                    elif args['data_type'] == 'tabular':
                        model_to_save['n_layers'] = hyper_params['n_layers']

                    torch.save(model_to_save,'{}_model_auxiliary_task_{}.pth'.format(args['outprefix'], task_label))

    return val_task_preds


def to_sd_diag(state):
    return tree_map(lambda x: x.exp(), state.log_sd_diag)


def get_statistics(logits):
    # apply softmax function to convert logits into probabilities
    probs = torch.nn.functional.softmax(logits, dim=-1)
    # replaces probs smaller than 1e-5 with 1e-5 for numerical stability
    probs = torch.where(probs < 1e-5, 1e-5, probs) 
    # renormalize the probabilities to ensure they sum to 1
    probs /= probs.sum(dim=-1, keepdim=True)
    # calculates the mean of the probabilities, averaging over multiple samples
    expected_probs = probs.mean(dim=1)
    # gets probability of the positive class in a binary classification problem
    y_pred = expected_probs[:,1].cpu().detach().numpy().squeeze()
    # calculates the entropy of the expected probabilities, which is a measure of uncertainty
    # H(p) = -sum(p_i * log(p_i))
    # interpretation:
    # High Entropy (Close to maximum): The model is very uncertain about its prediction. In binary classification, this might mean probabilities close to [0.5, 0.5]. This could indicate that the input is ambiguous, lies on a decision boundary, or is unlike the training data.
    # Low Entropy (Close to 0): The model is very confident in its prediction. In binary classification, this might mean probabilities like [0.99, 0.01].However, be cautious of overconfidence, especially on out-of-distribution data.
    y_pred_entropy = -(torch.log(expected_probs) * expected_probs).mean(1).cpu().detach().numpy().squeeze()
    # decomposes the uncertainty into aleatoric uncertainty and epistemic uncertainty
    # Total Uncertainty: Calculated using the expected (average) probabilities across all Monte Carlo samples. Represents the overall predictive uncertainty.
    total_uncertainty = -(torch.log(expected_probs) * expected_probs).mean(1)
    # Aleatoric Uncertainty: Calculated by first computing the entropy for each Monte Carlo sample separately, then averaging. Captures the average uncertainty in individual predictions.
    aleatoric_uncertainty = -(torch.log(probs) * probs).mean(2).mean(1)
    # Epistemic Uncertainty: The difference between total and aleatoric uncertainty. Represents the variability in predictions across different Monte Carlo samples.
    epistemic_uncertainty = total_uncertainty - aleatoric_uncertainty

    aleatoric_uncertainty = aleatoric_uncertainty.cpu().detach().numpy().squeeze()
    epistemic_uncertainty = epistemic_uncertainty.cpu().detach().numpy().squeeze()
    y_pred_bin = expected_probs.argmax(dim=-1).cpu().detach().numpy().squeeze()

    batch_results = {
            'y_pred': y_pred,
            'y_pred_entropy': y_pred_entropy,
            'y_pred_bin': y_pred_bin,
            'aleatoric_uncertainty': aleatoric_uncertainty,
            'epistemic_uncertainty': epistemic_uncertainty
        }

    return batch_results


def forward(model, state, x_tabular, x_longitudinal, args):
    
    # sample multiple sets of parameters from the learned posterior distribution over the model parameters

    sd_diag = to_sd_diag(state)

    sampled_params = posteriors.diag_normal_sample(
        state.params, sd_diag, (args['num_mc_samples'],)
    )

    # apply the model with each set of parameters to the input data
    # collect all these outputs to approximate the posterior predictive distribution

    def model_func(p, x):
        return torch.func.functional_call(model, p, x)

    logits = torch.vmap(model_func, in_dims=(0, None))(sampled_params, (x_tabular, x_longitudinal)).transpose(
        0, 1
    )

    batch_results = get_statistics(logits)

    return batch_results


def get_predictions(taskloader, label_map, data, model, model_states, hyper_params, args, pred_for='val'):
    
    all_task_preds = {}
    
    # get predictions for each task
    for batch in taskloader:
        for task in batch:
            
            task_label = label_map[task.item()]
            state = model_states['local'][task_label]
            
            # get the task data to use for predictions
            if pred_for == 'val':
                # for validation, sample from the validation patients (WITHOUT class balancing)
                X_tab_spt, X_tab_qry, X_long_spt, X_long_qry, y_spt, y_qry, _, _ = data.sample_supportquery_data(task_label, args['batch_size']*10, sample_seed=0, eval=False, as_tensor=True, device=args['device'], dataset='val')
                X_data = torch.cat((X_tab_spt, X_tab_qry), dim=0)
                X_long_data = torch.cat((X_long_spt, X_long_qry), dim=0)
                y_data = torch.cat((y_spt, y_qry), dim=0)
            elif pred_for == 'test':
                # for testing, use the entire set of withheld test patients (no class balancing)
                print("Loading the data for evaluation of task {}".format(task_label))
                X_train, X_long_train, Y_train, train_ids, X_data, X_long_data, y_data, data_ids = data.get_taskdata(task_label, eval=True, as_tensor=True, device=args['device'], return_ids=True)
            
            # calculate metrics in batches otherwise get OOM errors
            results = {
                    'y_pred': [],
                    'y_pred_entropy': [],
                    'y_pred_bin': [],
                    'aleatoric_uncertainty': [],
                    'epistemic_uncertainty': []
                }

            print("Creating a dataset")
            eval_dataset = TensorDataset(X_data, X_long_data)
            print("Creating a data loader")
            eval_dataloader = DataLoader(eval_dataset, batch_size=args['batch_size'], shuffle=False)
            print("Running the inference in batches for task {}".format(task_label))
            total_batches = len(eval_dataloader)
            with tqdm(total=total_batches, desc=f"Task {task_label}") as pbar:
                for X_batch, X_long_batch in eval_dataloader:
                    batch_results = forward(model, state, X_batch, X_long_batch, args)
                    for key in results:
                        results[key].append(batch_results[key])
                    pbar.update(1)

            for key in results:
                results[key] = np.concatenate(results[key])

            results['y_actual'] = y_data.cpu()

            if pred_for == 'test': results['data_ids'] = data_ids
            all_task_preds[task_label] = results

    return all_task_preds


def hierarchical_objective(hyper_params, train_loader, task_loader, data, args):
    """Bayesian neural network objective for optuna trial
    """
    val_task_preds =  bnn_model(train_loader, task_loader, data, hyper_params, args, pred_for='val')

    return val_task_preds


def hierarchical_best_model(best_params, train_loader, task_loader, data, args):

    test_task_preds_best_model = bnn_model(train_loader, task_loader, data, best_params, args, pred_for='test')

    return test_task_preds_best_model