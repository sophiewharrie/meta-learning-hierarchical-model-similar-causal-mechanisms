import numpy as np
from sklearn.cluster import SpectralClustering
from sklearn.cluster import KMeans
import torch
import pickle
import networkx as nx
from cdt.metrics import SHD


def get_similarity_matrix(causal_models_for_tasks, args, delta=1): 
    num_tasks = len(causal_models_for_tasks)
    dist_matrix = np.zeros((num_tasks, num_tasks))
    indices = np.tril_indices(num_tasks, k=-1)

    # uncomment to save embeddings
    # embedding_list = []
    # for t in range(num_tasks):
    #     embedding_list.append(causal_models_for_tasks[t]['Z'].flatten())
    # np.savez('{}_embeddings.npz'.format(args['outprefix']), *embedding_list)
    # print('{}_embeddings.npz'.format(args['outprefix']))

    # all causal distances are symmetric - only compute once
    # distance between same task is 0
    for t1, t2 in zip(indices[0], indices[1]):
        causal_model1 = causal_models_for_tasks[t1]
        causal_model2 = causal_models_for_tasks[t2]
        if args['causal_distance']=='inferred_alt':
            sim_score = SHD(causal_model1['G'], causal_model2['G'])
        else:
            sim_score = euclidean_distance(causal_model1['Z'].flatten(), causal_model2['Z'].flatten())
        dist_matrix[t1,t2] = sim_score
        dist_matrix[t2,t1] = sim_score
    
    # convert distance matrix to similarity matrix using RBF kernel
    if np.max(dist_matrix) > 0: dist_matrix = dist_matrix / np.max(dist_matrix) # scale between 0 and 1
    sim_matrix = np.exp(- dist_matrix ** 2 / (2. * delta ** 2))

    return sim_matrix


def get_causal_for_Y(adj_matrix):
    """Return the variables (index of the column in the feature set) that have a direct (--> Y) causal relationship on Y

    Note that this assumes that the index of Y is the final variable in each row/column of the adj_matrix
    """
    # find Aij=1, where j is the final column (corresponding to Y)
    j = adj_matrix.shape[1] 
    ij_col = adj_matrix.T[j-1][:-1] # don't include Y itself
    return np.where(ij_col==1)[0]


def get_representative_causal_models(causal_models_for_tasks, causal_groups, args):
    """
    Given the causal models for each task and the group assignments for each task,
    determine the representative causal model for each group

    Note that the order of nodes in the adjacency matrix is the same as the feature input, plus the final node is Y
    """
    num_groups = len(set(causal_groups))
    representative_models = []
    for group in range(num_groups):
        group_tasks = np.where(np.array(causal_groups)==group)[0]
        if args['causal_distance']=='diagnostic':
            group_models_param = [np.array(torch.concat([causal_models_for_tasks[t]['weights'], causal_models_for_tasks[t]['biases']])) for t in group_tasks]
            mean_param = np.mean(group_models_param, axis=0)
            representative_models.append({'param':mean_param})
        else:
            group_models_G = np.array([causal_models_for_tasks[t]['G'] for t in group_tasks])
            median_G = (np.median(group_models_G, axis=0)>=0.5).astype(int)
            group_models_Z = np.array([causal_models_for_tasks[t]['Z'] for t in group_tasks])
            mean_Z = np.mean(group_models_Z, axis=0)
            representative_models.append({'Z':mean_Z, 'G':median_G, 'causal_feat':get_causal_for_Y(median_G)})
    return representative_models


def get_groundtruth_representative_causal_models(train_loader, causal_groups_train, args):
    """Returns the ground truth causal models (assuming the .pkl file is present, generated for synthetic data)

    Note that the order of nodes in the adjacency matrix is the same as the feature input, plus the final node is Y
    """
    representative_models = []
    node_order = train_loader.dataset.features + ['Y']
    graphpath = args['datafile'][0:-8] + 'graphs.pkl' # NOTE assumes this file is present, which is created with our data simulator
    with open(graphpath, 'rb') as f:
        graph_data = pickle.load(f)
    for group in range(args['num_groups']):
        group_tasks = np.where(np.array(causal_groups_train) == group)[0]
        causal_models_for_tasks_G = [nx.from_dict_of_dicts(graph_data[task], create_using=nx.DiGraph()) for task in group_tasks]
        causal_models_for_tasks_A = np.array([nx.adjacency_matrix(G, nodelist=node_order).todense() for G in causal_models_for_tasks_G])
        group_model_A = (np.median(causal_models_for_tasks_A , axis=0)>=0.5).astype(int)
        representative_models.append({'G':group_model_A, 'causal_feat':get_causal_for_Y(group_model_A)})
    return representative_models


def get_diagnostic_causal_groups(causal_models_for_tasks, args):
    """Simple baseline groups using parameter similarity
    """
    print("Using diagnostic model for similar tasks")
    parameters = []
    for t in range(len(causal_models_for_tasks)):
        task_parameters = torch.concat([causal_models_for_tasks[t]['weights'], causal_models_for_tasks[t]['biases']])
        parameters.append(np.array(task_parameters))
    
    parameters = np.array(parameters)
    kmeans = KMeans(n_clusters=args['num_groups']).fit(parameters)
    causal_groups = kmeans.labels_
    return causal_groups


def get_causally_similar_tasks(causal_models_for_tasks, args):
    if args['causal_distance']=='diagnostic': 
        causal_groups = get_diagnostic_causal_groups(causal_models_for_tasks, args)
    else:
        similarity_matrix = get_similarity_matrix(causal_models_for_tasks, args, delta=1)
        clustering = SpectralClustering(n_clusters=args['num_groups'], affinity='precomputed')
        clustering.fit(similarity_matrix)
        causal_groups = clustering.labels_
    representative_models = get_representative_causal_models(causal_models_for_tasks, causal_groups, args)
    return causal_groups, representative_models


def euclidean_distance(a, b):
    return np.linalg.norm(a - b)


def calculate_distance(group_model, causal_model_for_new_task, args):
    if args['causal_distance']=='diagnostic':
        distance = euclidean_distance(group_model['param'], np.array(torch.concat([causal_model_for_new_task['weights'], causal_model_for_new_task['biases']])))
    elif args['causal_distance']=='inferred_alt':
        distance = SHD(group_model['G'], causal_model_for_new_task['G'])
    else:
        distance = euclidean_distance(group_model['Z'].flatten(), causal_model_for_new_task['Z'].flatten())
    return distance


def pred_causal_group_new_task(causal_model_for_new_task, group_causal_models, args):
    distances = []
    for group_model in group_causal_models:
        distances.append(calculate_distance(group_model, causal_model_for_new_task, args))

    # get group with minimum distance to task model    
    group = np.argmin(distances)
    return group


def pred_causal_group_new_tasks(causal_models_for_new_tasks, group_causal_models, args):
    """
    Given representative causal models for each group, assign causal groups for new tasks based on group causal model with shortest distance to task causal model
    """
    causal_groups = []
    for task in range(len(causal_models_for_new_tasks)):
        causal_groups.append(pred_causal_group_new_task(causal_models_for_new_tasks[task], group_causal_models, args))
    return causal_groups