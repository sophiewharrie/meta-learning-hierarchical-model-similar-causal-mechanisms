import networkx as nx
import numpy as np
from cdt.metrics import SHD
from pyemd import emd_samples


def get_n_samples(task, n=100, hard_intervention=False, intervention_node=None):
    """Helper function to sample n data points from the CGM
    """
    task_data = []
    nodes = task.node_list
    for sample in range(n):
        sample = task.get_data_sample(hard_intervention, intervention_node)
        task_data.append([sample[node] for node in nodes])
    return np.array(task_data)


def observational_distance(task1, task2, hard_intervention=False, intervention_node=None):
    """Estimate observational distance using Earth mover distance (Wasserstein distance)
    from samples generated from the CGMs for each of the tasks

    Samples take the form of a 2d numpy array with dimension (nsamples, nnodes)

    [1] Ofir Pele and Michael Werman. Fast and robust earth mover's distances. Proc. 
    2009 IEEE 12th Int. Conf. on Computer Vision, Kyoto, Japan, 2009, pp. 460-467.

    [2] Peyrard, Maxime, and Robert West. "A ladder of causal distances." arXiv preprint arXiv:2005.02480 (2020).
    """
    samples1 = get_n_samples(task1, hard_intervention=hard_intervention, intervention_node=intervention_node)
    samples2 = get_n_samples(task2, hard_intervention=hard_intervention, intervention_node=intervention_node)
    return emd_samples(samples1, samples2)


def interventional_distance(task1, task2):
    """Estimate interventional distance using Earth mover distance (Wasserstein distance)
    from samples generated from the CGMs for each of the tasks

    Assumes you can perform hard interventions on the data to get samples under each intervention

    Samples take the form of a 3d numpy array with dimension (ninterventions, nsamples_per_intervention, nnodes)

    [1] Ofir Pele and Michael Werman. Fast and robust earth mover's distances. Proc. 
    2009 IEEE 12th Int. Conf. on Computer Vision, Kyoto, Japan, 2009, pp. 460-467.

    [2] Peyrard, Maxime, and Robert West. "A ladder of causal distances." arXiv preprint arXiv:2005.02480 (2020).
    """
    # calculate observational distance for each intervention and take the average
    # (assumes interventions are sampled uniformly)
    interventions = [v for v in task1.node_list if v.startswith('T')]
    obs_distances = []
    for intervention in interventions:
        obs_distances.append(observational_distance(task1, task2, hard_intervention=True, intervention_node=intervention))
    # also append a distance for the no intervention case
    obs_distances.append(observational_distance(task1, task2))
    int_distance = np.mean(obs_distances)
    return int_distance


def structural_hamming_distance(g1, g2):
    """Computes the SHD between two DAGs
    """
    A_g1 = nx.to_numpy_array(g1)
    A_g2 = nx.to_numpy_array(g2)
    return SHD(A_g1, A_g2, double_for_anticausal=True)


def structural_intervention_distance(g1, g2):
    """Computes the SID between two DAGs, using the symmetrical variation of SID discussed in [1]

    [1] Peters, J., & Bühlmann, P. (2015). Structural intervention distance for evaluating 
    causal graphs. Neural computation, 27(3), 771-799.
    """
    return (compute_sid(g1, g2) + compute_sid(g2, g1)) / 2


def compute_path_matrix(G):
    """The path matrix is a p x p matrix, where p is the number of nodes.
    The entry (i,j) is one if and only if there is a directed path from i to j.
    """
    floyd_warshall = nx.floyd_warshall_numpy(G)
    path_matrix = ((floyd_warshall > 0) & (floyd_warshall != np.inf)).astype(int)
    return path_matrix


def roundp(G, i):
    """Compute all nodes that can be reached on a non-directed path.
    Returns a numpy list of length p, where entry j is 1 if node j can be reached on a
    non-directed path from node i in graph G.
    """
    G_und = G.to_undirected()
    nodes_list = list(G.nodes())
    reachable = list(nx.node_connected_component(G_und, nodes_list[i]))
    return np.array([int(r in reachable) for r in nodes_list])


def compute_sid(G, H):
    """Implementation of SID, as per Algorithm 1 described in [1]

    [1] Peters, J., & Bühlmann, P. (2015). Structural intervention distance for evaluating 
    causal graphs. Neural computation, 27(3), 771-799.
    """
    node_list = list(G.nodes())
    G_A = nx.to_numpy_array(G)
    p = G_A.shape[0]

    incorrect_causal_effects = np.zeros((p,p))
    path_matrix = compute_path_matrix(G)

    for i in range(p):
        node = node_list[i]
        pa_G = np.array(list(G.predecessors(node))) # parents of i in G
        pa_H = np.array(list(H.predecessors(node))) # parents of i in H
        pa_H_idx = np.array([node_list.index(x) for x in pa_H])
        
        reachable_on_non_directed_path = roundp(G, i)

        for j in range(p):
            if j != i:
                ijGNull, ijHNull, finished = False, False, False
                if path_matrix[i,j] == 0:
                    ijGNull = True # G predicts the causal effect to be zero
                if j in pa_H:
                    ijHNull = True # H predicts the causal effect to be zero
                if not ijGNull and ijHNull:
                    incorrect_causal_effects[i,j] = 1
                    finished = True # one mistake if only H predicts zero
                if ijGNull and ijHNull or (len(pa_G)==len(pa_H) and (pa_G == pa_H).all()):
                    finished = True # no mistakes if both predictions coincide
                if not finished:
                    # children of i in G that have j as a descendant
                    children_on_directed_path = np.array([node_list.index(c) for c in G.successors(node_list[i]) if node_list[j] in nx.descendants(G, c)])
                    if len(children_on_directed_path)>0 and len(pa_H_idx)>0 and np.sum(path_matrix[children_on_directed_path][:, pa_H_idx])>0:
                        incorrect_causal_effects[i,j] = 1
                    if reachable_on_non_directed_path[j] == 1:
                        incorrect_causal_effects[i,j] = 1

    sid = incorrect_causal_effects.sum()
    return sid