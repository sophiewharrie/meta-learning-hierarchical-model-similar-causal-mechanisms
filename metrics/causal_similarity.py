import networkx as nx
import numpy as np
from cdt.metrics import SHD
from pyemd import emd_samples


def get_n_samples(task, n=100):
    """Helper function to sample n data points from the CGM
    """
    task_data = []
    nodes = task.node_list
    for sample in range(n):
        sample, _ = task.get_data_sample()
        task_data.append([sample[node] for node in nodes])
    return np.array(task_data)


def observational_distance(task1, task2):
    """Estimate observational distance using Earth mover distance (Wasserstein distance)
    from samples generated from the CGMs for each of the tasks

    Samples take the form of a 2d numpy array with dimension (nsamples, nnodes)

    [1] Ofir Pele and Michael Werman. Fast and robust earth mover's distances. Proc. 
    2009 IEEE 12th Int. Conf. on Computer Vision, Kyoto, Japan, 2009, pp. 460-467.

    [2] Peyrard, Maxime, and Robert West. "A ladder of causal distances." arXiv preprint arXiv:2005.02480 (2020).
    """
    samples1 = get_n_samples(task1)
    samples2 = get_n_samples(task2)
    return emd_samples(samples1, samples2)


def structural_hamming_distance(g1, g2):
    """Computes the SHD between two DAGs
    """
    A_g1 = nx.to_numpy_array(g1)
    A_g2 = nx.to_numpy_array(g2)
    return SHD(A_g1, A_g2, double_for_anticausal=True)

