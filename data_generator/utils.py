"""Utility functions for synthetic data
"""

import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

def make_graph(A, G_ref):
    """Makes networkx graph object from numpy adjacency matrix A
    
    Assumes node ordering is same as for reference G_ref
    """
    G = nx.from_numpy_array(A, create_using=nx.DiGraph)
    node_labels = list(G_ref.nodes())
    G = nx.relabel_nodes(G, dict(zip(range(len(node_labels)),node_labels)))
    
    return G


def draw_graph(G, prefix):
    nx.draw(G, with_labels=True)
    filepath = '{}_dag.png'.format(prefix)
    plt.savefig(filepath)
    plt.clf()
    return filepath
    

def valid_DAG(A, G_ref):
    """Returns true if the grpah is a valid DAG
    """
    if A is None:
        return False
    
    G = make_graph(A, G_ref)
    # should be a valid DAG
    if not nx.is_directed_acyclic_graph(G):
        return False

    return True