import numpy as np
import networkx as nx
import random
import pandas as pd
import copy

from dags import get_task_DAG
from utils import make_graph
from functions import sigmoid_func

class Task:
    def __init__(self, task_number, group_number, A_ref, G_ref, sigma, functional_forms_ref, epsilon, node_list, args):
        self.task_number = task_number # identifier
        self.C = group_number # causal group assignment
        self.args = args
        self.node_list = node_list

        # generate CGM for task
        self.functional_forms = self.get_task_params(functional_forms_ref, sigma) 
        self.A, self.G = self.get_task_graph(A_ref, G_ref, epsilon) 


    def get_task_params(self, functional_forms, sigma):
        group_functional_forms = copy.deepcopy(functional_forms)
        for node in functional_forms:
            group_functional_forms[node]['alpha'] = np.random.normal(functional_forms[node]['alpha'], sigma)
        return group_functional_forms


    def get_task_graph(self, A_ref, G_ref, epsilon):
        A = get_task_DAG(A_ref, G_ref, epsilon)
        G = make_graph(A, G_ref)
        self.sorted_nodes = list(nx.topological_sort(G)) # sort nodes to make it easy to sample
        return A, G


    def get_noise(self):
        return np.random.normal(0, self.args['sigma_noise'])


    def get_neighbour_sum(self, node, node_values):
        node_neighbours = [edge[0] for edge in self.G.in_edges(node)]
        neighbour_sum = 0
        for neighbour in node_neighbours:
            func = self.functional_forms[neighbour]['func']
            alpha = self.functional_forms[neighbour]['alpha']
            x = node_values[neighbour]
            neighbour_sum += func(x, alpha)
        return neighbour_sum


    def get_data_sample(self, hard_intervention=False, intervention_node=None):
        """
        If hard_intervention is True, the data sample will have X=1 for the node X specified by intervention_node

        If hard_intervention is False, it is still possible for an intervention node to be non-zero (based on the SCM)

        NOTE: assumes only one intervention can be acted on in a single data sample
        """
        # visit each node in topological order and 
        # sample a new value by summing over the neighbours sampled previously and the noise
        node_values = {}
        for node in self.sorted_nodes:
            neighbour_sum = self.get_neighbour_sum(node, node_values)
            node_values[node] = neighbour_sum + self.get_noise()
            # make T variables binary (apply sigmoid function and round to 0 or 1)
            if node.startswith('T'):
                if hard_intervention:
                    node_values[node] = 1 if node == intervention_node else 0
                else:
                    # assume only one treatment can be acted on at a time - 
                    # check if another treatment is already acted on
                    already_treatment = any(v == 1 for k, v in node_values.items() if k.startswith('T'))
                    if already_treatment: node_values[node] = 0
                    else: node_values[node] = round(sigmoid_func(node_values[node]))
            # scale Y variables between 0 and 1 (apply sigmoid function)
            if node.startswith('Y'):
                node_values[node] = sigmoid_func(node_values[node])
                if self.args['binary']:
                    node_values[node] = round(node_values[node])

        return node_values


def simulate_task_data(task_number, metadata):
    # create CGM for task and simulate the data
    group_number = random.choice(range(metadata.args['C']))
    group_metadata = metadata.group_refs[group_number]
    task = Task(task_number, group_number, group_metadata.A, group_metadata.G, metadata.args['sigma_task'], group_metadata.functional_forms, metadata.args['eta_task'], metadata.node_list, metadata.args)
    total_num_samples = metadata.args['M_train'] + metadata.args['M_test']
    task_data = []
    for sample_idx in range(total_num_samples):
        # NOTE assumes equal probability for each hard intervention or no intervention
        intervention_node_list = [None] + metadata.T_vars
        intervention_node = random.choice(intervention_node_list) # randomly sample a hard intervention or no intervention
        if intervention_node is not None: data_sample = task.get_data_sample(hard_intervention=True, intervention_node=intervention_node)
        else: data_sample = task.get_data_sample()
        task_data.append(data_sample)
    return task, task_data
