import numpy as np
import networkx as nx
import random
import pandas as pd
import copy

from dags import get_task_DAG
from utils import make_graph
from functions import sigmoid_func

class Task:
    def __init__(self, task_number, group_number, A_ref, G_ref, sigma, functional_forms_ref, epsilon, node_list, interventions, args):
        self.task_number = task_number # identifier
        self.C = group_number # causal group assignment
        self.args = args
        self.node_list = node_list
        self.interventions = interventions

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


    def get_data_sample(self):
        # visit each node in topological order and 
        # sample a new value by summing over the neighbours sampled previously and the noise
        node_values = {}
        intervention_mask = {}
        # observational data occurs p_interv % of the time
        # interventional data occurs the other 1-p_interv % of the time
        observational_data = np.random.binomial(1, self.args['p_interv'])
        if observational_data: intervention_targets = [] 
        else: intervention_targets = random.choices([n for n in self.node_list if n!='Y']) # an intervention target is selected at random

        for node in self.sorted_nodes:
            neighbour_sum = self.get_neighbour_sum(node, node_values)
            node_values[node] = neighbour_sum + self.get_noise()
            intervention_mask[node] = 0
            if node in intervention_targets:
                node_values[node] = self.interventions[node]
                intervention_mask[node] = 1
            if node.startswith('Y'):
                node_values[node] = sigmoid_func(node_values[node]) # scale Y between 0 and 1
                if self.args['binary']:
                    node_values[node] = round(node_values[node])
        
        return node_values, intervention_mask


def simulate_task_data(task_number, metadata):
    # create CGM for task and simulate the data
    group_number = random.choice(range(metadata.args['C']))
    group_metadata = metadata.group_refs[group_number]
    task = Task(task_number, group_number, group_metadata.A, group_metadata.G, metadata.args['sigma_task'], group_metadata.functional_forms, metadata.args['eta_task'], metadata.node_list, metadata.interventions, metadata.args)
    total_num_samples = metadata.args['M_train'] + metadata.args['M_test']
    task_data = []
    interv_mask = []
    for _ in range(total_num_samples):
        data_sample, data_interv = task.get_data_sample()
        task_data.append(data_sample)
        interv_mask.append(data_interv)
    return task, task_data, interv_mask
