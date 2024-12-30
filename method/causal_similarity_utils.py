import numpy as np
import torch
import pandas as pd
import torch.nn.functional as F

class CausalSimilarity:
    def __init__(self, distancepath, task_labels, eta=1, avg_num_active_tasks=1, device=None):
        """
        Class for managing the causal similarity and kernel calculations

        Each task has a task-specific kernel, parameterised by the task value of kernel parameter eta
        """
        # Create a mapping from task names to their indices in training_tasks
        self.device = device
        self.task_to_index = {task: idx for idx, task in enumerate(task_labels)}
        self.weight_dict, self.activated_tasks = self.load_weights(distancepath, task_labels, self.task_to_index, eta, avg_num_active_tasks)


    def load_weights(self, distancepath, all_tasks, task_to_index, eta, avg_num_active_tasks):
        """
        Get lists of weights and activated tasks for each task

        A task t2 is "activated" for t1 (i.e. used as a relevant task) if w[task1, task2] > c

        Each list must be the same length as the number of training tasks
        """
        # Read the distance CSV file
        distance_df = pd.read_csv(distancepath)
        # Normalize weights between 0 and 1 and convert to similarity
        # Apply modulating factor: eta = 1: no effect on weights; 0 < eta < 1: toward equal weight for all tasks; eta > 1: more weight on most similar tasks
        distance_df['value'] = 1 - ((distance_df['value']- distance_df['value'].min()) / (distance_df['value'].max() - distance_df['value'].min()))
        distance_df['value'] = distance_df['value'] ** eta
        
        # Create a dictionary to store the weight tensors for each task
        weight_dict = {task: torch.zeros(len(all_tasks), dtype=torch.float32) for task in all_tasks}
        
        # Populate the weight dictionary
        for _, row in distance_df.iterrows():
            task1, task2, value = row['task1'], row['task2'], row['value']
            if task1 in all_tasks and task2 in all_tasks:
                idx1 = task_to_index[task1]
                idx2 = task_to_index[task2]
                # assume summetry
                weight_dict[task1][idx2] = value
                weight_dict[task2][idx1] = value

        activated_tasks = {}

        # Task is only activated if weight >= c (where c is a small value)
        all_weights = torch.stack(list(weight_dict.values())).flatten()
        num_tasks = len(all_tasks)
        num_weights = num_tasks * num_tasks
        # find c for which the average number of activated tasks per task is avg_num_active_tasks
        total_activations = num_tasks * avg_num_active_tasks
        c = torch.sort(all_weights)[0][num_weights - total_activations].item()
        print("Using threshold value {}".format(c))

        # Create vector of activated tasks using threshold c
        for task in all_tasks:
            activated_tasks[task] = (weight_dict[task] >= c).float()
            print("Using {} activated tasks for task {}".format(activated_tasks[task].sum(), task))
            # Set the weights of non-activated tasks to 0 and ensure the average weight (of activated tasks) is 1 (to maintain scale of likelihood)
            weight_dict[task] = torch.where(activated_tasks[task].bool(), weight_dict[task], torch.zeros_like(weight_dict[task]))
            weight_dict[task] = weight_dict[task] / (weight_dict[task].sum() / activated_tasks[task].sum())
            print(weight_dict[task])

        return weight_dict, activated_tasks


    def get_activated_tasks_and_weights(self, list_of_task_labels):
        """
        Get lists of activated tasks and kernel weights for each (auxiliary) task in list_of_task_labels

        A task is "activated" (i.e. used in the auxiliary task) if K[auxiliary task, task] > c for the auxiliary task kernel K

        Each list must be the same length as the number of training tasks

        NOTE the target tasks are not included in the kernel calculations
        """
        # returns tasks in order of task_labels used when initializing this class
        activated_tasks_list = []
        weights_list = []
        for task_label in list_of_task_labels:
            weights_list.append(self.weight_dict[task_label])
            activated_tasks_list.append(self.activated_tasks[task_label])
        
        activated_tasks = torch.stack(activated_tasks_list).to(self.device)
        weights = torch.stack(weights_list).to(self.device)

        return activated_tasks, weights
