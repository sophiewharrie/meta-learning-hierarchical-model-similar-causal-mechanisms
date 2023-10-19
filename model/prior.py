import torch
import numpy as np
import torch.nn as nn
import torch.functional as F

# NOTE this is the prior for the base learner parameters, which we need to manually override

class PriorWeightDistributionMVN(nn.Module):
    def __init__(self,
                 mean,
                 covariance_matrix, 
                 name):
        super().__init__()
        
        self.dist = torch.distributions.MultivariateNormal(mean, covariance_matrix)
        self._param_names = f'baselearner_prior_{name}'
        
    def log_prior(self, w):
        """
        Calculates the log_likelihood for each of the weights sampled relative to a prior distribution as a part of the complexity cost

        returns:
            torch.tensor with shape []
        """
        return self.dist.log_prob(w.flatten()).sum()
