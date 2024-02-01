import torch
import torch.nn as nn
import numpy as np


class PriorWeightDistribution(nn.Module):
    #Calculates a Scale Mixture Prior distribution for the prior part of the complexity cost on Bayes by Backprop paper
    def __init__(self,
                 pi=1,
                 sigma1=0.1,
                 sigma2=0.001,
                 dist=None):
        super().__init__()


        if (dist is None):
            self.pi = pi
            self.sigma1 = sigma1
            self.sigma2 = sigma2
            self.dist1 = torch.distributions.Normal(0, sigma1)
            self.dist2 = torch.distributions.Normal(0, sigma2)

        if (dist is not None):
            self.pi = 1
            self.dist1 = dist
            self.dist2 = None

        

    def log_prior(self, w):
        """
        Calculates the log_likelihood for each of the weights sampled relative to a prior distribution as a part of the complexity cost

        returns:
            torch.tensor with shape []
        """
        prob_n1 = torch.exp(self.dist1.log_prob(w))

        if self.dist2 is not None:
            prob_n2 = torch.exp(self.dist2.log_prob(w))
        if self.dist2 is None:
            prob_n2 = 0
        
        # Prior of the mixture distribution, adding 1e-6 prevents numeric problems with log(p) for small p
        prior_pdf = (self.pi * prob_n1 + (1 - self.pi) * prob_n2) + 1e-6

        return (torch.log(prior_pdf) - 0.5).sum()



def update_prediction_model_priors(model_to_update, model_with_priors, args):
    """Updates the priors of model using the variational distribution learned by model_with_priors
    """
    weight_dim = model_with_priors[0].weight_mu.flatten().shape[0]
    bias_dim = model_with_priors[0].bias_mu.shape[0]
    # N(mu, sigma*I) where covariance matrix is a diagonal matrix and sigma is obtained by converting from rho
    weight_cov_matrix = torch.eye(weight_dim).to(args['device'])*torch.log1p(torch.exp(model_with_priors[0].weight_rho.flatten())) # convert rho to sigma and make diagonal matrix by * Identity
    model_to_update[0].weight_prior_dist = PriorWeightDistributionMVN(mean=model_with_priors[0].weight_mu.flatten(), covariance_matrix=weight_cov_matrix, name='mean')
    bias_cov_matrix = torch.eye(bias_dim).to(args['device'])*torch.log1p(torch.exp(model_with_priors[0].bias_rho))
    model_to_update[0].bias_prior_dist = PriorWeightDistributionMVN(mean=model_with_priors[0].bias_mu, covariance_matrix=bias_cov_matrix, name='bias')
    return model_to_update


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
