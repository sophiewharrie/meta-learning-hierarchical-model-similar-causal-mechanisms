import time
from jax import vmap
from numpyro.infer import MCMC, NUTS
from numpyro import handlers
from numpyro.infer import SVI, Trace_ELBO, Predictive
from numpyro.infer.autoguide import AutoNormal
from numpyro.optim import Adam
import jax.random as random

import sys
sys.path.insert(0, 'metrics')
from convergence import plot_losses

"""Utility classes for inference methods for BNNs in numpyro

There are two inference approaches available in this script:
1. Hamiltonian Monte Carlo (HMC)
2. Stochastic Variational Inference (SVI)

Each approach has a class with the following methods:
1. run_inference: for running inference on a training set to estimate weight posteriors
2. predict: for running prediction on a test set to estimate predictive posteriors
"""

class HMCInf:
    def __init__(self, model, rng_key, rng_key_predict, layer_sizes, num_warmup, num_samples, num_chains, posterior_samples=None):
        self.model = model 
        self.rng_key = rng_key 
        self.rng_key_predict = rng_key_predict
        self.layer_sizes = layer_sizes
        self.num_warmup = num_warmup
        self.num_samples = num_samples
        self.num_chains = num_chains
        self.posterior_samples = posterior_samples
        self.num_chains = num_chains

    def run_inference(self, X, Y):
        start = time.time()
        kernel = NUTS(self.model)
        mcmc = MCMC(kernel, num_warmup=self.num_warmup, num_samples=self.num_samples, num_chains=self.num_chains, progress_bar=True)
        if self.posterior_samples is not None:
            mcmc.run(self.rng_key, X, Y, self.layer_sizes, self.posterior_samples)
        else:
            mcmc.run(self.rng_key, X, Y, self.layer_sizes)
        print("\nMCMC elapsed time:", time.time() - start)
        samples = mcmc.get_samples()
        return samples

    def predict_helper(self, samples, X, rng_key):
        model = handlers.substitute(handlers.seed(self.model, rng_key), samples)
        if self.posterior_samples is not None:
            model_trace = handlers.trace(model).get_trace(X=X, Y=None, layer_sizes=self.layer_sizes, samples=self.posterior_samples)
        else:
            model_trace = handlers.trace(model).get_trace(X=X, Y=None, layer_sizes=self.layer_sizes)
        return model_trace["Y"]["value"]

    def predict(self, samples, X):
        vmap_args = (
            samples,
            random.split(self.rng_key_predict, self.num_samples * self.num_chains),
        )
        predictions = vmap(
            lambda samples, rng_key: self.predict_helper(samples, X, rng_key)
        )(*vmap_args)
        return predictions


class SVIInf:
    def __init__(self, model, rng_key, rng_key_predict, layer_sizes, learning_rate, num_steps, num_samples, outprefix, posterior_samples=None):
        self.model = model 
        self.rng_key = rng_key 
        self.rng_key_predict = rng_key_predict
        self.layer_sizes = layer_sizes
        self.learning_rate = learning_rate
        self.num_steps = num_steps
        self.num_samples = num_samples
        self.outprefix = outprefix
        self.posterior_samples = posterior_samples

    def run_inference(self, X, Y):
        optimizer = Adam(step_size=self.learning_rate)
        guide = AutoNormal(self.model)
        svi = SVI(self.model, guide, optimizer, loss=Trace_ELBO())

        if self.posterior_samples is not None:
            svi_result = svi.run(self.rng_key, self.num_steps, X, Y, self.layer_sizes, self.posterior_samples)
            params = svi_result.params
            predictive = Predictive(guide, params=params, num_samples=self.num_samples)
            post_samples = predictive(self.rng_key, X, Y, self.layer_sizes, self.posterior_samples)
            predictive = Predictive(self.model, guide=guide, params=params, num_samples=self.num_samples)
        else:
            svi_result = svi.run(self.rng_key, self.num_steps, X, Y, self.layer_sizes)
            params = svi_result.params
            predictive = Predictive(guide, params=params, num_samples=self.num_samples)
            post_samples = predictive(self.rng_key, X, Y, self.layer_sizes)
            predictive = Predictive(self.model, guide=guide, params=params, num_samples=self.num_samples)
        
        params = svi_result.params
        losses = svi_result.losses
        
        plot_losses(losses, self.outprefix)

        return predictive, post_samples

    def predict(self, predictive, X):
        if self.posterior_samples is not None:
            samples = predictive(self.rng_key, X=X, Y=None, layer_sizes=self.layer_sizes, samples=self.posterior_samples)['Y']
        else:
            samples = predictive(self.rng_key, X=X, Y=None, layer_sizes=self.layer_sizes)['Y']
        return samples
