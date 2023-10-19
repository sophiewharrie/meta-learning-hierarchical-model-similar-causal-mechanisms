from jax import vmap
import jax.numpy as jnp
import jax.random as random

import numpyro
import numpyro.distributions as dist

from numpyro_utils import HMCInf
from numpyro_utils import SVIInf


def bnn_regression(X, Y, layer_sizes):
    """
    A two-layer Bayesian Neural Network for regression

    Ref. https://num.pyro.ai/en/stable/examples/bnn.html
    
    Params:
    X: numpy array of shape (nsamples, nfeatures)
    Y: numpy array of shape (nsamples,)
    layer_sizes: numpy array of shape (nfeatures, hidden1_size, hidden2_size, ...)
    """
    D_X, D_Y = X.shape[1], 1

    z = X
    # sample each of the hidden layers (we put unit normal priors on all weights)
    for i, (D_in, D_out) in enumerate(zip(layer_sizes[:-1], layer_sizes[1:])):
        w = numpyro.sample(f"w{i}", dist.Normal(jnp.zeros((D_in, D_out)), jnp.ones((D_in, D_out)))) 
        z = jnp.tanh(jnp.matmul(z, w)) 

    # sample final layer of weights and neural network output
    w_final = numpyro.sample(f"w_final", dist.Normal(jnp.zeros((D_out, D_Y)), jnp.ones((D_out, D_Y))))
    z_final = jnp.matmul(z, w_final).squeeze(-1)
    
    # we put a prior on the observation noise
    prec_obs = numpyro.sample("prec_obs", dist.Gamma(3.0, 1.0))
    sigma_obs = 1.0 / jnp.sqrt(prec_obs)
    Y = numpyro.sample("Y", dist.Normal(z_final, sigma_obs), obs=Y)


def bnn_classification(X, Y, layer_sizes):
    """
    A two-layer Bayesian Neural Network for classification

    Ref. https://num.pyro.ai/en/stable/examples/bnn.html
    
    Params:
    X: numpy array of shape (nsamples, nfeatures)
    Y: numpy array of shape (nsamples,)
    layer_sizes: numpy array of shape (nfeatures, hidden1_size, hidden2_size, ...)

    Return is of form y_pred_mean, y_pred_samples, y_true, y_task
    """
    D_X, D_Y = X.shape[1], 1

    z = X
    # sample each of the hidden layers (we put unit normal priors on all weights)
    for i, (D_in, D_out) in enumerate(zip(layer_sizes[:-1], layer_sizes[1:])):
        w = numpyro.sample(f"w{i}", dist.Normal(jnp.zeros((D_in, D_out)), jnp.ones((D_in, D_out)))) 
        z = jnp.tanh(jnp.matmul(z, w)) 

    # sample final layer of weights and neural network output
    w_final = numpyro.sample(f"w_final", dist.Normal(jnp.zeros((D_out, D_Y)), jnp.ones((D_out, D_Y))))
    z_final = jnp.matmul(z, w_final).squeeze(-1)

    # bernoulli likelihood for binary classification
    Y = numpyro.sample("Y", dist.Bernoulli(logits=z_final), obs=Y)


def train_bnn(X_train, Y_train, outpath, regression=True, inference_method='HMC', num_warmup=1000, num_samples=500, num_chains=1, hidden_layers=[5,5], learning_rate=0.001, num_steps=10000):
    layer_sizes = (X_train.shape[1], *hidden_layers)
    rng_key, rng_key_predict = random.split(random.PRNGKey(0))
    model = bnn_regression if regression else bnn_classification

    if inference_method=='HMC':
        print("Using HMC inference")
        inference = HMCInf(model, rng_key, rng_key_predict, layer_sizes, num_warmup, num_samples, num_chains)
        samples = inference.run_inference(X_train, Y_train)
        return inference, samples
    elif inference_method=='SVI':
        print("Using SVI inference")
        inference = SVIInf(model, rng_key, rng_key_predict, layer_sizes, learning_rate, num_steps, num_samples, outpath)
        predictive, post_samples = inference.run_inference(X_train, Y_train)
        return inference, predictive 


def predict_bnn(X_test, Y_test, task_test, inference_model, input):
    predictions = inference_model.predict(input, X_test)
    # compute mean prediction
    mean_prediction = jnp.mean(predictions, axis=0)
    return mean_prediction, predictions, Y_test, task_test