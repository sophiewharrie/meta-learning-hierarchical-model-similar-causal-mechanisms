import posteriors

def multivariate_normal_prior(x, mean, sd_diag, normalize=True):
    """Multivariate normal log probability for a diagonal covariance matrix

    Parameters:
    - x: value to evaluate log probability at (the model parameters)
    - mean: mean of the distribution (same shape as x, or if scalar, will be broadcast to same shape as x)
    - sd_diag: square-root diagonal of the covariance matrix (same shape as x, or if scalar, will be broadcast to same shape as x)
    """
    return posteriors.diag_normal_log_prob(x, mean=mean, sd_diag=sd_diag, normalize=normalize)

