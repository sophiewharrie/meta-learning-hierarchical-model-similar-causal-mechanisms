"""Model convergence metrics
"""
import matplotlib.pyplot as plt
import mlflow

def plot_losses(losses, outprefix):
    """Plot losses (e.g. from ELBO) for each optimization step.
    Give list of losses at each step as input
    """
    filename = f'{outprefix}_svi_losses.png'
    plt.plot(losses)
    plt.xlabel('Iteration')
    plt.ylabel('ELBO loss')
    plt.savefig(filename)
    mlflow.log_artifact(filename)
    plt.clf()