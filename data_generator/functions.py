"""Mathematical function utilities
"""

import numpy as np
import random
import math


def sigmoid_func(x):
  return 1 / (1 + math.exp(-x))

def linear_func(x, alpha):
    return alpha*x

def quadratic_func(x, alpha):
    return alpha*(np.square(x))

def cubic_func(x, alpha):
    return alpha*(np.power(x,3))

def get_functional_form(args):
    func_options = [linear_func, quadratic_func, cubic_func]
    alpha = np.random.normal(0,args['sigma_ref'])
    if args['linear']:
        func = {'func':linear_func, 'alpha':alpha}
    else:
        func = {'func':random.choice(func_options), 'alpha':alpha}
    return func