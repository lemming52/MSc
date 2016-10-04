import numpy as np

class Function:
    # Abstract (is this a thing?) class for any kind of mathematical function

    def __init__(self, constants):
        self.constants = constants
        self.name = 'abstract_func'

    def __str__(self):
        return self.name

    def eval(self, variables):
        return None

class ExponentialDist(Function):
    # f(x) = lambda * e^(-lambda*x)
    # evaluate for x
    # takes numpy arrays easily

    def __init__(self, rate_parameter):
        self.rate = rate_parameter
        self.name = 'exponential_dist'

    def eval(self, variables):
        return self.rate * np.exp(-1*self.rate*variables)
