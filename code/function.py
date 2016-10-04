import numpy

class Function:

    def __init__(self, constants):
        self.constants = constants
        self.name = 'abstract_func'

    def __str__(self):
        return self.name

    def eval(self, variables):
        return None

class ExponentialDist(Function):

    def __init__(self, rate_parameter):
        self.rate = rate_parameter
        self.name = 'exponential_dist'

    def eval(self, variables):
        return self.rate * numpy.exp(-1*self.rate*variables)
