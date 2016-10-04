import function
import numpy

class Generator:

    def __init__(self, distribution, samples):
        self.dist = distribution
        self.samples = samples

    def populate_zero_one(self):
        self.samples = numpy.random.rand(self.samples)

    def eval(self):
        return self.dist.eval(self.samples)