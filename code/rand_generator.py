import function
import numpy as np

class Generator:
    # Take a mathematical function and a sample space.
    # Generate random values for that space and evaluate the function
    # On each point

    def __init__(self, distribution, samples):
        self.dist = distribution
        self.samples = samples

    def populate_zero_one(self):
        self.samples = np.random.rand(self.samples)

    def eval(self):
        return self.dist.eval(self.samples)