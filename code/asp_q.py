import function
import rand_generator
import argparse
import matplotlib.pyplot as plt
from pprint import pprint

def q_one_nine():
    dist = function.ExponentialDist(2)
    gen = rand_generator.Generator(dist, 100000)
    gen.populate_zero_one()
    results = gen.eval()
    n, bins, patches = plt.hist(results, 50)
    plot = plt.plot(bins)
    plt.show()
    pprint(results)
