import function
import rand_generator
import graphing
from pprint import pprint


def one_nine():
    dist = function.ExponentialDist(2)
    gen = rand_generator.Generator(dist, 100000)
    gen.populate_zero_one()
    results = gen.eval()
    hist = graphing.Histogram()
    hist.plot(results, 30)
    hist.normalise(True)
    hist.color('r')
    hist.title('ASP Q1.9')
    hist.show()
