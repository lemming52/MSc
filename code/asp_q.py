import function
import rand_generator
import graphing
import numpy as np
from pprint import pprint


def one_nine():
    rate_param = 2
    exp_dist = function.ExponentialDist(rate_param)
    log_rate = function.LogRate(rate_param)
    gen = rand_generator.Generator(log_rate, 100000)
    gen.populate_zero_one()
    results = gen.eval()
    hist = graphing.Histogram()
    hist.plot(results, 5)
    hist.normalise(True)
    hist.color('b')
    hist.add_title('ASP Q1.9')
    hist.label_axis('Y = -(1/lambda) * ln(x)', 'Relative Frequency')
    hist.add_line(exp_dist, np.linspace(0, 10, 1000), 'r--', label='p(x)')
    hist.show()


def one_ten():
    batch_count = 10000
    batch_size = int(input('Enter batch size: '))

    samples = np.random.rand(batch_size, batch_count)
    average = np.average(samples, 0)
    hist = graphing.Histogram()
    hist.plot(average, 60)
    hist.normalise(True)
    hist.color('b')
    hist.add_title('ASP Q1.10 (Batch Count: %s' % batch_count)
    hist.label_axis('Average value (batch size: %s' % batch_size,
                    'Relative Frequency')
    hist.show()

