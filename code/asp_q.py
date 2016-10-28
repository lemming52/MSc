import function
import rand_generator
import graphing
import numpy as np
import pagerank
from pprint import pprint


def one_nine():
    rate_param = 2
    exp_dist = function.ExponentialDist(rate_param)
    log_rate = function.LogRate(rate_param)
    gen = rand_generator.Generator(log_rate, [100000])
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
    hist.add_title('ASP Q1.10 (Batch Count: %s)' % batch_count)
    hist.label_axis('Average value (batch size: %s)' % batch_size,
                    'Relative Frequency')
    hist.show()

def two_eight():

    graph_size = 1000000

    graph = np.zeros((graph_size,), dtype=np.int)

    for i in range(1, graph_size):
        index = np.random.random_integers(0, i-1)
        graph[i] = 1
        graph[index] += 1

    max_val = np.max(graph)

    hist = graphing.Histogram()
    hist.plot(graph, max_val-1)
    hist.normalise(True)
    rate_param = 1/2
    exp_dist = function.ExponentialDist(rate_param)
    hist.add_line(exp_dist, np.linspace(1, max_val, max_val), 'ro--', label='p(k)')
    hist.show()

def two_nine():

    sample_counts = [5, 10, 33, 100, 333, 1000, 3333, 10000, 33333, 100000, 333333, 1000000, 3333333, 10000000]
    estimates = np.empty(len(sample_counts))
    magnitude = function.Magnitude()

    for i in range(len(sample_counts)):

        count = 0

        gen = rand_generator.Generator(magnitude, [2, sample_counts[i]])
        gen.populate_zero_one()
        results = gen.eval()
        for value in results:
            if value <= 1:
                count += 1

        estimates[i] = count/sample_counts[i]*4

    plot = graphing.LineGraph()
    plot.init_line()

    plot.add_line(sample_counts, estimates, 'b+', markersize=14, label='Estimated Value')
    plot.add_line(sample_counts, [np.pi]*len(sample_counts), 'r', label='Pi')
    plot.add_title('ASP Q2.9: Estimate of Pi against sample count N')
    plot.label_axis('N', 'Estimate')
    plot.log_axes(True, False)
    plot.set_axes(0, 33333333, 1.5, 4.5)
    plot.show_legend()

    plot.show()

def two_ten():
    graph = [[1, 2], [1, 3], [1, 4], [1, 5],
        [2, 4],
        [3, 4],
        [4, 3], [4, 5],
        [5, 2], [5, 3], [5, 4]]

    print(pagerank.rank(graph, 0.85, 0.000000001))
    print(pagerank.rank(graph, 1, 0.0000000001))






