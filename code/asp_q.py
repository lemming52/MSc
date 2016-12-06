import function
import rand_generator
import graphing
import numpy as np
import matplotlib.pyplot as plt
import random
import integration
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



def three_9():

    STATE_COUNT = 100

    states = np.zeros(STATE_COUNT)

    initial_position = 10
    states[initial_position] = 1

    forward_rate = 0.05
    backward_rate = 0.05
    rate_param = forward_rate + backward_rate

    simulations = [1, 3, 10, 33, 100, 333, 1000, 3333, 10000, 33333, 100000]
    absorption_times = np.zeros(len(simulations))
    probabilities = np.zeros(len(simulations))

    for i in range(len(simulations)):
        sim_count = simulations[i]
        times = np.zeros(sim_count)
        endpoint = 0

        for iteration in range(sim_count):
            absorbed = False
            time = 0
            current_position = initial_position

            while (absorbed == False):

                time += -1/rate_param * np.log(random.random())

                fraction = forward_rate/rate_param
                if(random.random() >= fraction):
                    current_position += 1
                else:
                    current_position -= 1

                if (current_position == 0 or current_position == STATE_COUNT-1):
                    absorbed = True
                    times[iteration] = time
                    if(current_position == STATE_COUNT-1):
                        endpoint += 1

        absorption_times[i] = np.average(times)
        probabilities[i] = endpoint/sim_count

    plot = graphing.LineGraph()
    plot.init_line()

    plot.add_line(simulations, absorption_times, 'b+', markersize=14, label='Unconditional Absorption Time')
    plot.add_title('ASP Q3.9: Average unconditional absorptiontime against simulation runs')
    plot.label_axis('N', 'Average Time')
    plot.log_axes(True, False)
    plot.set_axes(0, 3333333, 0, 10000)
    plot.show_legend()

    plot.show()

    plot = graphing.LineGraph()
    plot.init_line()

    plot.add_line(simulations, probabilities, 'b+', markersize=14, label='Probability of absorption at n=N')
    plot.add_title('ASP Q3.9: Proabability of absorption at n=N for differnt simulation counts')
    plot.label_axis('N', 'Probability')
    plot.log_axes(True, False)
    plot.set_axes(0, 3333333, 0, 0.5)
    plot.show_legend()

    plot.show()





    """
    STATE_COUNT = 100
    first_state = states.RandomWalk1D()
    first_state.linkPrior(None)
    first_state.setRates(None, None)
    states = []
    states.append(first_state)

    for i in range(1, STATE_COUNT-1):

        """

class FourTwoX(function.Function):

    def __init__(self, b):
        self.name = "x differntial equation for question 4.2"
        self.b = b

    def eval(self, variables):
        return np.power(variables[0], 2) * variables[1] - (self.b + 1)*variables[0] + 1

class FourTwoY(function.Function):

    def __init__(self, b):
        self.name = "y differntial equation for question 4.2"
        self.b = b

    def eval(self, variables):
        return self.b*variables[0] - np.power(variables[0], 2) * variables[1]

def four_two():

    timestep = 0.001
    count = 100000

    x = FourTwoX(1.8)
    y = FourTwoY(1.8)

    results = integration.xy_euler_integrate(x, y, [1, 1], timestep, count)
    t = np.linspace(0, timestep*count, count)

    plot = graphing.LineGraph()
    plot.init_line();

    plot.add_line(t, results[0], 'b--', markersize=12, label='x')
    plot.add_line(t, results[1], 'r--', markersize=12, label='y')
    plot.add_title('ASP Q4.2: Euler Integration estimate for x(t)and y(t)')
    plot.label_axis('t', 'value')
    plot.log_axes(False, False)
    plot.show_legend()

    plot.show()

    xyplot = graphing.LineGraph()
    xyplot.init_line()

    xyplot.add_line(results[0], results[1], 'b--', markersize=12)
    xyplot.add_title('ASP Q4.2: Euler Integration estimate for x and y)')
    xyplot.label_axis('x(t)', 'y(t)')
    xyplot.log_axes(False, False)

    xyplot.show()

def four_two_d():

    omega = 5000
    b = 1.8
    timestep = 0.01
    count = 1000000*2

    processes = {
        'A' : [omega, 1, 0, 0, 0],
        'B' : [1, -1, 0, 1, 0],
        'C' : [b, -1, 1, 1, 0],
        'D' : [1/(omega*omega), 1, -1, 2, 1],
    }

    initial = [1000, 1000]

    t = np.linspace(0, timestep*count, count)
    results = np.zeros((2, count))
    results[0][0] = initial[0]
    results[1][0] = initial[1]

    times = np.zeros(count)
    time = 0

    for i in range(1, count):

        combined_rates = 0

        for process in processes:
            params = processes[process]
            rate_param = params[0] * np.power(results[0][i-1], params[3]) * np.power(results[1][i-1], params[4])
            combined_rates += rate_param

        rand = random.random()
        time += -1/combined_rates * np.log(random.random())
        times[i] = time

        cumul_fraction = 0

        for process in processes:
            params = processes[process]
            rate_param = params[0] * np.power(results[0][i-1], params[3]) * np.power(results[1][i-1], params[4])
            fraction = rate_param / combined_rates
            cumul_fraction += fraction
            if rand < cumul_fraction:
                results[0][i] = results[0][i-1] + params[1]
                results[1][i] = results[1][i-1] + params[2]
                break

    plot = graphing.LineGraph()
    plot.init_line()

    plot.add_line(times, results[0]/omega, 'b--', markersize=12, label='x')
    plot.add_line(times, results[1]/omega, 'r--', markersize=12, label='y')
    plot.add_title('ASP Q4.2: Gillespie simulation of Brusselator')
    plot.label_axis('t', 'n')
    plot.log_axes(False, False)
    plot.show_legend()

    plot.show()

    xyplot = graphing.LineGraph()
    xyplot.init_line()

    xyplot.add_line(results[0]/omega, results[1]/omega, 'b--', markersize=12)
    xyplot.add_title('ASP Q4.2: Gillespie simulation estimate for x and y)')
    xyplot.label_axis('x(t)', 'y(t)')
    xyplot.log_axes(False, False)

    xyplot.show()

    ratioplot = graphing.LineGraph()
    ratioplot.init_line()

    ratioplot.add_line(times, results[1]/results[0], 'b--', markersize=12, label='ratio')
    ratioplot.add_line(times, [b]*len(times), 'r--', markersize=12, label='b')
    ratioplot.add_title('ASP Q4.2: Gillespie simulation of Brusselator, ratio y/x')
    ratioplot.label_axis('t', 'ratio y/x')
    ratioplot.log_axes(False, False)

    ratioplot.show()






class FourThreeN(function.Function):

    def __init__(self, N):
        self.name = 'Differential equation for 4.3'
        self.N = N
        self.N2 = N * N
        self.N3 = self.N2 * N

    def eval(self, variables):
        return variables / self.N - 3 * np.power(variables, 2) / self.N2 + \
                2 * np.power(variables, 3) / self.N3

def four_three():

    initial_conditions = np.linspace(0, 1, 21)
    timestep = 0.00025
    count = 100000
    N = 1

    n = FourThreeN(N)

    results = np.zeros((len(initial_conditions), count))
    for i in range(len(initial_conditions)):

        results[i] = integration.euler_integrate(n, timestep, count,
                                                 (initial_conditions[i])*N)

    t = np.linspace(0, timestep*count, count)

    plot = graphing.LineGraph()
    plot.init_line();

    for i in range(len(initial_conditions)):
        plot.add_line(t, results[i], 'b--', markersize=12, label='%s' % initial_conditions[i])
    plot.add_title('ASP Q4.3: Euler Integration estimate')
    plot.label_axis('t', 'n')
    plot.log_axes(False, False)

    plot.show()

def five_six():

    time_step = 0.001
    time_step_root = time_step**(1/2)
    count = 1000000
    a = 2
    mean = 0
    D = 0.2
    variance = 2*D
    sd = variance**(1/2)

    x_init = 0


    results = np.zeros(count)
    squares = np.zeros(count)
    averages = np.zeros(count)
    results[0] = x_init
    squares[0] = x_init**2
    averages[0] = squares[0]

    total = squares[0]

    f = function.ConstantMultiply(a)

    for i in range(count-1):
        noise = np.random.normal(mean, sd)
        results[i + 1] = results[i] + time_step * f.eval(results[i]) + time_step_root * noise
        squares[i+1] = results[i+1]**2
        total = total + squares[i+1]
        averages[i+1] = (total/(i+1))

    t = np.linspace(0, time_step*count, count)

    plot = graphing.LineGraph()
    plot.init_line();

    plot.add_line(t, results, 'b--', markersize=12)
    plot.add_title('ASP Q5.6: OU Process')
    plot.label_axis('t', 'x(t)')
    plot.log_axes(False, False)

    plot.show()

    aplot = graphing.LineGraph()
    aplot.init_line();

    aplot.add_line(t, averages, 'b--', markersize=12)
    aplot.add_line(t, [D/a]*count, 'r--', label='D/a', markersize=12)
    aplot.add_title('ASP Q5.6: OU Process')
    aplot.label_axis('t', '<x(t)^2>')
    aplot.log_axes(False, False)

    aplot.show()

