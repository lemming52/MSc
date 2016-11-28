import function
import rand_generator
import graphing
import montecarlo
import numpy as np
from pprint import pprint

class Emergent(function.Function):

    def __init__(self, rate, energy, coulomb, absorption):
        self.name = 'Emergent Energy'
        self.rate = rate
        self.energy = energy
        self.attenuation = coulomb + absorption

    def eval(self, variables):
        return self.rate * self.energy * \
            np.exp(-1 * variables * self.attenuation)

class Secondary(function.Function):

    def __init__(self, rate, energy, coulomb, absorption, electron):
        self.name = 'Emergent Energy'
        self.rate = rate
        self.energy = energy
        self.alpha = coulomb + absorption - electron
        self.coulomb = coulomb
        self.electron = electron

    def eval(self, variables):
        return self.rate * self.energy * self.coulomb / self.alpha * \
            (1 - np.exp(-1 * variables * self.alpha)) * np.exp(-1 * self.electron * variables)

def one_eight():

    emergent = Emergent(1, 1, 1, 1)
    secondary = Secondary(1, 1, 1, 1, 1)

    t_vals = np.linspace(0, 10, 1000)

    plot = graphing.LineGraph()
    plot.init_line()
    plot.add_title('FPP1.9: Plot of emergent and secondary energy against t')

    plot.add_line(t_vals, emergent.eval(t_vals), 'r--', label='Emergent Energy')
    plot.add_line(t_vals, secondary.eval(t_vals), 'b--', label='Secondary Energy')
    plot.add_line(t_vals,
                  emergent.eval(t_vals) + secondary.eval(t_vals),
                  'k',
                  label='Total')

    plot.label_axis('t', 'E')
    plot.show_legend()

    plot.show()


def two_nine():

    sample_counts = [5, 10, 33, 100, 333, 1000, 3333, 10000, 33333, 100000, 333333, 1000000, 3333333, 10000000]
    estimates = np.empty(len(sample_counts))

    for i in range(len(sample_counts)):
        estimates[i] = montecarlo.pi_estimate(sample_counts[i], 3)

    plot = graphing.LineGraph()
    plot.init_line()

    plot.add_line(sample_counts, estimates, 'b+', markersize=14, label='Estimated Value')
    plot.add_line(sample_counts, [np.pi]*len(sample_counts), 'r', label='Pi')
    plot.add_title('FPP Q2.9: Estimate of Pi against sample count N')
    plot.label_axis('N', 'Estimate')
    plot.log_axes(True, False)
    plot.set_axes(0, 33333333, 1.5, 4.5)
    plot.show_legend()

    plot.show()

def two_ten():

    sample_counts = [1, 5, 10, 33, 50, 100, 333, 500, 1000, 3333, 10000, 33333, 100000, 333333, 1000000]
    estimates = np.empty(len(sample_counts))

    for i in range(len(sample_counts)):
        estimates[i] = montecarlo.e_estimate(sample_counts[i])

    plot = graphing.LineGraph()
    plot.init_line()

    plot.add_line(sample_counts, estimates, 'b+', markersize=14, label='Estimated Value')
    plot.add_line(sample_counts, [np.e]*len(sample_counts), 'r', label='e')
    plot.add_title('FPP Q2.10: Estimate of e against sample count N')
    plot.label_axis('N', 'Estimate')
    plot.log_axes(True, False)
    plot.set_axes(0, 33333333, 1.5, 4.5)
    plot.show_legend()

    plot.show()
