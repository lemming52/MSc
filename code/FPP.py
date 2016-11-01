import function
import rand_generator
import graphing
import numpy as np
import pagerank
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

