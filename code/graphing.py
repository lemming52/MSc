import numpy as np
import matplotlib.pyplot as plt


class Graph_2d:
    """ Parent for graph types, structure will be to perform all necessary
        configuration using plot, then display using show.
    """

    def __init__(self):
        self.title = 'abstract graph type'

    def __str__(self):
        return self.title

    def plot(self):
        return None

    def show(self):
        return None

    def title(self, title):
        self.title = title

    def label_axis(self, x_axis, y_axis):
        self.x_axis = x_axis
        self.y_axis = y_axis


class Histogram(Graph_2d):

    def __init__(self):
        self.norm = False
        self.range = None
        self.col = None
        self.stacked = False
        return

    def plot(self, values, bins):
        self.values = values
        self.bins = bins

    def normalise(self, norm):
        self.norm = norm
        self.stacked = norm

    def color(self, col):
        self.col = col

    def show(self):
        self.n, self.bins, patches = plt.hist(self.values,
                                              self.bins,
                                              range=self.range,
                                              normed=self.norm,
                                              stacked=self.stacked,
                                              color=self.col)
        self.eval_axes()
        graph = plt.plot(self.bins)
        plt.axis(self.axes_values)
        plt.show()

    def eval_axes(self):
        min_x = np.min(self.bins)
        max_x = np.max(self.bins)
        padding = (max_x - min_x) * 0.1
        self.axes_values = [
            min_x - padding,
            max_x + padding,
            0,
            np.max(self.n) * 1.125
        ]


