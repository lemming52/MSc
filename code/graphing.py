import numpy as np
import matplotlib.pyplot as plt


class Graph_2d:
    """ Parent for graph types, structure will be to perform all necessary
        configuration using plot, then display using show.
    """

    def __init__(self):
        self.title = ''
        self.line = False
        self.legend = False
        self.x_axis = ''
        self.y_axis = ''

    def __str__(self):
        return self.title

    def plot(self):
        return None

    def show(self):
        return None

    def add_title(self, title):
        self.title = title

    def label_axis(self, x_axis, y_axis):
        self.x_axis = x_axis
        self.y_axis = y_axis

    def show_features(self):
        if self.line:
            self.plot_line()
        if self.legend:
            plt.legend()
        if self.title:
            plt.title(self.title)
        if self.x_axis:
            plt.xlabel(self.x_axis)
            plt.ylabel(self.y_axis)

    def save_graph(self, fig):
        file_name = input("Enter filename for graph, N to cancel: ")
        if file_name != 'N':
            fig.savefig(file_name + '.png')

    def add_line(self, function, line_values, line_color, **kwargs):
        # This is not well constructed, as it only allows for one line
        self.line = True
        self.line_func = function
        self.line_values = line_values
        self.line_color = line_color
        if 'label' in kwargs:
            self.line_label = kwargs['label']
            self.legend = True
        else:
            self.line_label = None

    def plot_line(self):
        plt.plot(self.line_values,
                 self.line_func.eval(self.line_values),
                 self.line_color,
                 label=self.line_label)


class Histogram(Graph_2d):

    def init_hist(self):
        self.norm = False
        self.range = None
        self.col = None
        self.stacked = False

    def plot(self, values, bins, **kwargs):
        self.init_hist()
        self.values = values
        self.bin_count = bins
        if 'function' in kwargs:
            self.function = kwargs['function']
        else:
            self.function = None

    def normalise(self, norm):
        self.norm = norm
        self.stacked = norm

    def color(self, col):
        self.col = col

    def show(self):
        fig = plt.figure()
        self.n, self.bins, patches = plt.hist(self.values,
                                              self.bin_count,
                                              range=self.range,
                                              normed=self.norm,
                                              color=self.col)

        # Sanity check
        self.area = sum(np.diff(self.bins) * self.n)
        self.title = self.title + ' |Bins: %s |Integrated Total: %s' % \
            (self.bin_count, self.area)

        self.eval_axes()

        # Set alpha=0.0 as plot attempts to add a kind of trend line
        # By setting alpha=0.0, this trendline is transparent, and we can
        # add our own lines later
        self.graph = plt.plot(self.bins, alpha=0.0)

        plt.axis(self.axes_values)

        self.show_features()
        self.save_graph(fig)
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
