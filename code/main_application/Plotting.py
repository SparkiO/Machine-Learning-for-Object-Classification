import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt4agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt4agg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure

import numpy as np

PATH = "graph/"
LABELS = ["berry", "bird", "dog", "flower", "other"]

class Plot:
    def __init__(self, data, name_sample, name_plot):
        self.data = data
        self.name_sample = name_sample
        self.name_plot = name_plot

    def plot(self):
        fig,ax = plt.subplots()
        fig.set_size_inches(3.75, 3)
        ax.set_ylim([0,100])
        y_pos = np.arange(len(LABELS))
        data = self.data * 100
        classes = np.around(data, 2)
        b = []
        for j in range(len(LABELS)):
            b.append(classes[j])

        ax.bar(y_pos, b, color="green", tick_label=LABELS)
        path = "{}{}-{}".format(PATH, self.name_plot, self.name_sample)
        plt.savefig(path)
        plt.close()
        return path
