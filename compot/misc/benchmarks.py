from abc import ABC, abstractmethod

import numpy as np

import matplotlib.pyplot as plt

from pathlib import Path

from math import floor, log10

def fexp10(f):
    return int(floor(log10(abs(f)))) if f != 0 else 0

def fman10(f):
    return f / 10 ** fexp10(f)



class Benchmark(ABC):
    def __init__(self, name, config, optimizer_configs, num_runs):
        self.name = name
        self.config = config
        self.optimizer_configs = optimizer_configs
        self.num_runs = num_runs

    @abstractmethod
    def setup(self):
        pass

    @abstractmethod
    def get_criterion_keys(self):
        pass

    @abstractmethod
    def eval_criteria(self, problem, variable, status):
        pass

    @abstractmethod
    def get_refvals(self):
        pass

    @abstractmethod
    def get_axis_labels(self):
        pass

    def get_filename(self):
        return (self.config["filename_prefix"] + self.name + "_config_"
            + self.config["name"]
            + "_num_runs_" + str(self.num_runs)
            + "_seed_" + str(self.config["seed"]))

    @abstractmethod
    def setup_problem(self):
        pass

    @abstractmethod
    def setup_optimizer(self, optimizer_config, problem, callback):
        pass

    def run(self, overwrite_file=False):
        np.random.seed(self.config["seed"])

        filename = self.get_filename()
        suffix = ".npz"

        if overwrite_file or not Path(filename + suffix).is_file():
            self.problem_data = self.setup()

            self.refvals = self.get_refvals()

            self.criteria = dict()
            for key in self.get_criterion_keys():
                self.criteria[key] = dict()

        else:
            storage = np.load(filename + '.npz', allow_pickle=True)

            self.problem_data = storage["problem_data"].item()
            self.refvals = storage["refvals"].item()
            self.criteria = storage["criteria"].item()


        optimizer_added = False

        for optimizer_config in self.optimizer_configs:
            if optimizer_config["name"] in self.criteria[self.get_criterion_keys()[0]]:
                continue

            optimizer_added = True
            np.random.seed(self.config["seed"])

            for key in self.get_criterion_keys():
                self.criteria[key][optimizer_config["name"]] = [{ "xvals": [], "yvals": [] } for _ in range(self.num_runs)]

            for run in range(self.num_runs):
                problem = self.setup_problem()

                def callback(variable, status):
                    criteria, stop_algorithm = self.eval_criteria(problem, variable, status)
                    for key in self.get_criterion_keys():
                        self.criteria[key][optimizer_config["name"]][run]["xvals"].append(criteria[key]["x"])
                        self.criteria[key][optimizer_config["name"]][run]["yvals"].append(criteria[key]["y"])

                    msg = "nit: " + str(status.nit)
                    for key in self.get_criterion_keys():
                        msg = msg + "    " + key + ": " + str(criteria[key]["y"])

                    print(msg)

                    return stop_algorithm

                optimizer = self.setup_optimizer(optimizer_config,
                                                      problem,
                                                      callback)

                print(optimizer_config["name"])

                optimizer.run()

        if optimizer_added:
            np.savez(filename,
                     problem_data=self.problem_data,
                     refvals=self.refvals,
                     criteria=self.criteria,
                     )


    def linspace_values(self, x, y, interval):
        values = np.arange(0, x[-1], interval) * 0.

        j = 0
        for i in range(0, x[-1], interval):
            while True:
                if x[j] > i:
                    break
                j = j + 1

            # linearly interpolate the values at j-1 and j to obtain the value at i
            values[int(i / interval)] = (
                    y[j - 1]
                    + (i - x[j - 1])
                    * (y[j] - y[j - 1]) / (x[j] - x[j - 1])
            )
        return values

    def plot_mean_stdev(self, xvals, yvals, label, marker, color, refval=0., plotstdev=True, markevery=20,
                        plotevery=250):

        # compute new array with linspaced xvals with shortest length
        xvals_linspace = np.arange(0, xvals[0][-1], plotevery)
        for i in range(1, len(xvals)):
            arange = np.arange(0, xvals[i][-1], plotevery)
            if len(xvals_linspace) > len(arange):
                xvals_linspace = arange

        yvals_mean = np.zeros(len(xvals_linspace))

        for i in range(len(xvals)):
            y_values_interp = self.linspace_values(xvals[i],
                                                   yvals[i], plotevery)
            yvals_mean += y_values_interp[0:len(xvals_linspace)]

        yvals_mean = yvals_mean / len(xvals)

        plt.semilogy(xvals_linspace, yvals_mean - refval,
                     label=label,
                     marker=marker,
                     markevery=markevery,
                     color=color)

        if len(xvals) > 1 and plotstdev:
            yvals_stdev = np.zeros(len(xvals_linspace))

            for i in range(len(xvals)):
                y_values_interp = self.linspace_values(xvals[i],
                                                       yvals[i], plotevery)

                yvals_stdev += (yvals_mean - y_values_interp[0:len(xvals_linspace)]) ** 2

            yvals_stdev = np.sqrt(yvals_stdev / len(xvals))

            plt.fill_between(xvals_linspace,
                             yvals_mean - refval - yvals_stdev,
                             yvals_mean - refval + yvals_stdev,
                             alpha=0.5, facecolor=color,
                             edgecolor='white')


    def plot_criterion(self, key):
        xvals = dict()
        yvals = dict()
        for optimizer_config in self.optimizer_configs:
            xvals[optimizer_config["name"]] = [[] for _ in range(self.num_runs)]
            yvals[optimizer_config["name"]] = [[] for _ in range(self.num_runs)]
            for run in range(self.num_runs):
                xvals[optimizer_config["name"]][run] = np.array(self.criteria[key][optimizer_config["name"]][run]["xvals"])
                yvals[optimizer_config["name"]][run] = np.array(self.criteria[key][optimizer_config["name"]][run]["yvals"])



        self.plot(xvals, yvals, self.refvals[key], self.get_axis_labels()[key]["x"], self.get_axis_labels()[key]["y"])

    def plot(self, xvals, yvals, refval, xlabel, ylabel):
        for optimizer_config in self.optimizer_configs:
            if self.num_runs == 1:
                plt.semilogy(xvals[optimizer_config["name"]][0], np.array(yvals[optimizer_config["name"]][0]) - refval,
                             label=optimizer_config["label"],
                             marker=optimizer_config["marker"],
                             markevery=optimizer_config["markevery"],
                             color=optimizer_config["color"],
                             linestyle=optimizer_config["linestyle"],
                             )
            else:
                self.plot_mean_stdev(xvals[optimizer_config["name"]], yvals[optimizer_config["name"]],
                                     label = optimizer_config["label"],
                                     marker = optimizer_config["marker"],
                                     color = optimizer_config["color"],
                                     refval = refval,
                                     plotstdev= True,
                                     markevery=optimizer_config["markevery"],
                                     plotevery = optimizer_config["plotevery"])

            plt.xlabel(xlabel)
            plt.ylabel(ylabel)


        plt.tight_layout()
        plt.legend()

