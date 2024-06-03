from abc import ABC, abstractmethod

import numpy as np
import scipy


class OptimizationProblem(ABC):
    def __init__(self, x_init):
        self.x_init = x_init

    @abstractmethod
    def eval_objective(self, x):
        pass


class CompositeOptimizationProblem(OptimizationProblem):
    def __init__(self, x_init, diffable, proxable):
        super().__init__(x_init)
        self.diffable = diffable
        self.proxable = proxable

    def eval_objective(self, x):
        return self.diffable.eval(x) + self.proxable.eval(x)

class DiffableOptimizationProblem(OptimizationProblem):
    def __init__(self, x_init, diffable):
        super().__init__(x_init)
        self.diffable = diffable

    def eval_objective(self, x):
        return self.diffable.eval(x)

class Parameters:
    def __init__(self, maxit = 500, tol = 1e-5, epsilon = 1e-12):
        self.maxit = maxit
        self.tol = tol
        self.epsilon = epsilon

class Status:
    def __init__(self, nit = 0, res = np.Inf, success = False):
        self.nit = nit
        self.res = res
        self.success = success


class Optimizer(ABC):
    def __init__(self, params, problem, callback=None):
        self.params = params
        self.problem = problem
        self.callback = callback
        self.status = Status()

    @abstractmethod
    def run(self):
        pass

class ScipyBFGSWrapper(Optimizer):
    def __init__(self, params, problem, callback = None):
        super().__init__(params, problem, callback)
        self.x = np.zeros(problem.x_init.shape)
        self.x[:] = problem.x_init[:]

    def run(self):
        options = { "maxiter": self.params.maxit, "gtol": self.params.tol }

        callback = lambda x: self.callback(0, 0, 0, x, np.linalg.norm(self.problem.diffable.eval_gradient(x))) if not self.callback is None else None

        result = scipy.optimize.minimize(lambda x: self.problem.diffable.eval(x),
                                      self.problem.x_init,
                                      jac=lambda x: self.problem.diffable.eval_gradient(x),
                                      method="BFGS",
                                      options=options,
                                      callback=callback)
        self.x[:] = result.x[:]
        self.res = np.linalg.norm(self.problem.diffable.eval_gradient(self.x))
        self.k = result.nit
        self.success = result.success

        return self.success

class IterativeOptimizer(Optimizer):
    def __init__(self, params, problem, callback = None):
        super().__init__(params, problem, callback)
        self.params = params
        self.problem = problem
        self.callback = callback

        self.x = np.zeros(problem.x_init.shape)
        self.x[:] = problem.x_init[:]

    @abstractmethod
    def setup(self):
        pass

    @abstractmethod
    def step(self, k):
        pass

    @abstractmethod
    def pre_step(self, k):
        pass

    def update_status(self, k):
        self.status.nit = k
        self.status.res = self.pre_step(k)
        self.status.success = self.status.res <= self.params.tol

    def run(self):
        self.setup()

        k = 0
        for k in range(self.params.maxit):
            self.update_status(k)

            if not self.callback is None and self.callback(self.x, self.status):
                return self.status

            if self.status.res <= self.params.tol:
                self.status.success = True
                return self.status

            self.step(k)


        self.update_status(k)

        return self.status
