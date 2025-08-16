from abc import ABC, abstractmethod

import numpy as np
import compot.calculus.function as fun


class OptimizationProblem(ABC):
    def __init__(self, x_init):
        self.x_init = x_init

    @abstractmethod
    def eval_objective(self, x):
        pass


class CompositeOptimizationProblem(OptimizationProblem):
    def __init__(self, x_init, diffable, proxable):
        super().__init__(x_init)

        assert(isinstance(diffable, fun.Diffable))
        assert(isinstance(proxable, fun.Proxable))

        self.diffable = diffable
        self.proxable = proxable

    def eval_objective(self, x):
        return self.diffable.eval(x) + self.proxable.eval(x)

class DiffableOptimizationProblem(OptimizationProblem):
    def __init__(self, x_init, diffable):
        super().__init__(x_init)

        assert (isinstance(diffable, fun.Diffable))

        self.diffable = diffable


    def eval_objective(self, x):
        return self.diffable.eval(x)


class SaddlePointProblem(ABC):
    def __init__(self, x_init, y_init, A, b):
        self.x_init = x_init
        self.y_init = y_init
        self.A = A
        self.b = b

    @abstractmethod
    def eval_primal_dual(self, x, y):
        pass

# min_x g(Ax - b) + f(x)
# max_y -f^*(-A^T y) - g^*(y) - <b, y>
# min_x sup_y <Ax - b, y> + f(x) - g^*(y)
class DualizableSaddlePointProblem(SaddlePointProblem):
    def __init__(self, x_init, y_init, A, b, f, g):
        super().__init__(x_init, y_init, A, b)

        assert(isinstance(f, fun.Dualizable))
        assert(isinstance(g, fun.Dualizable))

        self.g = g
        self.f = f

    def eval_primal(self, x):
        return self.g.eval(self.A.apply(x) - self.b) + self.f.eval(x)

    def eval_dual(self, y):
        return -self.g.get_conjugate().eval(y) - self.f.get_conjugate().eval(-self.A.apply_transpose(y)) - np.dot(y, self.b)

    def eval_primal_dual(self, x, y):
        return np.dot(self.A.appy(x) - self.b, y) + self.f.eval(x) - self.g.get_conjugate().eval(y)


class Parameters:
    def __init__(self, maxit = 500, tol=1e-5, epsilon = 1e-12):
        self.maxit = maxit
        self.tol = tol
        self.epsilon = epsilon

class Status:
    def __init__(self, nit = 0, res = np.inf, success = False):
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
        while True:
            self.update_status(k)

            if not self.callback is None and self.callback(self.x, self.status):
                return self.status

            if self.status.res <= self.params.tol:
                self.status.success = True
                return self.status

            if k == self.params.maxit:
                return self.status

            self.step(k)

            k += 1

class PrimalDualIterativeOptimizer(Optimizer):
    def __init__(self, params, problem, callback = None):
        super().__init__(params, problem, callback)
        self.params = params
        self.problem = problem
        self.callback = callback

        self.x = np.zeros(problem.x_init.shape)
        self.x[:] = problem.x_init[:]

        self.s = np.zeros(problem.x_init.shape)
        self.s[:] = problem.x_init[:]

        self.y = np.zeros(problem.y_init.shape)
        self.y[:] = problem.y_init[:]

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
        while True:
            self.update_status(k)

            if not self.callback is None and self.callback((self.x, self.s, self.y), self.status):
                return self.status

            if self.status.res <= self.params.tol:
                self.status.success = True
                return self.status

            if k == self.params.maxit:
                return self.status

            self.step(k)

            k += 1
