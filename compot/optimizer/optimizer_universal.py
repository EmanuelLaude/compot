import numpy as np
import compot.optimizer.optimizer_base as base


class Parameters(base.Parameters):
    def __init__(self, maxit = 500, tol = 1e-5, epsilon = 1e-12, gamma_init = 1., Gamma_init = 1.,
                 init_proc_gamma = 1, q = 1., alpha = 0.):
        super().__init__(maxit, tol, epsilon)

        self.gamma_init = gamma_init
        self.Gamma_init = Gamma_init
        self.init_proc_gamma = init_proc_gamma
        self.q = q
        self.alpha = alpha

class Status(base.Status):
    def __init__(self, nit=0, res=np.Inf, success=False, cumsum_backtracks = 0):
        super().__init__(nit, res, success)
        self.cumsum_backtracks = cumsum_backtracks
        self.gamma = 0.


class NesterovUniversalProximalGradientMethod(base.IterativeOptimizer):
    def __init__(self, params, problem, callback = None):
        super().__init__(params, problem, callback)
        self.status = Status()

    def setup(self):
        self.gamma = self.params.gamma_init
        self.res = np.Inf


    def pre_step(self, _):
        self.status.gamma = self.gamma
        return self.res

    def step(self, k):
        grad = self.problem.diffable.eval_gradient(self.x)
        value = self.problem.diffable.eval(self.x)

        while True:
            self.status.cumsum_backtracks = self.status.cumsum_backtracks + 1

            x = self.problem.proxable.eval_prox(self.x - self.gamma * grad, self.gamma)

            upper_bound = value + np.dot(grad, x - self.x) + 0.5 / self.gamma * np.dot(x - self.x, x - self.x) + self.params.epsilon / 2
            if self.problem.diffable.eval(x) <= upper_bound:
                break

            self.gamma = self.gamma * 0.5

        self.res = np.linalg.norm(x - self.x, 2) / self.gamma
        self.gamma = self.gamma * 2

        self.x[:] = x[:]


class NesterovUniversalFastProximalGradientMethod(base.IterativeOptimizer):
    def __init__(self, params, problem, callback = None):
        super().__init__(params, problem, callback)
        self.status = Status()


    def setup(self):
        self.y = np.zeros(self.problem.x_init.shape)
        self.y[:] = self.x[:]

        self.A = 0

        self.gamma = self.params.gamma_init

        self.v = np.zeros(self.problem.x_init.shape)
        self.v[:] = self.x[:]

        self.phi = np.zeros(self.problem.x_init.shape)
        self.theta = 0
        self.res = np.Inf


    def pre_step(self, _):
        self.status.gamma = self.gamma
        return self.res

    def step(self, k):
        while True:
            self.status.cumsum_backtracks = self.status.cumsum_backtracks + 1

            a = (self.gamma + np.sqrt(self.gamma ** 2 + 4 * self.gamma * self.A)) / 2
            A = self.A + a
            tau = a / A

            x = tau * self.v + (1 - tau) * self.y

            grad = self.problem.diffable.eval_gradient(x)
            value = self.problem.diffable.eval(x)
            x_hat = self.problem.proxable.eval_prox(self.v - a * grad, a)

            y = tau * x_hat + (1 - tau) * self.y

            upper_bound = (value + np.dot(grad, y - x)
                           + (0.5 / self.gamma) * np.dot(y - x, y - x)
                           + 0.5 * self.params.epsilon * tau)

            if self.problem.diffable.eval(y) <= upper_bound:
                break

            self.gamma = self.gamma * 0.5

        self.res = np.linalg.norm(self.problem.proxable.eval_prox(x - a * grad, a) - x) / a

        self.gamma = self.gamma * 2

        self.y[:] = y[:]
        self.A = A

        self.phi = self.phi + a * grad
        self.theta = self.theta + a

        self.v = self.problem.proxable.eval_prox(self.problem.x_init - self.phi, self.theta)

        self.x[:] = x[:]


class AdaptiveProximalGradientMethod(base.IterativeOptimizer):
    def __init__(self, params, problem, callback = None):
        super().__init__(params, problem, callback)
        self.status = Status()

    def setup(self):
        self.grad = self.problem.diffable.eval_gradient(self.x)
        self.gamma = self.params.gamma_init

        if self.params.init_proc_gamma == 0:
            self.gamma_plus = self.gamma
            self.x_plus = self.problem.proxable.eval_prox(self.x - self.gamma_plus * self.grad, self.gamma_plus)
            self.grad_plus = self.problem.diffable.eval_gradient(self.x_plus)
        else:
            x = self.problem.proxable.eval_prox(self.x - self.params.gamma_init * self.grad, self.params.gamma_init)
            grad = self.problem.diffable.eval_gradient(x)

            L = np.linalg.norm(self.grad - grad) / np.linalg.norm(self.x - x)

            if self.params.q - 2 * L <= 0:
                self.gamma = self.params.gamma_init
            else:
                self.gamma = self.params.gamma_init * (self.params.q * 2 * L) / (self.params.q - 2 * L)
            self.gamma_plus = self.params.gamma_init
            self.x_plus = x
            self.grad_plus = grad

    def pre_step(self, k):
        self.status.gamma = self.gamma_plus
        self.s = np.linalg.norm(self.x_plus - self.x)

        if k == 0:
            return np.Inf

        return self.s / self.gamma_plus

    def step(self, k):
        ell = np.dot(self.grad_plus - self.grad, self.x_plus - self.x) / self.s ** 2
        L = np.linalg.norm(self.grad_plus - self.grad) / self.s

        rho = self.gamma_plus / self.gamma
        alpha = np.sqrt(1 / self.params.q + rho)
        delta = self.gamma_plus ** 2 * L ** 2 - (2 - self.params.q) * self.gamma_plus * ell + 1 - self.params.q

        if delta <= 0.:
            beta = np.Inf
        else:
            beta = 1 / np.sqrt(2 * delta)

        self.gamma = self.gamma_plus

        self.gamma_plus = self.gamma * np.minimum(alpha, beta)

        self.x[:] = self.x_plus[:]
        self.x_plus[:] = self.problem.proxable.eval_prox(self.x - self.gamma_plus * self.grad_plus, self.gamma_plus)
        self.grad[:] = self.grad_plus[:]
        self.grad_plus = self.problem.diffable.eval_gradient(self.x_plus)


class AutoConditionedFastGradientMethod(base.IterativeOptimizer):
    def __init__(self, params, problem, callback = None):
        super().__init__(params, problem, callback)
        self.status = Status()


    def setup(self):
        self.grad = self.problem.diffable.eval_gradient(self.x)
        self.gamma = self.params.gamma_init

        self.beta = 1 - np.sqrt(3) / 2

        self.y = np.copy(self.x)
        self.tau_plus = 0
        self.tau = 0

        self.gamma_plus = self.gamma
        for i in range(10):  # Terminate the ill-defined line-search after 10 iterations
            self.x_plus = self.problem.proxable.eval_prox(self.x - self.gamma_plus * self.grad, self.gamma_plus)
            self.grad_plus = self.problem.diffable.eval_gradient(self.x_plus)
            self.L = (np.sqrt(
                np.power(np.linalg.norm(self.x_plus - self.x), 2) * np.power(np.linalg.norm(self.grad_plus - self.grad),
                                                                             2)
                + np.power(self.params.epsilon / 4, 2)
            ) - self.params.epsilon / 4) / np.power(np.linalg.norm(self.x_plus - self.x), 2)
            if self.beta / (4 * (1. - self.beta) * self.L) <= self.gamma_plus and self.gamma_plus <= 1 / (
                    3 * self.L):
                break
            self.gamma_plus = self.gamma_plus * 0.5


    def pre_step(self, k):
        self.status.gamma = self.gamma
        if k == 0:
            return np.Inf

        return np.linalg.norm(self.x - self.x_plus) / self.gamma_plus


    def step(self, k):
        if k == 0:
            self.gamma = self.gamma_plus
            self.gamma_plus = self.beta / (2 * self.L)
        else:
            # Compute Lipschitz estimate
            dist = np.abs(self.problem.diffable.eval(self.x) - \
                   self.problem.diffable.eval(self.x_plus) - np.inner(self.grad_plus, self.x - self.x_plus))
            # print("L", np.linalg.norm(self.x_plus), np.linalg.norm(self.x), np.linalg.norm(self.grad_plus), np.linalg.norm(self.grad))
            self.L = np.power(np.linalg.norm(self.grad_plus - self.grad), 2.) / (2 * dist + self.params.epsilon / self.tau_plus)

            self.gamma = self.gamma_plus
            self.gamma_plus = np.minimum(
                np.abs(self.beta * self.tau_plus / (4 * self.L)),
                ((self.tau + 1) / self.tau_plus) * self.gamma
            )
        self.z = self.problem.proxable.eval_prox(self.y - self.gamma_plus * self.grad_plus, self.gamma_plus)
        self.y = (1 - self.beta) * self.y + self.beta * self.z


        self.x = self.x_plus[:]
        if k == 0:
            self.tau = 0
            self.tau_plus = 2.
        else:
            self.tau = self.tau_plus
            self.tau_plus = self.tau + self.params.alpha / 2 + 2 * self.gamma * self.L * (1 - self.params.alpha) / (
                    self.beta * self.tau)

        self.x_plus = (self.z + self.tau_plus * self.x) / (1 + self.tau_plus)

        self.grad[:] = self.grad_plus[:]
        self.grad_plus = self.problem.diffable.eval_gradient(self.x_plus)
