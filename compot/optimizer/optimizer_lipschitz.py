from abc import abstractmethod
import numpy as np
import compot.optimizer.optimizer_base as base


class Parameters(base.Parameters):
    def __init__(self, maxit=500, tol=1e-5, epsilon=1e-12, gamma_init=1., mem=200, backtracking=True, alpha = 0.5):
        super().__init__(maxit, tol, epsilon)

        self.gamma_init = gamma_init
        self.mem = mem
        self.backtracking = backtracking
        self.alpha = alpha

class Status(base.Status):
    def __init__(self, nit=0, res=np.Inf, success=False, cumsum_backtracks = 0):
        super().__init__(nit, res, success)
        self.cumsum_backtracks = cumsum_backtracks
        self.gamma = 0.
        self.tau = 0.
        self.merit = np.Inf


class Panoc(base.IterativeOptimizer):
    def __init__(self, params, problem, callback = None):
        super().__init__(params, problem, callback)
        self.status = Status()

    def setup(self):
        if self.params.gamma_init <= 0:
            self.L = self.problem.diffable.get_Lip_gradient()
        else:
            self.L = 0.9 / self.params.gamma_init

        self.gamma = 0.9 / self.L
        self.sigma = 0.9 * self.gamma * (1 - self.gamma * self.L) / 2
        self.res = np.Inf

    def pre_step(self, _):
        self.grad = self.problem.diffable.eval_gradient(self.x)
        return self.res

    def merit_fun(self, x):
        grad = self.problem.diffable.eval_gradient(x)
        z = self.problem.proxable.eval_prox(x - self.gamma * grad, self.gamma)
        return (self.problem.diffable.eval(x) + np.dot(self.grad, z - x) + (0.5 / self.gamma) * np.dot(z - x, z - x)
                + self.problem.proxable.eval(z))

    @abstractmethod
    def get_update_direction(self, k):
        pass

    @abstractmethod
    def post_step(self, x):
        pass

    def step(self, k):
        fx = self.problem.diffable.eval(self.x)

        if self.params.backtracking:
            while True:
                z = self.problem.proxable.eval_prox(self.x - self.gamma * self.grad, self.gamma)
                self.R = (self.x - z) / self.gamma

                if self.problem.diffable.eval(z) <= fx + np.dot(self.grad, z - self.x) \
                    + (0.5 / self.gamma) * np.dot(z - self.x, z - self.x) + 1e-10:
                    break

                self.gamma = self.gamma / 2
                self.L = 2 * self.L
                self.sigma = self.sigma / 2
        else:
            z = self.problem.proxable.eval_prox(self.x - self.gamma * self.grad, self.gamma)
            self.R = (self.x - z) / self.gamma

        self.status.gamma = self.gamma
        self.res = np.linalg.norm(self.R, 2)

        d = self.get_update_direction(k)

        meritx = fx + np.dot(self.grad, z - self.x) \
               + (0.5 / self.gamma) * np.dot(z - self.x, z - self.x) + self.problem.proxable.eval(z)
        self.status.merit = meritx

        tau = 1
        while True:
            self.status.cumsum_backtracks = self.status.cumsum_backtracks + 1

            x = self.x - (1 - tau) * self.gamma * self.R + tau * d

            if tau < 1e-15:
                x = self.x - self.gamma * self.R
                break

            if self.merit_fun(x) <= meritx - self.sigma * np.dot(self.R, self.R) + 1e-12:
                break

            tau = tau * 0.5

        self.status.tau = tau
        self.post_step(x)

        self.x[:] = x[:]

class SemiSmoothNewtonPanoc(Panoc):
    def get_update_direction(self, k):
        A = self.problem.proxable.eval_Jacobian_prox(self.x - self.gamma * self.grad, self.gamma)
        B = np.identity(self.x.shape[0]) - self.gamma * self.problem.diffable.eval_Hessian(self.x)
        J = (np.identity(self.x.shape[0]) - np.dot(A, B)) / self.gamma

        d = np.linalg.solve(J, -self.R)

        return d

    def post_step(self, x):
        pass

class BFGSPanoc(Panoc):
    def __init__(self, params, problem, callback=None):
        super().__init__(params, problem, callback)
        self.H = np.identity(self.x.shape[0])


    def get_update_direction(self, k):
        d = -np.dot(self.H, self.R)
        return d

    def post_step(self, x):
        s = x - self.x

        grad = self.problem.diffable.eval_gradient(x)
        R = (x - self.problem.proxable.eval_prox(x - self.gamma * grad, self.gamma)) / self.gamma
        y = R - self.R

        rho = 1 / np.dot(y, s)

        V = np.identity(self.x.shape[0]) - rho * np.outer(y, s)
        self.H = np.dot(np.dot(V.T, self.H), V) + rho * np.outer(s, s)

class LBFGSPanoc(Panoc):
    def __init__(self, params, problem, callback=None):
        super().__init__(params, problem, callback)

        self.S = []
        self.Y = []

    def get_update_direction(self, k):
        q = self.R

        alpha = [0.] * len(self.S)
        rho = [0.] * len(self.S)
        for j in reversed(range(len(self.S))):
            rho[j] = 1 / np.dot(self.Y[j], self.S[j])
            alpha[j] = rho[j] * np.dot(self.S[j], q)

            q = q - alpha[j] * self.Y[j]

        if k > len(self.S):
            H = (np.dot(self.S[-1], self.Y[-1]) / np.dot(self.Y[-1], self.Y[-1])) * np.identity(self.x.shape[0])
        else:
            H = np.identity(self.x.shape[0])

        d = np.dot(H, q)
        for j in range(len(self.S)):
            beta = rho[j] * np.dot(self.Y[j], d)
            d = d + (alpha[j] - beta) * self.S[j]

        return -d

    def post_step(self, x):
        if len(self.S) >= self.params.mem:
            self.S.pop(0)
            self.Y.pop(0)

        self.S.append(x - self.x)
        grad = self.problem.diffable.eval_gradient(x)
        R = (x - self.problem.proxable.eval_prox(x - self.gamma * grad, self.gamma)) / self.gamma
        self.Y.append(R - self.R)


class FISTA(base.IterativeOptimizer):
    def __init__(self, params, problem, callback=None):
        super().__init__(params, problem, callback)
        self.status = Status()

    def setup(self):
        self.res = np.Inf
        self.y = np.copy(self.x)
        if self.params.gamma_init <= 0:
            self.gamma = 1 / self.problem.diffable.get_Lip_gradient()
        else:
            self.gamma = self.params.gamma_init
        self.t = 1.
        self.status.gamma = self.gamma

    def pre_step(self, _):
        return self.res

    def step(self, k):
        self.grad = self.problem.diffable.eval_gradient(self.y)

        if self.params.backtracking:
            fx = self.problem.diffable.eval(self.y)

            while True:
                self.status.cumsum_backtracks += 1
                x = self.problem.proxable.eval_prox(self.y - self.gamma * self.grad, self.gamma)
                if (self.problem.diffable.eval(x) <=
                        fx
                        + np.dot(self.grad, x - self.y)
                        + 0.5 / self.gamma * np.dot(x - self.y, x - self.y)
                        + self.params.epsilon):
                    break

                self.gamma = self.params.alpha * self.gamma

            self.status.gamma = self.gamma
        else:
            x = self.problem.proxable.eval_prox(self.y - self.gamma * self.grad, self.gamma)

        self.res = np.linalg.norm(x - self.y) / self.gamma
        t = (1 + np.sqrt(1 + 4 * self.t * self.t)) / 2

        self.y = x + ((self.t - 1) / t) * (x - self.x)
        self.t = t
        self.x[:] = x[:]


class ProximalGradientDescent(base.IterativeOptimizer):
    def __init__(self, params, problem, callback = None):
        super().__init__(params, problem, callback)
        self.status = Status()

    def setup(self):
        self.gamma_min = 1.99 / self.problem.diffable.get_Lip_gradient()

        if self.params.gamma_init > 0:
            self.gamma = self.params.gamma_init
        else:
            self.gamma = self.gamma_min

        self.status.gamma = self.gamma
        self.res = np.Inf

    def pre_step(self, k):
        return self.res

    def step(self, k):
        self.grad = self.problem.diffable.eval_gradient(self.x)

        if self.params.backtracking:
            fx = self.problem.diffable.eval(self.x)
            while True:
                self.gamma = np.maximum(self.gamma_min, self.gamma)
                x = self.problem.proxable.eval_prox(self.x - self.gamma * self.grad, self.gamma)
                if self.gamma == self.gamma_min:
                    break

                if (self.problem.diffable.eval(x) <= fx
                        + np.dot(self.grad, x - self.x)
                        + (0.5 / self.gamma) * np.dot(x - self.x, x - self.x)
                        + self.params.epsilon):
                    break

                self.gamma = self.gamma * self.params.alpha

            self.status.gamma = self.gamma
        else:
            x = self.problem.proxable.eval_prox(self.x - self.gamma * self.grad, self.gamma)

        self.res = np.linalg.norm(x - self.x) / self.gamma

        if self.params.backtracking:
            self.gamma = self.gamma / self.params.alpha
        self.x[:] = x[:]
