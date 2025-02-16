from abc import abstractmethod
import numpy as np
import compot.optimizer.base as base

class Parameters(base.Parameters):
    def __init__(self, maxit = 500, tol = 1e-5, epsilon = 1e-12,
                 gamma_init=1., mem = 200, Wolfe = True, sigma = 1e-4, eta = 0.9):
        super().__init__(maxit, tol, epsilon)

        self.gamma_init = gamma_init
        self.mem = mem
        self.Wolfe = Wolfe
        self.sigma = sigma
        self.eta = eta

class Status(base.Status):
    def __init__(self, nit=0, res=np.inf, success=False, cumsum_backtracks = 0):
        super().__init__(nit, res, success)
        self.cumsum_backtracks = cumsum_backtracks
        self.gamma = 0.

class DescentMethodBaseClass(base.IterativeOptimizer):
    def __init__(self, params, problem, callback = None):
        super().__init__(params, problem, callback)
        self.status = Status()


    def setup(self):
        pass

    def pre_step(self, _):
        self.grad = self.problem.diffable.eval_gradient(self.x)
        return np.linalg.norm(self.grad)

    @abstractmethod
    def get_descent_direct(self, k):
        pass

    @abstractmethod
    def post_step(self, x):
        pass

    def step(self, k):
        sigma = self.params.sigma
        eta = self.params.eta


        d = self.get_descent_direct(k)
        fx = self.problem.diffable.eval(self.x)

        direc_deriv = np.dot(self.grad, d)

        gamma = self.params.gamma_init
        gamma_low = 0
        gamma_high = np.inf
        while True:
            self.status.cumsum_backtracks += 1

            fx_plus = self.problem.diffable.eval(self.x + gamma * d)
            grad_plus = self.problem.diffable.eval_gradient(self.x + gamma * d)

            if fx_plus > fx + sigma * gamma * direc_deriv + self.params.epsilon:
                gamma_high = gamma
                gamma = 0.5 * (gamma_low + gamma_high)
            elif self.params.Wolfe and np.dot(grad_plus, d) < eta * np.dot(self.grad, d) - self.params.epsilon:
                gamma_low = gamma
                if gamma_high == np.inf:
                    gamma = 2 * gamma_low
                else:
                    gamma = 0.5 * (gamma_low + gamma_high)
            else:
                break
        self.status.gamma = gamma
        x = self.x + gamma * d

        self.post_step(x)
        self.x[:] = x[:]

class SteepestDescent(DescentMethodBaseClass):
    def get_descent_direct(self, k):
        d = -self.grad
        return d

    def post_step(self, x):
        pass

class NewtonsMethod(DescentMethodBaseClass):
    def get_descent_direct(self, k):
        H = self.problem.diffable.eval_Hessian(self.x)
        return np.linalg.solve(H, -self.grad)#-np.dot(np.linalg.inv(Hess), self.grad)

    def post_step(self, x):
        pass


class BFGS(DescentMethodBaseClass):
    def __init__(self, params, problem, callback = None):
        super().__init__(params, problem, callback)

        self.H = np.identity(self.x.shape[0])

    def get_descent_direct(self, k):
        d = -np.dot(self.H, self.grad)
        return d

    def post_step(self, x):
        s = x - self.x
        y = self.problem.diffable.eval_gradient(x) - self.grad

        rho = 1 / np.dot(y, s)

        V = np.identity(self.x.shape[0]) - rho * np.outer(y, s)
        self.H = np.dot(np.dot(V.T, self.H), V) + rho * np.outer(s, s)


class LBFGS(DescentMethodBaseClass):
    def __init__(self, params, problem, callback = None):
        super().__init__(params, problem, callback)

        self.S = []
        self.Y = []

    def get_descent_direct(self, k):
        q = self.grad

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
        self.Y.append(self.problem.diffable.eval_gradient(x) - self.grad)
