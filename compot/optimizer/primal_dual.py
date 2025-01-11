import numpy as np

import compot.calculus.function as fun
import compot.optimizer.base as base
import compot.optimizer.classic as cls



class StepSizeSchedule:
    def get_next_tau_sigma_eps(self, nit, x, s, y, tau, sigma, eps):
        return (tau, sigma, eps)

##
# min_x max_y <Ax - b, y> + f(x) - g*(y)
##
class Parameters(base.Parameters):
    def __init__(self, maxit=500, tol=1e-5, epsilon=1e-12, sigma_init = 1.0, tau_init = 1.0, rho = 0.5, class_oracle = cls.LBFGS,
                 params_oracle = cls.Parameters(), step_size_schedule = StepSizeSchedule()):
        super().__init__(maxit, tol, epsilon)

        self.sigma_init = sigma_init
        self.tau_init = tau_init
        self.rho = rho
        self.class_oracle = class_oracle
        self.params_oracle = params_oracle
        self.step_size_schedule = step_size_schedule

class Penalty(fun.SecondDiffable):
    def __init__(self, proxable, sigma):
        self._proxable = proxable
        self._sigma = sigma

    def set_sigma(self, sigma):
        self._sigma = sigma

    def eval(self, x):
        y = self._proxable.eval_prox(x, 1 / self._sigma)
        return 0.5 * self._sigma * np.dot(x - y, x - y) + self._proxable.eval(y)

    def eval_gradient(self, x):
        y = self._proxable.eval_prox(x, 1 / self._sigma)
        return self._sigma * (x - y)

    def eval_Hessian(self, x):
        assert isinstance(self._proxable, fun.SemidiffableProxable)

        return self._sigma * (np.identity(x.shape[0]) - self._proxable.eval_Jacobian_prox(x, 1 / self._sigma))

class AugmentedLagrangianFunction(fun.SecondDiffable):
    def __init__(self, x, s, y, problem, tau, sigma):
        #primal
        self.x = x
        self.s = s
        self.sigma = sigma

        #dual
        self.y = y
        self.tau = tau

        self.penalty = Penalty(problem.g, sigma)
        self.problem = problem

    def set_sigma(self, sigma):
        self.sigma = sigma
        self.penalty.set_sigma(sigma)

    def set_tau(self, tau):
        self.tau = tau

    def eval(self, s):
        y = (self.problem.A.apply(s) - self.problem.b) + self.y / self.sigma

        return (
                self.problem.f.eval(s)
                + self.penalty.eval(y)#+ (1 / self.sigma) * self.inf_convolution.eval(y)#
                + (0.5 / self.tau) * np.dot(s - self.x, s - self.x)
        )


    def eval_gradient(self, s):
        y = (self.problem.A.apply(s) - self.problem.b) + self.y / self.sigma
        return (
                self.problem.f.eval_gradient(s)
                + self.problem.A.apply_transpose(self.penalty.eval_gradient(y))
                + (1 / self.tau) * (s - self.x)
        )

    def eval_Hessian(self, s):
        y = (self.problem.A.apply(s) - self.problem.b) + self.y / self.sigma
        H = self.penalty.eval_Hessian(y)

        return (
                self.problem.f.eval_Hessian(s)
                + np.dot(np.dot(self.problem.A._A.T, H), self.problem.A._A)
                + (1 / self.tau) * np.identity(s.shape[0])
        )

class Status(base.Status):
    def __init__(self, nit=0, res=np.Inf, success=False, tau=1., sigma=1., eps=1e-13, status_oracle=base.Status(), cumsum_iters_inner=0):
        super().__init__(nit, res, success)
        self.tau = tau
        self.sigma = sigma
        self.eps = eps
        self.status_oracle = status_oracle
        self.cumsum_iters_inner = cumsum_iters_inner


class AugmentedLagrangianMethod(base.PrimalDualIterativeOptimizer):
    def __init__(self, params: Parameters, problem: base.SaddlePointProblem, callback = None):
        super().__init__(params, problem, callback)
        self.status = Status()



    def setup(self):
        self.tau = self.params.tau_init
        self.sigma = self.params.sigma_init
        self.eps = self.params.epsilon

        self.augmented_lagrangian = AugmentedLagrangianFunction(self.x, self.s, self.y, self.problem,
                                                                self.tau, self.sigma)


    def pre_step(self, _):
        return np.inf

    def step(self, k):
        self.subproblem = base.DiffableOptimizationProblem(self.s, self.augmented_lagrangian)

        def callback_oracle(s, status):
            print("    ", status.nit, status.gamma, np.linalg.norm(self.subproblem.diffable.eval_gradient(s)))

            return self.stopping_criterion(s, self.augmented_lagrangian.eval_gradient(s))

        oracle = self.params.class_oracle(self.params.params_oracle, self.subproblem, callback=callback_oracle)
        self.status.status_oracle = oracle.run()
        self.s[:] = oracle.x[:]
        e = self.augmented_lagrangian.eval_gradient(self.s)
        self.x[:] = self.s - self.tau * e

        self.status.cumsum_iters_inner += self.status.status_oracle.nit

        #self.y[:] = self.augmented_lagrangian.inf_convolution.eval_log_gradient((np.dot(self.problem.A._A, self.s) - self.problem.b) * self.sigma + self.y)

        self.y[:] = self.problem.g.get_conjugate().eval_prox(
             (self.problem.A.apply(self.s) - self.problem.b) * self.sigma + self.y
        )

        self.tau, self.sigma, self.eps = self.params.step_size_schedule.get_next_tau_sigma_eps(k + 1, self.x, self.s, self.y,
                                                                                               tau=self.tau, sigma=self.sigma, eps=self.eps)
        self.augmented_lagrangian.set_tau(self.tau)
        self.augmented_lagrangian.set_sigma(self.sigma)
        self.status.tau = self.tau
        self.status.sigma = self.sigma
        self.status.eps = self.eps



    def stopping_criterion(self, s, e):
        left = 0.5 * self.tau * self.tau * np.dot(e, e)

        y = self.problem.g.get_conjugate().eval_prox(
            (self.problem.A.apply(s) - self.problem.b) * self.sigma + self.y
        )

        right = (self.params.rho *
                  (0.5 * np.dot(y - self.y, y - self.y) / self.tau
                   + 0.5 * np.dot(s - self.x, s - self.x) / self.sigma)
                 )

        return left <= right + self.eps