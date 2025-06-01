import numpy as np

import compot.calculus.function as fun
import compot.optimizer.base as base
import compot.optimizer.classic as cls


from abc import abstractmethod, ABC

class StepSizeSchedule(ABC):
    @abstractmethod
    def get_next_tau_sigma_eps(self, nit, x, s, y, tau, sigma, eps):
        pass

class ConstantStepSizeSchedule(StepSizeSchedule):
    def get_next_tau_sigma_eps(self, nit, x, s, y, tau, sigma, eps):
        return (tau, sigma, eps)

class FeasibilityStepSizeSchedule(StepSizeSchedule):
    def __init__(self, x_init, A, b, penalty, beta, zeta, sigma_max):
        self.A = A
        self.b = b
        self.feas = penalty.eval(self.A.apply(x_init) - self.b)
        self.sigma_max = sigma_max
        self.beta = beta
        self.zeta = zeta
        self.penalty = penalty


    def get_next_tau_sigma_eps(self, nit, x, s, _, tau, sigma, eps):
        feas = self.penalty.eval(self.A.apply(s) - self.b)

        if feas > self.zeta * self.feas:
            sigma = min(self.sigma_max, sigma * self.beta)

        self.feas = feas

        return (tau, sigma, eps)


class ExpIncrStepSizeSchedule(StepSizeSchedule):
    def __init__(self, tau_max, sigma_max, beta, step_inter):
        self.tau_max = tau_max
        self.sigma_max = sigma_max
        self.beta = beta
        self.step_inter = step_inter


    def get_next_tau_sigma_eps(self, nit, x, s, _, tau, sigma, eps):
        if nit % self.step_inter == (self.step_inter - 1):
            tau = min(self.tau_max, tau * self.beta)
            sigma = min(self.sigma_max, sigma * self.beta)

        return (tau, sigma, eps)

##
# min_x max_y <Ax - b, y> + f(x) - g*(y)
##
class Parameters(base.Parameters):
    def __init__(self, maxit=500, tol=1e-5, epsilon=1e-12, sigma_init = 1.0, tau_init = 1.0, rho = 0.5, class_oracle = cls.LBFGS,
                 params_oracle = cls.Parameters(), step_size_schedule = ConstantStepSizeSchedule, beta = 0.5, zeta = 0.9, sigma_max = np.inf, tau_max = np.inf,
                step_inter = 10, theta=1.):
        super().__init__(maxit, tol, epsilon)

        self.sigma_init = sigma_init
        self.tau_init = tau_init
        self.rho = rho
        self.class_oracle = class_oracle
        self.params_oracle = params_oracle
        self.step_size_schedule = step_size_schedule
        self.sigma_max = sigma_max
        self.tau_max = tau_max
        self.step_inter = step_inter
        self.beta = beta
        self.zeta = zeta
        self.theta = theta

class Penalty(fun.SecondDiffable):
    def __init__(self, proxable, sigma):
        self._proxable = proxable
        self._sigma = sigma

    def set_sigma(self, sigma):
        self._sigma = sigma

    def eval(self, x):
        y = self._proxable.eval_prox(x, 1 / self._sigma)

        return 0.5 * self._sigma * np.dot(x.reshape(-1) - y.reshape(-1), x.reshape(-1) - y.reshape(-1)) + self._proxable.eval(y)



    def eval_gradient(self, x):
        y = self._proxable.eval_prox(x, 1 / self._sigma)



        #lamb, U = np.linalg.eig(x)

        #np.dot(U, np.dot(np.diag(np.maximum(0., lamb)), U.T))
        #print(np.dot(U, np.dot(np.diag(np.maximum(0., lamb)), U.T)) - self._sigma * (x - y))

        return self._sigma * (x - y)

    def eval_Hessian(self, x):
        assert isinstance(self._proxable, fun.SemidiffableProxable)

        return self._sigma * (np.identity(x.shape[0]) - self._proxable.eval_Jacobian_prox(x, 1 / self._sigma))

class ProximalAugmentedLagrangianFunction(fun.SecondDiffable):
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
                + np.dot(np.dot(self.problem.A._matrix.T, H), self.problem.A._matrix)
                + (1 / self.tau) * np.identity(s.shape[0])
        )

class Status(base.Status):
    def __init__(self, nit=0, res=np.inf, success=False, tau=1., sigma=1., eps=1e-13, status_oracle=base.Status(), cumsum_iters_inner=0):
        super().__init__(nit, res, success)
        self.tau = tau
        self.sigma = sigma
        self.eps = eps
        self.status_oracle = status_oracle
        self.cumsum_iters_inner = cumsum_iters_inner


class ProximalAugmentedLagrangianMethod(base.PrimalDualIterativeOptimizer):
    def __init__(self, params: Parameters, problem: base.SaddlePointProblem, callback = None):
        super().__init__(params, problem, callback)
        self.status = Status()



    def setup(self):
        self.tau = self.params.tau_init
        self.sigma = self.params.sigma_init
        self.eps = self.params.epsilon

        self.augmented_lagrangian = ProximalAugmentedLagrangianFunction(self.x, self.s, self.y, self.problem,
                                                                        self.tau, self.sigma)

        if self.params.step_size_schedule == FeasibilityStepSizeSchedule:
            self.step_size_schedule = FeasibilityStepSizeSchedule(self.problem.x_init, self.problem.A, self.problem.b,
                                                                  self.augmented_lagrangian.penalty, self.params.beta,
                                                                  self.params.zeta, self.params.sigma_max)
        elif self.params.step_size_schedule == ExpIncrStepSizeSchedule:
            self.step_size_schedule = ExpIncrStepSizeSchedule(self.params.tau_max, self.params.sigma_max,
                                                              self.params.beta, self.params.step_inter)
        elif self.params.step_size_schedule == ConstantStepSizeSchedule:
            self.step_size_schedule = ConstantStepSizeSchedule()
        else:
            self.step_size_schedule = None


    def pre_step(self, _):
        return np.inf

    def step(self, k):
        self.subproblem = base.DiffableOptimizationProblem(self.s, self.augmented_lagrangian)

        def callback_oracle(s, status):
            print("    ", status.nit, np.linalg.norm(self.subproblem.diffable.eval_gradient(s)))

            return self.stopping_criterion(s, self.augmented_lagrangian.eval_gradient(s))

        oracle = self.params.class_oracle(self.params.params_oracle, self.subproblem, callback=callback_oracle)
        self.status.status_oracle = oracle.run()
        self.s[:] = oracle.x[:]
        e = self.augmented_lagrangian.eval_gradient(self.s)
        self.x[:] = self.s - self.tau * e

        self.status.cumsum_iters_inner += self.status.status_oracle.nit


        self.y[:] = self.problem.g.get_conjugate().eval_prox(
            (self.problem.A.apply(self.s) - self.problem.b) * self.sigma + self.y,
            self.sigma
        )

        self.tau, self.sigma, self.eps = self.step_size_schedule.get_next_tau_sigma_eps(k + 1, self.x, self.s, self.y,
                                                                                               tau=self.tau, sigma=self.sigma, eps=self.eps)
        self.augmented_lagrangian.set_tau(self.tau)
        self.augmented_lagrangian.set_sigma(self.sigma)
        self.status.tau = self.tau
        self.status.sigma = self.sigma
        self.status.eps = self.eps



    def stopping_criterion(self, s, e):
        left = 0.5 * self.tau * self.tau * np.dot(e.reshape(-1), e.reshape(-1))

        y = self.problem.g.get_conjugate().eval_prox(
            (self.problem.A.apply(s) - self.problem.b) * self.sigma + self.y,
            self.sigma
        )

        right = (self.params.rho *
                  (0.5 * np.dot(y.reshape(-1) - self.y.reshape(-1), y.reshape(-1) - self.y.reshape(-1)) ** 2 / self.tau
                   + 0.5 * np.dot(s - self.x, s - self.x) / self.sigma)
                 )

        return left <= right + self.eps


class PrimalDualHybridGradientMethod(base.PrimalDualIterativeOptimizer):
    def __init__(self, params: Parameters, problem: base.SaddlePointProblem, callback = None):
        super().__init__(params, problem, callback)
        self.status = Status()

    def setup(self):
        L = self.problem.A.get_norm()
        self.tau = 1 / L
        self.sigma = 1 / L
        self.theta = self.params.theta

    def pre_step(self, _):
        self.status.tau = self.tau
        self.status.sigma = self.sigma
        return np.inf

    def step(self, k):
        x = self.problem.f.eval_prox(
            self.x - self.tau * self.problem.A.apply_transpose(self.y),
            self.tau
        )
        self.s = x + self.theta * (x - self.x)
        self.x = x
        self.y = fun.FunctionTransform(
            self.problem.g.get_conjugate(),
            c = self.problem.b
        ).eval_prox(
            self.y + self.sigma * self.problem.A.apply(self.s),
            self.sigma
        )
        self.status.cumsum_iters_inner += 1


