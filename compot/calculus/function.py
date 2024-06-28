from abc import ABC, abstractmethod
import numpy as np


##
# Interfaces
##
class Function(ABC):
    @abstractmethod
    def eval(self, x):
        pass

class Proxable(Function):
    @abstractmethod
    def eval_prox(self, x, step_size=1.0):
        pass

class SemidiffableProxable(Proxable):
    @abstractmethod
    def eval_Jacobian_prox(self, x, step_size=1.0):
        pass

class Diffable(Function):
    @abstractmethod
    def eval_gradient(self, x):
        pass

    def get_Lip_gradient(self):
        return np.Inf

class SecondDiffable(Diffable):
    @abstractmethod
    def eval_Hessian(self, x):
        pass

    def get_Lip_Hessian(self):
        return np.Inf

#\rho f(x / \sigma - b) + <c,x> + \gamma/2 * |x|^2
class FunctionTransform(SecondDiffable, SemidiffableProxable):
    def __init__(self, function, rho = 1., sigma = 1., gamma = 0., b = None, c = None):
        self._rho = rho
        self._sigma = sigma
        self._gamma = gamma
        self._b = b
        self._c = c
        self._function = function

    def eval(self, x):
        y = x / self._sigma if self._b is None else x / self._sigma - self._b

        return (
            self._rho * self._function.eval(y)
            + (0. if self._c is None else np.dot(x, self._c))
            + (0. if self._gamma == 0. else self._gamma * 0.5 * np.dot(x, x))
        )

    def eval_gradient(self, x):
        assert isinstance(self._function, Diffable)

        y = x / self._sigma if self._b is None else x / self._sigma - self._b

        return (self._rho * self._function.eval_gradient(y) / self._sigma
                + (0. if self._c is None else self._c)
                + (0. if self._gamma == 0. else self._gamma * x)
        )

    def get_Lip_gradient(self):
        assert isinstance(self._function, Diffable)

        return (self._rho / self._sigma ** 2) * self._function.get_Lip_gradient() + self._gamma


    def eval_Hessian(self, x):
        assert isinstance(self._function, SecondDiffable)

        y = x / self._sigma if self._b is None else x / self._sigma - self._b

        return (self._rho * self._function.eval_Hessian(y) / (self._sigma ** 2)
                + (0. if self._gamma == 0. else self._gamma * np.identity(x.shape[0]))
        )

    def eval_prox(self, x, step_size=1.0):
        assert isinstance(self._function, Proxable)

        omega = 1. + step_size * self._gamma
        delta = 1. / ((self._sigma ** 2) * omega)

        #r = (self._rho / omega) * x - ((step_size * self._rho) / self._sigma) * (1. / omega) * self._c
        #z = r + ((step_size * self._rho * self._gamma - self._rho) * self._b) / omega
        q = delta * self._sigma * (
                     (x - step_size * (self._c if not self._c is None else 0.))
                    - self._sigma * omega * (self._b if not self._b is None else 0.)
        )

        return (self._function.eval_prox(q, step_size * self._rho * delta) + (self._b if not self._b is None else 0.)) * self._sigma

    def eval_Jacobian_prox(self, x, step_size=1.0):
        assert isinstance(self._function, SemidiffableProxable)

        omega = 1. + step_size * self._gamma
        delta = 1. / ((self._sigma ** 2) * omega)

        q = delta * self._sigma * (
                     (x - step_size * (self._c if not self._c is None else 0.))
                    - self._sigma * omega * (self._b if not self._b is None else 0.)
        )

        return self._sigma * delta * self._sigma * self._function.eval_Jacobian_prox(q, step_size * self._rho * delta)

##
# Implementations
##
class NormPower(SecondDiffable, Proxable):
    def __init__(self, power, norm = 2):
        assert norm > 1. and power > 1.
        assert norm == power or norm == 2

        self._norm = norm
        self._power = power

    def eval(self, x):
        if self._norm == self._power:
            return np.sum(np.power(np.abs(x), self._power)) / self._power
        elif self._norm == 2:
            return np.power(np.linalg.norm(x, 2), self._power) / self._power

    def eval_gradient(self, x):
        if self._norm == self._power:
            return np.sign(x) * np.power(np.abs(x), self._power - 1)
        elif self._norm == 2:
            return np.power(np.linalg.norm(x, 2), self._power - 2) * x

    def eval_Hessian(self, x):
        assert self._power >= 2

        if self._norm == self._power:
            return (self._power - 1) * np.diag(np.power(np.abs(x), self._power - 2))
        elif self._norm == 2:
            return ((self._power - 2) * np.power(np.linalg.norm(x, 2), self._power - 4) * np.outer(x, x)
                    + np.power(np.linalg.norm(x, 2), self._power - 2) * np.identity(x.shape[0]))

    def eval_prox(self, x, step_size=1.0):
        assert self._power == 2 and self._norm == 2

        return x / (1. + step_size)

    def get_Lip_gradient(self):
        if self._norm == 2. and self._power == 2.:
            return 1.

        return np.Inf


class PowerHingeLoss(Diffable):
    def __init__(self, power):
        assert power > 1

        self._power = power

    def eval(self, x):
        return np.sum(np.power(np.maximum(0., 1 + x), self._power)) / self._power

    def eval_gradient(self, x):
        return np.power(np.maximum(0., 1 + x), self._power - 1)

    def get_Lip_gradient(self):
        if self._power == 2.:
            return 1.

        return np.Inf

# Code adapted from https://github.com/foges/pogs/blob/master/src/include/prox_lib.h
def prox_logistic(v, rho):
    #Initial guess based on piecewise approximation.
    # if v < -2.5:
    #     x = v
    # elif v > 2.5 + 1 / rho:
    #     x = v - 1 / rho
    # else:
    #     x = (rho * v - 0.5) / (0.2 + rho)

    x = (rho * v[:] - 0.5) / (0.2 + rho)
    x[v > 2.5 + 1 / rho] = v[v > 2.5 + 1 / rho] - 1 / rho
    x[v < -2.5] = v[v < -2.5]

    #Newton iteration.
    l = v[:] - 1 / rho
    u = v[:] + 0.0

    for i in range(500):
        inv_ex = 1 / (1 + np.exp(-x[:]))
        f = inv_ex[:] + rho * (x[:] - v[:])
        g = inv_ex[:] * (1 - inv_ex[:]) + rho

        # if f < 0:
        #     l = x
        # else:
        #     u = x
        #     x = x - f / g
        #     x = np.minimum(x, u)
        #     x = np.maximum(x, l)

        l[f < 0] = x[f < 0]

        u[f >= 0] = x[f >= 0]
        x[f >= 0] = x[f >= 0] - f[f >= 0] / g[f >= 0]
        x[f >= 0] = np.minimum(x[f >= 0], u[f >= 0])
        x[f >= 0] = np.maximum(x[f >= 0], l[f >= 0])




    #Guarded method if not converged.
    for i in range(500):
        if np.amax(u - l) < 1e-15:
            break
        g_rho = 1 / (rho * (1 + np.exp(-x[:]))) + (x[:] - v[:])
        # if g_rho > 0:
        #     l = np.maximum(l, x - g_rho)
        #     u = x
        # else:
        #     u = np.minimum(u, x - g_rho)
        #     l = x

        l[g_rho > 0] = np.maximum(l[g_rho > 0], x[g_rho > 0] - g_rho[g_rho > 0])
        u[g_rho > 0] = x[g_rho > 0]
        u[g_rho <= 0] = np.minimum(u[g_rho <= 0], x[g_rho <= 0] - g_rho[g_rho <= 0])
        l[g_rho <= 0] = x[g_rho <= 0]

        x[:] = (u[:] + l[:]) / 2

    return x

class LogisticLoss(SecondDiffable, Proxable):
    def eval(self, x):
        theta = np.maximum(0., x)
        v = np.log(np.exp(-theta) + np.exp(x - theta)) + theta
        return np.sum(v)

    def eval_gradient(self, x):
        theta = np.maximum(0., x)
        v = np.exp(x - theta) / (np.exp(-theta) + np.exp(x - theta))
        return v

    def eval_Hessian(self, x):
        theta = np.maximum(0., x)
        return np.diag((np.exp(x - theta) * np.exp(-theta)) / ((np.exp(-theta) + np.exp(x - theta)) ** 2))

    def eval_prox(self, x, step_size=1.0):
        return prox_logistic(x, 1 / step_size)

    def get_Lip_gradient(self):
        return 0.25

    def get_Lip_Hessian(self):
        return 1 / (6 * np.sqrt(3))

class SumExp(SecondDiffable):
    def eval(self, x):
        return np.sum(np.exp(x))

    def eval_gradient(self, x):
        return np.exp(x)

    def eval_Hessian(self, x):
        return np.diag(np.exp(x))


class LogSumExp(SecondDiffable):
    def eval(self, x):
        theta = np.max(x)
        return (np.log(np.sum(np.exp(x - theta))) + theta)

    def eval_gradient(self, x):
        theta = np.max(x)
        return np.exp(x - theta) / np.sum(np.exp(x - theta))

    def get_Lip_gradient(self):
        return 1.

    def eval_Hessian(self, x):
        theta = np.max(x)
        z = np.exp(x - theta)
        Z = np.sum(z)
        return (Z * np.diag(z) - np.outer(z, z)) / (Z ** 2)


class Zero(SecondDiffable, SemidiffableProxable):
    def eval(self, x):
        return 0.

    def eval_prox(self, x, step_size=1.0):
        return x

    def eval_Jacobian_prox(self, x, step_size=1.0):
        return np.identity(x.shape[0])

    def eval_gradient(self, x):
        return np.zeros(x.shape)

    def get_Lip_gradient(self):
        return 0.

    def eval_Hessian(self, x):
        return np.diag(0.)

    def get_Lip_Hessian(self):
        return 0.

class IndicatorBox(SemidiffableProxable):
    def __init__(self, l = -1., u = 1.):
        self._l = l
        self._u = u

    def eval(self, x):
        if np.max(x) <= self._u and np.min(x) >= self._l:
            return 0
        return np.Inf

    def eval_prox(self, x, _):
        return np.maximum(self._l, np.minimum(self._u, x))

    def eval_Jacobian_prox(self, x, step_size=1.0):
        v = np.zeros(x.shape[0])
        v[np.logical_and(self._l < x, x < self._u)] = 1.
        return np.diag(v)

# code adapted from https://gist.github.com/mblondel/6f3b7aaad90606b98f71
def projection_simplex_sort(v, z=1):
    n_features = v.shape[0]
    u = np.sort(v)[::-1]
    cssv = np.cumsum(u) - z
    ind = np.arange(n_features) + 1
    cond = u - cssv / ind > -1e-18
    if len(ind[cond]) == 0:
        print(u - cssv / ind)

    rho = ind[cond][-1]
    theta = cssv[cond][-1] / float(rho)
    w = np.maximum(v - theta, 0)
    return w

class IndicatorSimplex(SemidiffableProxable):
    def eval(self, x):
        if np.sum(x >= -1e10) == x.shape[0] and np.abs(np.sum(x) - 1) <= 1e-10:
            return 0
        return np.Inf

    def eval_prox(self, x, step_size=1.0):
        return projection_simplex_sort(x)

    def eval_Jacobian_prox(self, x, step_size=1.0):
        y = self.eval_prox(x, step_size)
        S = y > 0
        return np.diag(S) - np.outer(S, S) / np.sum(S)#np.multiply(np.outer(S, S), np.identity(y.shape[0]) - 1 / np.sum(S))



def shrinkage(x, threshold):
    return np.maximum(0., np.abs(x) - threshold) * np.sign(x)


class OneNorm(SemidiffableProxable):
    def eval(self, x):
        return np.sum(np.abs(x))

    def eval_prox(self, x, step_size=1.0):
        return shrinkage(x, step_size)

    def eval_Jacobian_prox(self, x, step_size=1.0):
        v = np.zeros(x.shape)
        v[np.abs(x) >= step_size] = 1
        return np.diag(v)

class TwoNorm(SemidiffableProxable):
    def eval(self, x):
        return np.linalg.norm(x, 2)

    def eval_prox(self, x, step_size=1.0):
        t = np.linalg.norm(x, 2)
        if t >= step_size:
            return (1 - step_size / t) * x
        else:
            return np.zeros(x.shape)

    def eval_Jacobian_prox(self, x, step_size=1.0):
        t = np.linalg.norm(x, 2)
        w = x / t
        return (1 - step_size / t) * np.identity(x.shape[0]) + (step_size / t) * np.outer(w, w)

class Indicator2NormBall(SemidiffableProxable):
    def __init__(self, radius):
        self._radius = radius


    def eval(self, x):
        if np.linalg.norm(x, 2) <= self._radius + 1e-12:
            return 0
        return np.Inf

    def eval_prox(self, x, step_size=1.0):
        if np.linalg.norm(x, 2) <= self._radius:
            return x
        else:
            return self._radius * (x / np.linalg.norm(x, 2))

    def eval_Jacobian_prox(self, x, step_size=1.0):
        w = np.linalg.norm(x, 2)
        if w <= self._radius:
            return np.identity(x.shape[0])
        else:
            return self._radius * (np.identity(x.shape[0]) - np.outer(x, x) / (w ** 2)) / w



class LinearTransform:
    def __init__(self, A):
        self._A = A
        self._norm = np.linalg.norm(self._A, 2)

    def apply(self, x):
        return np.dot(self. _A, x)

    def get_norm(self):
        return self._norm

    def apply_transpose(self, x):
        return np.dot(self._A.T, x)


class AffineCompositeLoss(SecondDiffable):
    def __init__(self, loss, A, b = None):
        self._A = A
        self._norm_A = A.get_norm()
        self._b = b
        self._loss = loss

    def eval(self, x):
        return self._loss.eval(self._A.apply(x) - (0. if self._b is None else self._b))


    def eval_gradient(self, x):
        return self._A.apply_transpose(
            self._loss.eval_gradient(
                self._A.apply(x) - (0. if self._b is None else self._b)
            )
        )

    def eval_Hessian(self, x):
        assert isinstance(self._loss, SecondDiffable)

        B = self._loss.eval_Hessian(
                self._A.apply(x) - (0. if self._b is None else self._b)
            )
        return np.dot(np.dot(self._A._A.T, B), self._A._A)

    def get_Lip_gradient(self):
        return self._norm_A * self._norm_A * self._loss.get_Lip_gradient()

class AdditiveComposite(SecondDiffable):
    def __init__(self, diffables):
        self._diffables = diffables

    def eval(self, x):
        value = 0.
        for diffable in self._diffables:
            value += diffable.eval(x)

        return value

    def eval_gradient(self, x):
        grad = np.zeros(x.shape)
        for diffable in self._diffables:
            grad = grad + diffable.eval_gradient(x)

        return grad

    def eval_Hessian(self, x):
        H = np.zeros([x.shape[0], x.shape[0]])

        for diffable in self._diffables:
            assert isinstance(diffable, SecondDiffable)
            H = H + diffable.eval_Hessian(x)

        return H

    def get_Lip_gradient(self):
        Lip = 0.0
        for diffable in self._diffables:
            Lip += diffable.get_Lip_gradient()

        return Lip
