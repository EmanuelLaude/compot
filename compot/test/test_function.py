import numpy as np
import compot.calculus.function as fun


def test_prox_logistic():
    n = 300
    y = np.random.randn(n)

    step_size = 0.277

    logistic = fun.LogisticLoss()

    x = logistic.eval_prox(y, step_size)

    grad = (x - y) / step_size + logistic.eval_gradient(x)

    print(np.linalg.norm(grad))

test_prox_logistic()


def test_proxable_transform(f, n, b, c, rho, sigma, gamma):
    y = 1 * np.random.randn(n)

    #\rho f(x / \sigma - b) + <c,x> + \gamma/2 * |x|^2
    proxable = fun.FunctionTransform(f, rho = rho, sigma = sigma, gamma = gamma, b = b, c = c)

    step_size = np.random.rand()

    x = proxable.eval_prox(y, step_size)

    grad = (1 / step_size) * (x - y) + (rho / sigma) * f.eval_gradient(x / sigma - (b if not b is None else 0.)) + (c if not c is None else 0.) + gamma * x

    print(np.linalg.norm(grad))

    if np.linalg.norm(grad) < 1e-12:
        return True
    else:
        return False

def test_diffable_transform(f, n, b, c, rho, sigma, gamma):
    x = np.random.randn(n)

    diffable = fun.FunctionTransform(f, rho = rho, sigma = sigma, gamma = gamma, b = b, c = c)


    res = (rho / sigma) * f.eval_gradient(x / sigma - (b if not b is None else 0.)) + (c if not c is None else 0.) + gamma * x - diffable.eval_gradient(x)


    print(np.linalg.norm(res))

    if np.linalg.norm(res) < 1e-12:
        return True
    else:
        return False

callback = None


test_proxable_transform(fun.NormPower(2, 2), 300, None, None, np.random.rand(), np.random.rand(), np.random.rand())
test_proxable_transform(fun.NormPower(2, 2), 300, None, None, 1., 1., 0.)
test_proxable_transform(fun.LogisticLoss(), 300, np.random.randn(300), np.random.randn(300), np.random.rand(), np.random.rand(), np.random.rand())

test_diffable_transform(fun.NormPower(2, 2), 300, None, None, np.random.rand(), np.random.rand(), np.random.rand())
test_diffable_transform(fun.LogisticLoss(), 300, np.random.randn(300), np.random.randn(300), np.random.rand(), np.random.rand(), np.random.rand())

