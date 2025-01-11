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


def test_proxable_transform(f, n, b, c, C, rho, sigma):
    y = 1 * np.random.randn(n)

    #\rho f(x / \sigma - b) + <c,x>
    proxable = fun.FunctionTransform(f, rho = rho, sigma = sigma, b = b, c = c, C = C)

    step_size = np.random.rand()

    x = proxable.eval_prox(y, step_size)

    grad = (1 / step_size) * (x - y) + (rho / sigma) * f.eval_gradient(x / sigma - (b if not b is None else 0.)) + (c if not c is None else 0.)

    print(np.linalg.norm(grad))

    if np.linalg.norm(grad) < 1e-12:
        return True
    else:
        return False

def test_diffable_transform(f, n, b, c, C, rho, sigma):
    x = np.random.randn(n)

    diffable = fun.FunctionTransform(f, rho = rho, sigma = sigma, b = b, c = c, C = C)


    res = (rho / sigma) * f.eval_gradient(x / sigma - (b if not b is None else 0.)) + (c if not c is None else 0.) - diffable.eval_gradient(x)


    print(np.linalg.norm(res))

    if np.linalg.norm(res) < 1e-12:
        return True
    else:
        return False

callback = None


test_proxable_transform(fun.NormPower(2, 2), 300, None, None, 5., np.random.rand(), np.random.rand())
test_proxable_transform(fun.NormPower(2, 2), 300, None, None, 5., 1., 1.)
test_proxable_transform(fun.LogisticLoss(), 300, np.random.randn(300), np.random.randn(300), -5., np.random.rand(), np.random.rand())

test_diffable_transform(fun.NormPower(2, 2), 300, None, None, 5., np.random.rand(), np.random.rand())
test_diffable_transform(fun.LogisticLoss(), 300, np.random.randn(300), np.random.randn(300), 5., np.random.rand(), np.random.rand())

linear = fun.FunctionTransform(fun.Constant(C = 10*np.random.rand()-5), b=np.random.randn(300), c=np.random.randn(300), C=10*np.random.rand()-5, rho=np.random.rand(), sigma=np.random.rand())
conj = linear.get_conjugate()
x = 2*np.random.rand(300)-1
x_star = linear.eval_gradient(x)
print(np.dot(x_star, x) - conj.eval(x_star) - linear.eval(x))


lse = fun.LogSumExp()
x = 2*np.random.rand(300)-1
conj = lse.get_conjugate()

x_star = lse.eval_gradient(x)
print(np.dot(x_star, x) - conj.eval(x_star) - lse.eval(x))

eus = fun.EntropyUnitSimplex()
a = 2*np.random.rand(300)-1
x = np.exp(a) / np.sum(np.exp(a))
x_star = eus.eval_gradient(x)
conj = eus.get_conjugate()

print(np.dot(x_star, x) - conj.eval(x_star) - eus.eval(x))

ll = fun.LogisticLoss()
x = 10*np.random.rand(300)-5
conj = ll.get_conjugate()
x_star = ll.eval_gradient(x)

print(np.dot(x_star, x) - conj.eval(x_star) - ll.eval(x))


fde = fun.FermiDiracEntropy()
x = np.random.rand(300)
conj = fde.get_conjugate()
x_star = fde.eval_gradient(x)

print(np.dot(x_star, x) - conj.eval(x_star) - fde.eval(x))






