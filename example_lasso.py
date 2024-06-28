import matplotlib.pyplot as plt
import numpy as np
import compot.calculus.function as fun
import compot.optimizer.base as base
import compot.optimizer.lipschitz as lip

##
# minimize 1/2|Ax - b|^2 + lamb |x|_1
##
m, n = 300, 400
np.random.seed(12)
A = 2 * np.random.rand(m, n) - 1
b = 2 * np.random.rand(m)
lamb = 0.015

# setup individual cost functions in composite minimization problem
#min_x f(x) + g(x)
f = fun.AffineCompositeLoss(
            fun.NormPower(2, 2),
            fun.LinearTransform(A),
            b
        )


g = fun.FunctionTransform(
    fun.OneNorm(),
    rho=lamb
)


# define composite optimization problem with initial point
x0 = np.random.rand(n)
problem = base.CompositeOptimizationProblem(x0, f, g)


#setup optimizer Panoc with LBFGS oracle
params = lip.Parameters()
params.maxit = 50000
params.tol = 1e-13

#collect objective values for plotting within callback
def callback(x, status):
    objective_values.append(problem.eval_objective(x))
    print("k", status.nit, "objective", problem.eval_objective(x), "residual", status.res, "gamma", status.gamma, "tau", status.tau)


objective_values = []
optimizer = lip.LBFGSPanoc(params, problem, callback)
optimizer.run()

print(problem.eval_objective(optimizer.x))

# plot suboptimality
plt.semilogy(np.array(objective_values) - objective_values[-1], label="$F(x) - F^*$")
plt.legend()
plt.show()


