import matplotlib.pyplot as plt
import numpy as np
import convex_optim.function as fun
import convex_optim.optimizer_base as opt_base
import convex_optim.optimizer_lipschitz as opt_lip

##
# minimize 1/2|Ax - b|^2 + lamb |x|_1
##
m, n = 300, 400
np.random.seed(12)
A = 2 * np.random.rand(m, n) - 1
b = 2 * np.random.rand(m)
lamb = 0.015

x0 = np.random.rand(n)

# setup problem define splitting
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
problem = opt_base.CompositeOptimizationProblem(x0, f, g)


#setup optimizer
params = opt_lip.Parameters()
params.maxit = 12000
params.tol = 1e-13
params.mem = 10
params.backtracking = False
params.epsilon = 1e-12
params.gamma_init = -1

#collect objective values for plotting within callback
def callback(x, status):
    objective_values.append(problem.eval_objective(x))
    print("k", status.nit, "objective", problem.eval_objective(x), "residual", status.res, "gamma", status.gamma, "tau", status.tau)


objective_values = []
optimizer = opt_lip.LBFGSPanoc(params, problem, callback)
optimizer.run()

params = opt_lip.Parameters()
params.maxit = 20000
params.tol = 1e-13
params.mem = 10
params.backtracking = True
params.epsilon = 1e-15
params.gamma_init = 100

# plot suboptimality
plt.semilogy(np.array(objective_values) - objective_values[-1], label="$F(x) - F^*$")
plt.legend()
plt.show()


