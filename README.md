# compot - COMPosite Optimization Toolbox

Python library for nonsmooth convex composite optimization
problems ![equation](https://latex.codecogs.com/png.image?\dpi{110}\inline\min_x&space;f(x)&plus;g(x))

****

[//]: # (The frontend includes an implementations of a simple calculus for derivatives and proximal operators for a convinient description of the optimization problem. The backend includes implementations of &#40;among other methods&#41; LBFGS, Semi-smooth Newton and universal adaptive proximal gradient methods)

## Usage and example

* Specify problem data for ![equation](https://latex.codecogs.com/png.image?\dpi{110}f(x)=\tfrac{1}{2}\|Ax-b\|^2)
  and ![equation](https://latex.codecogs.com/png.image?\dpi{110}g(x)=\lambda\|x\|_1) using numpy

      import numpy as np
        
      m, n = 300, 400
      np.random.seed(12)
      A = 2 * np.random.rand(m, n) - 1
      b = 2 * np.random.rand(m)
      lamb = 0.015

* Setup composite problem with composite terms and initial point

      import compot.calculus.function as fun
      import compot.optimizer.base as base

      f = fun.AffineCompositeLoss(
              fun.NormPower(2, 2),
              fun.MatrixLinearTransform(A),
              b
      )
        
      g = fun.FunctionTransform(
              fun.OneNorm(),
              rho=lamb
      )
        
      x0 = np.random.rand(n)
      problem = base.CompositeOptimizationProblem(x0, f, g)

* Setup composite optimizer and run

      
      import compot.optimizer.lipschitz as lip
        
      params = lip.Parameters()
      params.maxit = 5000
      params.tol = 1e-13

      def callback(x, status):
          print("k", status.nit, "objective", problem.eval_objective(x), "residual", status.res)
        

      optimizer = lip.LBFGSPanoc(params, problem, callback)
      optimizer.run()

* Access solution

      minimizer = optimizer.x
      print(problem.eval_objective(minimizer))

****
If you intend to use this package for scientific purposes please cite

**Adaptive proximal gradient methods are universal without approximation** (K. A. Oikonomidis, E. Laude, P. Latafat, A.
Themelis, P. Patrinos; ICML 2024) [https://arxiv.org/abs/2402.06271]

**A simple and efficient algorithm for nonlinear model predictive control** (L. Stella, A. Themelis, P. Sopasakis, P.
Patrinos; CDC 2017) [https://arxiv.org/abs/1709.06487]
