# Solvers for the Replica symmetric equations

In this repository you can find the numerical routines implemented by Emanuele Massa to solve the RS equations for the following models:

1) Log-Logistic AFT model  $T|\mathbf{X} \sim  \frac{\rho\big(T {\rm e}^{\mathbf{X}'\beta+\phi}\big)^{\rho-1}}{\lambda\big(1+(T {\rm e}^{\mathbf{X}'\beta+\phi })^{\rho}\big)^2}$
2) Weibull model $T|\mathbf{X} \sim \rho_0 T^{\rho_0-1} {\rm e}^{\mathbf{X}'\beta_0+\phi_0 + T^{\rho_0} \exp(\mathbf{X}'\beta_0+\phi_0)} \$
4) Logit regression model for binary data $T = {\rm sign}\Big(\mathbf{X}'\mathbf{\beta}_0+ \phi_0 + \frac{1}{2} Z\Big), \quad Z \sim \frac{{\rm e}^{-z}}{\big(1+{\rm e}^{-z}\big)^2} $

Here $T$ is the response and $\mathbf{X}\in\mathbb{R}^p$ are the covariates or predictors.
The sample size is indicated with $n$.
For further reference to the theory we refer to [https://arxiv.org/abs/2204.05827].

For models 1 and 2 you can find a routine that computes the order parameters of the theory from the RS equations, the user need only to specify $\zeta = p/n$. In particular the program in the file "solver_log_log.py" and "solver_weibull.py" compute the order parameters for several values of zeta (see the code).

For model 3 you can find two routines, the routine "ideal_solver_logit.py" computes the solution of the RS equations by taking as input $\zeta$, the true signal strength and the true intercept of the model. These are generally unknown, or must be estimated. For this reason we also provide a different implementation "solver_logit.py" (again see [https://arxiv.org/abs/2204.05827] for further details) that takes as input only measurable quantities: $\zeta$ and the Maximum Likelihood (ML) estimate of the signal strength $\hat{\theta}_n := \sum_{i=1}^n \|\mathbf{X}_i'\hat{\beta}_n\|^2/n$ and of the intercept $\hat{\phi}_n$.

We also share the python routines used to simulate the data and produce the figures of the manuscript [https://arxiv.org/abs/2204.05827]. These are named 
1) "data_generator_log_log.py"
2) "data_generator_weibull.py"
3) "data_generator_logit.py"
for the three models respectively.
