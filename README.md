# Solvers for the Replica symmetric equations

In this repository you can find the numerical routines implemented by Emanuele Massa to solve the RS equations for the following models:

1) Log-Logistic AFT model without censoring
2) Weibull model without censoring
3) Logit regression model for binary data

For further reference to the theory we refer to [@massa].

For models 1 and 2 you can find a routine that computes the order parameters of the theory from the RS equations, the user need only to specify zeta = p/n. In particular the program in the file "solver_log_log.py" and "solver_weibull.py" compute the order parameters for several values of zeta (see the code).

For model 3 you can find two routines, the routine "ideal_solver_logit.py" computes the solution of the RS equations by taking as input zeta, the true signal strength and the true intercept of the model. These are generally unknown, or must be estimated. For this reason we also provide a different implementation "solver_logit.py" (again see () for further details) that takes as input only measurable quantities: zeta, the Maximum Likelihood (ML) estimate of the signal strength and the ML estimate of the intercept.

We also share the python routines used to simulate the data and produce the figures of the paper (). These are named 
1) "data_generator_log_log.py"
2) "data_generator_weibull.py"
3) "data_generator_logit.py"
for the three models respectively.

references:
  - id: massa_overfitting
    title: Correction of overfitting bias in regression models
    author:
      - family: Massa
        given: Emanuele 
    container-title: Arxiv
    type: article-journal
    issued:
      year: 2023
