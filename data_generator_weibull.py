import numpy as np
import matplotlib.pyplot as plt
import numpy.random as rnd
import scipy.optimize as opt 
from solver_weibull import RS_solver

#parameters of the simulation
n = 200    #sample size 
p = 70     #number of covariates     
m = 500    #number of experiments
zeta = p/n #dimensionality ratio

#true parameters
beta0 = np.zeros(p)
beta0[0] = 1.0
phi0  = np.log(0.3)
rho0 = 0.5

#generate the data
X = rnd.normal(size = (n,p))
s = rnd.random(size = n)
lp0 = X@beta0
Y = (np.log(-np.log(s))-phi0 - lp0)/rho0


#define minus the log-likelihood 
def l(theta):
    beta = theta[:-2]
    phi = theta[-2]    
    lrho = theta[-1]
    lp = X@beta
    return (-sum(np.exp(lrho)*Y + phi + lp - np.exp(np.exp(lrho)*Y+ phi + lp))/n - lrho)

#define the matrix that contains the ML estimators 
D = np.empty((m,p+2))

#repeat the experiment m times
for k in range(m):
    #generate the data
    X = rnd.normal(size = (n,p))
    s = rnd.random(size = n)
    lp0 = X@beta0
    Y = (np.log(-np.log(s))-phi0 - lp0)/rho0
    #fit the model
    x0 = np.zeros(p+2)
    #the minimization routine is called three times, so that at each step 
    #we obtain a better approximation of the minimum
    fit = opt.minimize(l,x0,method = 'BFGS',tol = 1.0e-13)
    fit = opt.minimize(l,fit.x,method = 'BFGS',tol = 1.0e-13)
    fit = opt.minimize(l,fit.x,method = 'BFGS',tol = 1.0e-13)
    #store the resutl in the matrix D 
    D[k,:] = fit.x

#store the results
Beta_ml = D[:,:-2]
Phi_ml = D[:,-2]
Rho_ml = np.exp(D[:,-1])

#compute the solution of the RS equations for the weibull model 
RS = RS_solver(zeta)

#store the results 
v = RS[0]
phi=RS[1]
rho=RS[2]


#define the standard normal density 
def normal(x,mu,sigma):
    w = (x-mu)
    return np.exp(-0.5*((w/sigma)**2+np.log(2*np.pi) +2.0*np.log(sigma) ))


#plot the figures 

#beta1
plt.figure()
plt.title(r"$n=$"+str(n)+r" $\zeta=$"+'{:.2f}'.format(p/n)+r" $\theta_0=$"+str(beta0[0]))
plt.hist(Beta_ml[:,0],density=True,color='grey')
x = np.linspace(min(Beta_ml[:,0]),max(Beta_ml[:,0]),10000)
plt.plot(x,normal(x,rho*beta0[0],v/np.sqrt(p)))
plt.xlabel(r"$\hat{\mathbf{\beta}}_n'\mathbf{e}_1$")
plt.savefig('beta1.png')

#beta3
plt.figure()
plt.title(r"$n=$"+str(n)+r" $\zeta=$"+'{:.2f}'.format(p/n)+r" $\theta_0=$"+str(beta0[0]))
plt.hist(Beta_ml[:,p-1],density=True, color='grey')
x = np.linspace(min(Beta_ml[:,2]),max(Beta_ml[:,2]),10000)
plt.plot(x,normal(x,rho*beta0[2],v/np.sqrt(p)))
plt.xlabel(r"$\hat{\mathbf{\beta}}_n'\mathbf{e}_3$")
plt.savefig('betap.png')

#intercept (phi)
plt.figure()
plt.title(r"$n=$"+str(n)+r" $\zeta=$"+'{:.2f}'.format(p/n)+r" $\theta_0=$"+str(beta0[0]))
plt.hist(Phi_ml,density=True,color = 'lightgrey',label = r"$\hat{\phi}_n$")
plt.hist((Phi_ml-phi)/rho,density=True,color = 'grey',label = r"$\hat{\phi}^{\sim}_n$")
plt.axvline(x= phi0,color = 'black', linestyle = '-.')
plt.legend()
plt.savefig('phi.png')


# power parameter (rho)
plt.figure()
plt.title(r"$n=$"+str(n)+r" $\zeta=$"+'{:.2f}'.format(p/n)+r" $\theta_0=$"+str(beta0[0]))
plt.hist(Rho_ml,density=True,color = 'lightgrey',label = r"$\hat{\rho}_n$")
plt.hist(Rho_ml/rho,density=True,color = 'grey',label = r"$\hat{\rho}^{\sim}_n$")
plt.axvline(x= rho0,color = 'black', linestyle = '-.')
plt.legend()
plt.savefig('rho.png')
plt.show()

    
    
