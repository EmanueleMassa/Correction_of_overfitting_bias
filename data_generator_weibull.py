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
plt.hist(Beta_ml[:,0],density=True,color='lightgrey',label = r"$\mathbf{e}_1'\hat{\mathbf{\beta}}_n$")
beta =Beta_ml[:,0]/rho
plt.hist(beta,density=True,color='grey',label = r"$\mathbf{e}_1'\tilde{\mathbf{\beta}}_n$")
x = np.linspace(min(beta),max(beta),10000)
plt.plot(x,normal(x,beta0[0],v/(rho*np.sqrt(p))),'k-')
plt.axvline(x= beta0[0],color = 'black', linestyle = '-.')
plt.xlabel(r"$\beta_1$")
plt.savefig('betasim1.zeta'+'{:.2f}'.format(p/n)+'.png')

#beta3
plt.figure()
plt.title(r"$n=$"+str(n)+r" $\zeta=$"+'{:.2f}'.format(p/n)+r" $\theta_0=$"+str(beta0[0]))
plt.hist(Beta_ml[:,1],density=True, color='lightgrey',label = r"$\mathbf{e}_2'\hat{\mathbf{\beta}}_n$")
beta =Beta_ml[:,1]/rho
plt.hist(beta,density=True,color='grey',label = r"$\mathbf{e}_2'\tilde{\mathbf{\beta}}_n$")
x = np.linspace(min(beta),max(beta),10000)
plt.plot(x,normal(x,beta0[1],v/(rho*np.sqrt(p))),'k-')
plt.axvline(x= beta0[1],color = 'black', linestyle = '-.')
plt.xlabel(r"$\beta_2$")
plt.savefig('betasim2.zeta'+'{:.2f}'.format(p/n)+'.png')

#phi
plt.figure()
plt.title(r"$n=$"+str(n)+r" $\zeta=$"+'{:.2f}'.format(p/n)+r" $\theta_0=$"+str(beta0[0]))
plt.hist(Phi_ml,density=True,color = 'lightgrey',label = r"$\hat{\phi}_n$")
plt.hist((Phi_ml-phi)/rho,density=True,color = 'grey',label = r"$\tilde{\phi}_n$")
plt.axvline(x= phi0,color = 'black', linestyle = '-.')
plt.xlabel(r"$\phi$")
plt.savefig('phi.zeta'+'{:.2f}'.format(p/n)+'.png')

# rho
plt.figure()
plt.title(r"$n=$"+str(n)+r" $\zeta=$"+'{:.2f}'.format(p/n)+r" $\theta_0=$"+str(beta0[0]))
plt.hist(Rho_ml,density=True,color = 'lightgrey',label = r"$\hat{\rho}_n$")
plt.hist(Rho_ml/rho,density=True,color = 'grey',label = r"$\tilde{\rho}_n$")
plt.axvline(x= rho0,color = 'black', linestyle = '-.')
plt.xlabel(r"$\sigma$")
plt.savefig('rho.zeta'+'{:.2f}'.format(p/n)+'.png')

plt.show()

    
    
