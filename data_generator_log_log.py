import numpy as np
import matplotlib.pyplot as plt
import numpy.random as rnd
import scipy.optimize as opt 
from solver_log_log import RS_solver

#parameters of the simulations
n = 200 #sample size 
p = 20  #number of covariates
m = 500 #number of experiments
zeta = p/n #dimensionality ratio

#true parameters
beta0 = np.zeros(p)
beta0[0] = 2.0
phi0  = 0.5
sigma0 = 1.0

#generate the data
X = rnd.normal(size = (n,p))
s = rnd.random(size = n)
Y = phi0 + X@beta0 + sigma0*np.log(s/(1.0-s))


#define minus the log-likelihood of the model
def l(theta):
    beta = theta[:-2] #associtiations 
    phi = theta[-2]    #intercept = location parameter
    lsigma = theta[-1] #logarithm of scale parameter
    lp = X@beta #linear predictor
    return (-sum(np.exp(-lsigma)*(-Y+phi+lp) - 2*np.log(1.0+np.exp(np.exp(-lsigma)*(-Y+ phi + lp))))/n + lsigma)

#define the matrix that contains the ML estimators 
D = np.empty((m,p+2))

for k in range(m):
    #generate the data
    X = rnd.normal(size = (n,p))
    s = rnd.random(size = n)
    Y = phi0 +  X@beta0 + sigma0*np.log(s/(1.0-s))
    #fit the model
    x0 = np.zeros(p+2)
    #the minimization routine is called three times with increasing precision, so that at each step 
    #we obtain a better approximation of the minimum
    fit = opt.minimize(l,x0,method = 'BFGS',tol = 1.0e-13)
    fit = opt.minimize(l,fit.x,method = 'BFGS',tol = 1.0e-13)
    fit = opt.minimize(l,fit.x,method = 'BFGS',tol = 1.0e-13)
    #store the results in the matrix D 
    D[k,:] = fit.x
#store the results    
Beta_ml = D[:,:-2]
Phi_ml = D[:,-2]
Sigma_ml = np.exp(D[:,-1])
#call the routine that solves the RS equations for the log-logistic model it just needs the value of zeta = p/n
RS = RS_solver(zeta)
#save the output of RS_solver
v = RS[1]
sigma=RS[2]

#define the standard normal density 
def normal(x,mu,sigma):
    w = (x-mu)
    return np.exp(-0.5*((w/sigma)**2+np.log(2*np.pi) +2.0*np.log(sigma) ))

#plot the results

#first component of $\hat{\bbeta}_n$
plt.figure()
plt.title(r"$n=$"+str(n)+r" $\zeta=$"+'{:.2f}'.format(p/n)+r" $\theta_0=$"+str(beta0[0]))
beta =Beta_ml[:,0]
plt.hist(beta,density=True,color='grey',label = r"$\mathbf{e}_1'\hat{\mathbf{\beta}}^{\sim}_n$")
x = np.linspace(min(beta),max(beta),10000)
plt.plot(x,normal(x,beta0[0],v*sigma*sigma0/np.sqrt(p)),'k-',label = r"$\mathcal{N}(\mathbf{e}_1'\mathbf{\beta}_0,v^2/p^2)$")
plt.axvline(x= beta0[0],color = 'black', linestyle = '-.',label = r"$\mathbf{e}_1'\mathbf{\beta}_0$")
plt.legend()
plt.savefig('beta1.zeta'+'{:.2f}'.format(p/n)+'.png')

#third component of $\hat{\bbeta}_n$
plt.figure()
plt.title(r"$n=$"+str(n)+r" $\zeta=$"+'{:.2f}'.format(p/n)+r" $\theta_0=$"+str(beta0[0]))
beta =Beta_ml[:,3]
plt.hist(beta,density=True,color='grey',label = r"$\mathbf{e}_3'\hat{\mathbf{\beta}}^{\sim}_n$")
x = np.linspace(min(beta),max(beta),10000)
plt.plot(x,normal(x,beta0[3],v*sigma*sigma0/np.sqrt(p)),'k-',label =  r"$\mathcal{N}(\mathbf{e}_3'\mathbf{\beta}_0,v^2/p^2)$")
plt.axvline(x= beta0[3],color = 'black', linestyle = '-.',label = r"$\mathbf{e}_3'\mathbf{\beta}_0$")
plt.legend()
plt.savefig('beta3.zeta'+'{:.2f}'.format(p/n)+'.png')

#location parameter
plt.figure()
plt.title(r"$n=$"+str(n)+r" $\zeta=$"+'{:.2f}'.format(p/n)+r" $\theta_0=$"+str(beta0[0]))
plt.hist(Phi_ml,density=True,color = 'grey',label =r'$\hat{\phi}_n$')
plt.axvline(x= phi0,color = 'black', linestyle = '-.')
plt.xlabel(r'$\hat{\phi}_n$')
plt.legend()
plt.savefig('phi.zeta'+'{:.2f}'.format(p/n)+'.png')


#scale parameter
plt.figure()
plt.title(r"$n=$"+str(n)+r" $\zeta=$"+'{:.2f}'.format(p/n)+r" $\theta_0=$"+str(beta0[0]))
plt.hist(Sigma_ml,density=True,color = 'lightgrey',label =r'$\hat{\sigma}_n$')
plt.hist(Sigma_ml/sigma,density=True,color = 'grey',label =r'$\hat{\sigma}^{\sim}_n$')
plt.axvline(x= sigma0,color = 'black', linestyle = '-.')
plt.legend()
plt.savefig('sigma.zeta'+'{:.2f}'.format(p/n)+'.png')

plt.show()
    
    
