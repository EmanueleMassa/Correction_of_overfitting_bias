import numpy as np
import matplotlib.pyplot as plt
import numpy.random as rnd
import scipy.optimize as opt 
from solver_weibull import RS_solver

n = 200
p = 70
m = 500
zeta = p/n

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


#def the loss function, or minus the utility
def l(theta):
    beta = theta[:-2]
    phi = theta[-2]    
    lrho = theta[-1]
    lp = X@beta
    return (-sum(np.exp(lrho)*Y + phi + lp - np.exp(np.exp(lrho)*Y+ phi + lp))/n - lrho)

#define the matrix that contains the ML estimators 
D = np.empty((m,p+2))

for k in range(m):
    #generate the data
    X = rnd.normal(size = (n,p))
    s = rnd.random(size = n)
    lp0 = X@beta0
    Y = (np.log(-np.log(s))-phi0 - lp0)/rho0
    #fit the model
    x0 = np.zeros(p+2)
    fit = opt.minimize(l,x0,method = 'BFGS',tol = 1.0e-13)
    fit = opt.minimize(l,fit.x,method = 'BFGS',tol = 1.0e-13)
    fit = opt.minimize(l,fit.x,method = 'BFGS',tol = 1.0e-13)
    #store the resutl in the matrix D 
    D[k,:] = fit.x
    
Beta_ml = D[:,:-2]
Phi_ml = D[:,-2]
Rho_ml = np.exp(D[:,-1])

RS = RS_solver(zeta)

v = RS[0]
phi=RS[1]
rho=RS[2]

def normal(x,mu,sigma):
    w = (x-mu)
    return np.exp(-0.5*((w/sigma)**2+np.log(2*np.pi) +2.0*np.log(sigma) ))


plt.figure()
plt.title(r"$n=$"+str(n)+r" $\zeta=$"+'{:.2f}'.format(p/n)+r" $\theta_0=$"+str(beta0[0]))
plt.hist(Beta_ml[:,0],density=True,color='grey')
x = np.linspace(min(Beta_ml[:,0]),max(Beta_ml[:,0]),10000)
plt.plot(x,normal(x,rho*beta0[0],v/np.sqrt(p)))
plt.xlabel(r"$\hat{\mathbf{\beta}}_n'\mathbf{e}_1$")
plt.savefig('beta1.png')

plt.figure()
plt.title(r"$n=$"+str(n)+r" $\zeta=$"+'{:.2f}'.format(p/n)+r" $\theta_0=$"+str(beta0[0]))
plt.hist(Beta_ml[:,p-1],density=True, color='grey')
x = np.linspace(min(Beta_ml[:,2]),max(Beta_ml[:,2]),10000)
plt.plot(x,normal(x,rho*beta0[2],v/np.sqrt(p)))
plt.xlabel(r"$\hat{\mathbf{\beta}}_n'\mathbf{e}_3$")
plt.savefig('betap.png')

plt.figure()
plt.title(r"$n=$"+str(n)+r" $\zeta=$"+'{:.2f}'.format(p/n)+r" $\theta_0=$"+str(beta0[0]))
plt.hist(Phi_ml,density=True,color = 'lightgrey',label = r"$\hat{\phi}_n$")
plt.hist((Phi_ml-phi)/rho,density=True,color = 'grey',label = r"$\hat{\phi}^{\sim}_n$")
plt.axvline(x= phi0,color = 'black', linestyle = '-.')
plt.legend()
plt.savefig('phi.png')

plt.figure()
plt.title(r"$n=$"+str(n)+r" $\zeta=$"+'{:.2f}'.format(p/n)+r" $\theta_0=$"+str(beta0[0]))
plt.hist(Rho_ml,density=True,color = 'lightgrey',label = r"$\hat{\rho}_n$")
plt.hist(Rho_ml/rho,density=True,color = 'grey',label = r"$\hat{\rho}^{\sim}_n$")
plt.axvline(x= rho0,color = 'black', linestyle = '-.')
plt.legend()
plt.savefig('rho.png')
plt.show()

    
    
