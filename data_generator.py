import numpy as np
import matplotlib.pyplot as plt
import numpy.random as rnd
import scipy.optimize as opt 
from solver_zeta import RS_solver

n = 200
p = 20
m = 500
zeta = p/n
#true parameters
beta0 = np.zeros(p)
beta0[0] = 2.0
phi0  = 0.5
sigma0 = 1.0

#generate the data
X = rnd.normal(size = (n,p))
s = rnd.random(size = n)
Y = phi0 + X@beta0 + sigma0*np.log(s/(1.0-s))


#def the loss function, or minus the utility
def l(theta):
    beta = theta[:-2]
    phi = theta[-2]    
    lsigma = theta[-1]
    lp = X@beta
    return (-sum(np.exp(-lsigma)*(-Y + phi + lp) - 2*np.log(1.0+np.exp(np.exp(-lsigma)*(-Y+ phi + lp))))/n + lsigma)

#define the matrix that contains the ML estimators 
D = np.empty((m,p+2))

for k in range(m):
    #generate the data
    X = rnd.normal(size = (n,p))
    s = rnd.random(size = n)
    Y = phi0 +  X@beta0 + sigma0*np.log(s/(1.0-s))
    #fit the model
    x0 = np.zeros(p+2)
    fit = opt.minimize(l,x0,method = 'BFGS',tol = 1.0e-13)
    fit = opt.minimize(l,fit.x,method = 'BFGS',tol = 1.0e-13)
    fit = opt.minimize(l,fit.x,method = 'BFGS',tol = 1.0e-13)
    #store the resutl in the matrix D 
    D[k,:] = fit.x
    
Beta_ml = D[:,:-2]
Phi_ml = D[:,-2]
Sigma_ml = np.exp(D[:,-1])

#K_ml = Beta_ml@beta0/(beta0@beta0)
#V_ml = np.sqrt(np.sum(Beta_ml**2,axis =1) - K_ml**2)

RS = RS_solver(zeta)

v = RS[1]
sigma=RS[2]
print(RS)

#plt.figure()
#plt.plot(beta0,Beta_ml.transpose(), 'k.', alpha = 0.1)
#plt.savefig('cloud.png')

#plt.figure()
#plt.hist(K_ml,color='grey')
#plt.axvline(x= rho)
#plt.savefig('K.png')

#plt.figure()
#plt.hist(V_ml,color='grey')
#plt.axvline(x= v)
#plt.savefig('V.png')

def normal(x,mu,sigma):
    w = (x-mu)
    return np.exp(-0.5*((w/sigma)**2+np.log(2*np.pi) +2.0*np.log(sigma) ))


plt.figure()
plt.title(r"$n=$"+str(n)+r" $\zeta=$"+'{:.2f}'.format(p/n)+r" $\theta_0=$"+str(beta0[0]))
beta =Beta_ml[:,0]
plt.hist(beta,density=True,color='grey',label = r"$\mathbf{e}_1'\hat{\mathbf{\beta}}^{\sim}_n$")
x = np.linspace(min(beta),max(beta),10000)
plt.plot(x,normal(x,beta0[0],v*sigma*sigma0/np.sqrt(p)),'k-',label = r"$\mathcal{N}(\mathbf{e}_1'\mathbf{\beta}_0,v^2/p^2)$")
plt.axvline(x= beta0[0],color = 'black', linestyle = '-.',label = r"$\mathbf{e}_1'\mathbf{\beta}_0$")
plt.legend()
plt.savefig('beta1.zeta'+'{:.2f}'.format(p/n)+'.png')

plt.figure()
plt.title(r"$n=$"+str(n)+r" $\zeta=$"+'{:.2f}'.format(p/n)+r" $\theta_0=$"+str(beta0[0]))
beta =Beta_ml[:,3]
plt.hist(beta,density=True,color='grey',label = r"$\mathbf{e}_3'\hat{\mathbf{\beta}}^{\sim}_n$")
x = np.linspace(min(beta),max(beta),10000)
plt.plot(x,normal(x,beta0[3],v*sigma*sigma0/np.sqrt(p)),'k-',label =  r"$\mathcal{N}(\mathbf{e}_3'\mathbf{\beta}_0,v^2/p^2)$")
plt.axvline(x= beta0[3],color = 'black', linestyle = '-.',label = r"$\mathbf{e}_3'\mathbf{\beta}_0$")
plt.legend()
plt.savefig('beta3.zeta'+'{:.2f}'.format(p/n)+'.png')


plt.figure()
plt.title(r"$n=$"+str(n)+r" $\zeta=$"+'{:.2f}'.format(p/n)+r" $\theta_0=$"+str(beta0[0]))
plt.hist(Phi_ml,density=True,color = 'grey',label =r'$\hat{\phi}_n$')
plt.axvline(x= phi0,color = 'black', linestyle = '-.')
plt.xlabel(r'$\hat{\phi}_n$')
plt.legend()
plt.savefig('phi.zeta'+'{:.2f}'.format(p/n)+'.png')

plt.figure()
plt.title(r"$n=$"+str(n)+r" $\zeta=$"+'{:.2f}'.format(p/n)+r" $\theta_0=$"+str(beta0[0]))
plt.hist(Sigma_ml,density=True,color = 'lightgrey',label =r'$\hat{\sigma}_n$')
plt.hist(Sigma_ml/sigma,density=True,color = 'grey',label =r'$\hat{\sigma}^{\sim}_n$')
plt.axvline(x= sigma0,color = 'black', linestyle = '-.')
plt.legend()
plt.savefig('sigma.zeta'+'{:.2f}'.format(p/n)+'.png')

plt.show()
    
    
