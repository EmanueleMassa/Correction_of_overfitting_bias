import numpy as np
import matplotlib.pyplot as plt
import numpy.random as rnd
import scipy.optimize as opt 
from solver_zeta_phi_beta import RS_solver
from ideal_solver import RS_ideal_solver
n = 400
p = 100
m = 200
zeta = p/n
phi0  = -0.6
theta0 = 0.8
#true parameters
beta0 = np.zeros(p+1)
beta0[0] = theta0
beta0[-1] = phi0


#generate the data
X = rnd.normal(size = (n,p+1))
X[:,-1] = np.ones(n)
T = np.array(rnd.random(n)<0.5*(1.0-np.tanh(X@beta0)))
T = 2*T.astype(int)-np.ones(n)


#def the loss function, or minus the utility
def l(theta):
    lp = X@theta
    return (-sum(np.log(1.0-T*np.tanh(lp)))/n)
#define the matrix that contains the ML estimators 
D = np.empty((m,p+1))
R = np.empty((m,7))
Theta = np.empty(m)
for k in range(m):
    #generate the data
    X = rnd.normal(size = (n,p+1))
    X[:,-1] = np.ones(n)
    lp0 = X@beta0
    T = np.array(rnd.random(n)<0.5*(1.0-np.tanh(X@beta0)))
    T = 2*T.astype(int)-np.ones(n)
    #fit the model
    x0 = np.zeros(p+1)
    fit = opt.minimize(l,x0,method = 'BFGS',tol = 1.0e-13)
    fit = opt.minimize(l,fit.x,method = 'BFGS',tol = 1.0e-13)
    fit = opt.minimize(l,fit.x,method = 'BFGS',tol = 1.0e-13)
    #store the resutl in the matrix D 
    D[k,:] = fit.x
    beta_ml = fit.x[:-1]
    phi_ml = fit.x[-1]
    y_ml = np.tanh(X@fit.x)
    #H = (X.transpose()@np.diag(1.0-y_ml**2)@X)/n 
    #H_inv = np.linalg.inv(H[:-1,:-1])
    #f = np.diag(X[:,:-1]@H_inv@X[:,:-1].transpose())*(T-y_ml)/n
    #xi = X[:,:-1]@beta_ml - f/(1.0-f)
    #theta = np.sqrt(sum(xi*xi)/n - (sum(xi)/n)**2)
    #print(theta,np.sqrt((X[:,:-1]@beta_ml)@(X[:,:-1]@beta_ml)))
    theta = sum((X[:,:-1]@beta_ml)**2)/n 
    Theta[k] = theta
    R[k,:] = RS_solver(zeta,theta,phi_ml)
    print(theta0,R[k,3],phi0,R[k,4])
Beta_ml = D[:,:-1]
Phi_ml = D[:,-1]

kappa_rs = R[:,2]
v_rs = R[:,1]
Theta_rs = R[:,3]
Phi_rs = R[:,4]

RS = RS_ideal_solver(zeta,theta0,phi0)
w = RS[3]
v = RS[2]

def normal(x,mu,sigma):
    w = (x-mu)
    return np.exp(-0.5*((w/sigma)**2+np.log(2*np.pi) +2.0*np.log(sigma) ))


#plt.figure()
#plt.title(r'$n=$'+str(n)+r'$p=$'+str(p)+r',  $\theta_0 =$'+str(theta0) + r"$\phi_0=$"+str(phi0))
#plt.hist(Beta_ml[:,0],density=True,color='grey')
#x = np.linspace(min(Beta_ml[:,0]),max(Beta_ml[:,0]),10000)
#plt.plot(x,normal(x,w,v/np.sqrt(p)))
#plt.xlabel(r"$\mathbf{e}_1'\hat{\mathbf{\beta}}_n$")
#plt.savefig('beta1.zeta'+'{:.2f}'.format(p/n)+'.png')

#plt.figure()
#plt.title(r'$n=$'+str(n)+r'$p=$'+str(p)+r',  $\theta_0 =$'+str(theta0) + r"$\phi_0=$"+str(phi0))
#plt.hist(Beta_ml[:,2],density=True, color='grey')
#x = np.linspace(min(Beta_ml[:,2]),max(Beta_ml[:,2]),10000)
#plt.plot(x,normal(x,0.0,v/np.sqrt(p)))
#plt.xlabel(r"$\mathbf{e}_3'\hat{\mathbf{\beta}}_n$")
#plt.savefig('beta3.zeta'+'{:.2f}'.format(p/n)+'.png')

plt.figure()
plt.title(r'$n=$'+str(n)+r'$p=$'+str(p)+r',  $\theta_0 =$'+str(theta0) + r"$\phi_0=$"+str(phi0))
x = np.linspace(min(Beta_ml[:,0]),max(Beta_ml[:,0]),10000)
plt.hist(Beta_ml[:,0]/kappa_rs,density=True,color='grey',label = r"$\mathbf{e}_1'\hat{\mathbf{\beta}}^{\sim}_n$")
plt.hist(Beta_ml[:,0],density=True,color='lightgrey',label = r"$\mathbf{e}_1'\hat{\mathbf{\beta}}_n$")
plt.plot(x,normal(x,beta0[0],v*theta0/(w*np.sqrt(p))))
plt.legend()
plt.savefig('beta1_sim.zeta'+'{:.2f}'.format(p/n)+'.png')

plt.figure()
plt.title(r'$n=$'+str(n)+r'$p=$'+str(p)+r',  $\theta_0 =$'+str(theta0) + r"$\phi_0=$"+str(phi0))
x = np.linspace(min(Beta_ml[:,2]),max(Beta_ml[:,2]),10000)
plt.hist(Beta_ml[:,2]/kappa_rs,density=True,color='grey',label = r"$\mathbf{e}_3'\hat{\mathbf{\beta}}^{\sim}_n$")
plt.hist(Beta_ml[:,2],density=True,color='lightgrey',label = r"$\mathbf{e}_3'\hat{\mathbf{\beta}}_n$")
plt.plot(x,normal(x,beta0[2],v*theta0/(w*np.sqrt(p))))
plt.legend()
plt.savefig('beta3_sim.zeta'+'{:.2f}'.format(p/n)+'.png')

#plt.figure()
#plt.title(r'$n=$'+str(n)+r'$p=$'+str(p)+r',  $\theta_0 =$'+str(theta0) + r"$\phi_0=$"+str(phi0))
#plt.hist(Beta_ml[:,0]/v_rs,density=True,color='grey',label = r"$\mathbf{e}_1'\hat{\mathbf{\beta}}^{var}_n$")
#plt.hist(Beta_ml[:,0],density=True,color='lightgrey',label = r"$\mathbf{e}_1'\hat{\mathbf{\beta}}_n$")
#plt.legend()
#plt.savefig('beta1_var.zeta'+'{:.2f}'.format(p/n)+'.png')

plt.figure()
plt.title(r'$n=$'+str(n)+r'$p=$'+str(p)+r',  $\theta_0 =$'+str(theta0) + r"$\phi_0=$"+str(phi0))
plt.hist(Phi_rs,density=True,color = 'grey',label = r"$\hat{\phi}^{\sim}_n$")
plt.hist(Phi_ml,density=True,color = 'lightgrey', label = r"$\hat{\phi}_n$")
plt.axvline(x= phi0,color = 'black', linestyle = '-.')
plt.legend()
plt.savefig('phi.zeta'+'{:.2f}'.format(p/n)+'.png')

plt.figure()
plt.title(r'$n=$'+str(n)+r'$p=$'+str(p)+r',  $\theta_0 =$'+str(theta0) + r"$\phi_0=$"+str(phi0))
plt.hist(Theta,density=True,color = 'lightgrey',label = r"$\hat{\theta}_n$")
plt.hist(Theta_rs,density=True,color = 'grey',label = r"$\hat{\theta}^{\sim}_n$")
plt.axvline(x= theta0,color = 'black', linestyle = '-.')
plt.legend()
plt.savefig('theta.zeta'+'{:.2f}'.format(p/n)+'.png')

plt.show()

    
    
