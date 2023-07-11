import numpy as np
import matplotlib.pyplot as plt 
import numpy.random as rnd 

GH = np.loadtxt('GH.txt') #load the list of point and weights for Hermite Gauss quadrature

gp=  np.sqrt(2)*GH[:,0]     #gaussian points
gw = GH[:,1]/np.sqrt(np.pi) #gaussian weights           

#define the implicit function $\chi$ as in the main text of the article
def chi(x,y,mu):
    err = 1.0
    z0 = (x-mu*y)/(1.0+mu)
    its = 0
    while(err>1.0e-13):
        w = np.tanh(z0)
        delta = -(z0-(x-mu*y) + mu*w)/(1+mu*(1-w*w))
        z = z0 + delta
        err = np.sqrt(np.max((z-(x-mu*y) + mu*np.tanh(z))**2))
        its = its + 1
        z0 = z
        if(its>10000):
            print('error=',err)
            return np.inf
    return z

#define the likelihood of a bernoulli model, with exponential parametrization
def f(x,y):
	return np.exp(-y*x)/(2*np.cosh(x))

#solve the RS equations, given the true data generating process (known in simulations)
def RS_ideal_solver(zeta,beta_true,phi_true):
#initialization
    tau0 = zeta
    w0 = beta_true
    v0 = zeta
    eta = 0.5
    phi0 = phi_true

    lp_true = beta_true*gp+phi_true
    f1 = f(lp_true,1)
    f0 =  f(lp_true,-1)

    err = 1.0
    its = 0
#start of the self consistent loop 
    while (err>1.0e-13):
        M = np.add.outer(v0*gp,w0*gp) + phi0 			#matrix argument for function chi 
        chi1 = chi(M,1,tau0)
        chi0 = chi(M,-1,tau0)
        w1 = (gw)@(chi1 + phi0)@(gp*gw*f1) + (gw)@(chi0 +phi0)@(gp*gw*f0) #w = kappa*theta_0
        tau1 = zeta/((gw)@(1.0/(tau0 + np.cosh(chi1)**2))@(gw*f1) + (gw)@(1.0/(tau0 + np.cosh(chi0)**2))@(gw*f0))
        v1 = (gw)@((tau0 + tau0*np.tanh(chi1))**2)@(gw*f1) + (gw)@((-tau0 +tau0*np.tanh(chi0))**2)@(gw*f0)
        v1 = np.sqrt(v1/zeta)
        phi1 = (gw)@(chi1)@(gw*f1) + (gw)@(chi0)@(gw*f0)
        w = w1*eta + w0*(1.0-eta)
        v = v1*eta + v0*(1.0-eta)
        tau = tau1*eta + tau0*(1.0-eta)
        phi = phi1*eta + phi0*(1.0-eta)
        err = abs(v-v0)+ abs(w-w0) + abs(tau-tau0) + abs(phi-phi0) 
        its = its +1
        w0 = w
        v0 = v
        tau0 = tau
        phi0 = phi
    return np.array([zeta,tau,v,w,phi,err,its])
