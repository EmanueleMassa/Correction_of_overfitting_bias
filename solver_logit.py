import numpy as np
import matplotlib.pyplot as plt 
import numpy.random as rnd 

GH = np.loadtxt('GH.txt')
gp =  np.sqrt(2)*GH[:,0]        #gaussian points
gw = GH[:,1]/np.sqrt(np.pi)     #gaussian weights
    

#compute the chi related to the proximal operator of the logit likelihood#
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
        if(its>1000000):
            print('error=',err)
            return np.inf
    return z

def f(x,y):
    return np.exp(-y*x)/(2*np.cosh(x))

def a(x):
    return (np.log(x) - np.log(1-x))

def inv(s,x):
    z = 0 
    err = 1.0
    while(err>1.0e-13):
        z = z - (g(s,z)+x)/dg(s,z)
        err = abs(g(s,z)+x)
    return z

def g(t,y):
    return  gw@np.tanh(y+t*gp)

def dg(t,y):
    return gw@(np.tanh(y+t*gp)*gp)/t
    
def RS_solver(zeta,beta,phi):
    #initialization 
    tau0 = zeta
    k0 = 1.0
    v0 = zeta
    eta = 0.5
    phi_true = 0.0 

    err = 1.0
    its = 0
    while (err>1.0e-13):
        beta_true = np.sqrt((beta*beta - v0*v0*(1-zeta))/(k0*k0)) 
        lp_true = beta_true*gp+phi_true
        f1 = f(lp_true,1)
        f0 =  f(lp_true,-1)
        M = np.add.outer(v0*gp,k0*beta_true*gp) + phi
        chi1 = chi(M,1,tau0)
        chi0 = chi(M,-1,tau0)
        k1 = ((gw)@(chi1)@(gp*gw*f1) + (gw)@(chi0)@(gp*gw*f0))/beta_true
        tau1 = zeta/((gw)@(1.0/(tau0 + np.cosh(chi1)**2))@(gw*f1) + (gw)@(1.0/(tau0 + np.cosh(chi0)**2))@(gw*f0))
        v1 = (gw)@((tau0 + tau0*np.tanh(chi1))**2)@(gw*f1) + (gw)@((-tau0 +tau0*np.tanh(chi0))**2)@(gw*f0)
        v1 = np.sqrt(v1/zeta)
        k = k1*eta + k0*(1.0-eta)
        v = v1*eta + v0*(1.0-eta)
        tau = tau1*eta + tau0*(1.0-eta)
        err = abs(v-v0)+ abs(k-k0) + abs(tau-tau0)
        its = its +1
        k0 = k
        v0 = v
        tau0 = tau
        phi_1 = inv(beta_true,-((gw)@(np.tanh(chi1))@(gw*f1) + (gw)@(np.tanh(chi0))@(gw*f0)))
        phi_true = eta*phi_1 + (1-eta)*phi_true
    return np.array([tau,v,k,beta_true,phi_true,err,its])
	
	
	
	
	
