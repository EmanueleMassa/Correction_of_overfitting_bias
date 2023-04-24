import numpy as np
import matplotlib.pyplot as plt
import numpy.random as rnd
import math 

GH = np.loadtxt('GH.txt')
GL = np.loadtxt('GL.txt')

ep=  GL[:,0]#exponential points
gp=  np.sqrt(2)*GH[:,0]#gaussian points

ew = GL[:,1]#exponential weights
gw = GH[:,1]/np.sqrt(np.pi)#gaussian weights

gpw = gw*gp

gamma_e = 0.577215664901532860606512090082402431042159335

def Lambf(x):
    err = 1.0
    a = np.array(x<np.e,int)
    #print(a)
    A = np.ones(np.shape(a),int)-a
    y = a*x + (A)*(np.log(x*A + 5*a) - np.log(np.log(x*A +5*a)))
    its = 0
    while(err>1.0e-13):
        y = y - y*(np.log(y/x)+y)/(1.0+y)
        err = np.sqrt(np.max((+y + np.log(y/x))**2))
        its = its +1
        if(its >=1000000):
            print('error Lambf not converging')
            return 0
    return y

def inv(z,x,mu):
    err = 1.0
    y0 = 0.0
    while(err>1.0e-13):
        m = Lambf(mu*np.exp(mu+ y0 + x))
        y = y0 - (ew@(m/(1.0+m))@gw-z)/(ew@(m/((1.0+m)**3))@gw )
        m = Lambf(mu*np.exp(mu+ y + x))
        err = abs(ew@(m/(1.0+m))@gw-z)
        y0 = y
    return y0

def RS_solver(zeta):
    eta = 0.5
    #initialization
    tau0 = zeta
    v0 = zeta
    sigma0 = 1.0-zeta
    phi0 = 0.0
    err = 1.0
    its = 0
    while (err>1.0e-13):
        M = np.add.outer(np.log(ep)/sigma0,v0*gp)
        chi  = Lambf(tau0*np.exp(tau0 + phi0 + M))
        phi1 = inv(zeta,M,tau0)
        v1 = np.sqrt((ew@((tau0-chi)**2)@gw)/zeta)
        tau1 = ew@(chi)@gw
        sigma1 = gamma_e + (ew*np.log(ep))@(chi)@gw/tau0
        phi = eta*phi1 + (1-eta)*phi0
        v = eta*v1 + (1-eta)*v0
        tau = eta*tau1 + (1-eta)*tau0
        sigma = eta*sigma1 + (1-eta)*sigma0
        err = max([abs(v-v0),abs(tau-tau0),abs(phi - phi0),abs(sigma-sigma0)])
        v0 = v
        tau0 = tau
        phi0 = phi
        sigma0 = sigma
        its = its+1
    return np.array([v,phi,1/sigma,tau,its])
            

    
        
    


