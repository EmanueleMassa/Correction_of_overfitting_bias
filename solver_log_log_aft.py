import numpy as np
import matplotlib.pyplot as plt
import numpy.random as rnd


GH = np.loadtxt('GH.txt')
GL = np.loadtxt('GL.txt')

ep=  GL[:,0]#exponential points
gp=  np.sqrt(2)*GH[:,0]#gaussian points

ew = GL[:,1]/((1.0+np.exp(-ep))**2)#exponential weights
gw = GH[:,1]/np.sqrt(np.pi)#gaussian weights




def chi(x,mu):
    err = 1.0
    y0 = x/(1.0+mu)
    its = 0
    while(err>1.0e-13):
        w = np.tanh(y0)
        delta = -(y0-x + mu*w)/(1+mu*(1-w*w))
        y = y0 + delta
        err = np.sqrt(np.max((y-x + mu*np.tanh(y))**2))
        its = its + 1
        y0 = y
        if(its>10000):
            print('error=',err)
            return np.inf
    return y

def prox(x,mu):
    return x - mu*np.tanh(chi(x,mu))


def RS_solver(zeta):
    #variables initialization
    v0 = zeta
    tau0 = zeta
    sigma0 = 1.0
    eta = 0.5

    err = 1.0
    its = 0

    while(err>1.0e-13):
        M1 = chi(0.5*np.add.outer(v0*gp,-ep/sigma0),0.5*tau0)
        M0 = chi(0.5*np.add.outer(v0*gp,ep/sigma0),0.5*tau0)
        t1 = np.tanh(M1)
        c1 = np.cosh(M1)
        t0 = np.tanh(M0)
        c0 = np.cosh(M0)
        tau1 = zeta/(gw@(1.0/(tau0 + 2*(c1**2)) + 1.0/(tau0 + 2*(c0**2)))@ew)
        v1 = tau0*np.sqrt(gw@(t1**2 + t0**2)@ew/zeta)
        sigma1 = -(gw@(t1-t0)@(ew*ep))
        tau = tau1*eta + tau0*(1-eta)
        v   = v1*eta + v0*(1-eta)
        sigma  = sigma1*eta + sigma0*(1-eta)
        err = abs(tau-tau0) + abs(v-v0)+ abs(sigma-sigma0)
        v0 = v
        tau0= tau
        sigma0 = sigma
        its = its +1 
    
    return np.array([tau,v,sigma,err,int(its)])


    


