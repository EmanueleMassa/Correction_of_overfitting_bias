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
    #eta = 0.5
    while(err>1.0e-13):
        #y = eta*(x-mu*np.tanh(y0)) + (1-eta)*y0
        #err = np.sqrt(np.sum((y-y0)**2))
        #y0 = y
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


#control parameter
tau = 0.01

#variables initialization
v0 = 1.0e-1
zeta0 = 1.0e-1
sigma0 = 1.0
eta = 0.5
zeta = 0.0 
D = []

while(zeta<=0.8):
    err = 1.0
    its = 0
    while(err>1.0e-13):
        M1 = chi(0.5*np.add.outer(v0*gp,-ep/sigma0),0.5*tau)
        M0 = chi(0.5*np.add.outer(v0*gp,ep/sigma0),0.5*tau)
        t1 = np.tanh(M1)
        c1 = np.cosh(M1)
        t0 = np.tanh(M0)
        c0 = np.cosh(M0)
        zeta1 = gw@(tau/(tau + 2*(c1**2)) + tau/(tau + 2*(c0**2)))@ew
        v1 = tau*np.sqrt(gw@(t1**2 + t0**2)@ew/zeta0)
        sigma1 = -(gw@(t1-t0)@(ew*ep))
        zeta = zeta1*eta + zeta0*(1-eta)
        v   = v1*eta + v0*(1-eta)
        sigma  = sigma1*eta + sigma0*(1-eta)
        err = abs(zeta-zeta0) + abs(v-v0)+ abs(sigma-sigma0)
        v0 = v
        zeta0= zeta
        sigma0 = sigma
        its = its +1 
        #print(tau,zeta,v,sigma,err,its)
    print(tau,zeta,v,sigma,err,its)
    tau = tau + 0.01*tau
    D = D + [[tau,zeta,v,sigma,its,err]]

D = np.array(D)

plt.figure()
plt.plot(D[:,0],D[:,1])
plt.show()



np.savetxt('rs.log_log_aft.txt',D)


    


