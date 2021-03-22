import numpy as np
import scipy.stats
import matplotlib.pyplot as plt
np.random.seed(1)

class GBM():
    def __init__(self,s0,drift,vol,t):
        t_hor = 1/252
        self.s0 = s0
        self.drift = drift
        self.vol = vol
        self.t = t #in years
        self.curve = np.vstack([np.arange(0,int(t),t_hor),s0*np.ones((int(t/t_hor)))]).T
        
    def simulate(self):
        curve = self.curve
        s0 = self.s0
        u = self.drift
        vol = self.vol

        #s_t = s0*exp((drift-.5*vol^2)*t+N.inv(p)*vol*sqrt(t))
        p = np.array([np.random.random() for i in range(curve.shape[0])]).T
        #p = np.array([.05 for i in range(curve.shape[0])]).T
        N_inv = (scipy.stats.norm.ppf(p))
        curve[:,1] = s0*np.exp((u-.5*vol**2)*curve[:,0]+np.multiply(N_inv*vol,np.sqrt(curve[:,0])))
        
    def graph(self):
        fig = plt.figure()
        plt.plot(self.curve[:,0],self.curve[:,1])
        plt.show()

if __name__=='__main__':
    g = GBM(50,.1,.1,1)
    g.simulate()
    g.graph()