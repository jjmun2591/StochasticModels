import numpy as np
# from n_graph import *
np.random.seed(1)

class log_reg():
    def __init__(self,trainDataX,trainDataY):
        self.trainDataX = trainDataX
        self.trainDataY = trainDataY
        self.theta = np.random.rand(trainDataX.shape[1],1)

    def sigmoid(self,x):
        return 1/(1+np.exp(-x))

    def predict(self,x,theta):
        return self.sigmoid(np.dot(x,theta))

    def cost(self,x,theta):
        m = x.shape[0]
        y = self.trainDataY
        h = self.predict(x,theta)
        J = 1/m*-np.dot(y.T,np.log(h)-np.dot((1-y).T,np.log(1-h)))
        return J

    def gradientDesc(self,alpha,epochs,testing=False):
        theta = self.theta
        x = self.trainDataX
        y = self.trainDataY
        m = self.trainDataY.shape[0]

        
        for i in range(epochs):
            yhat = self.predict(x,theta)
            #print(yhat.shape)
            delta = np.add(yhat,-y) #nx1
            ddJ = np.dot(x.T,delta) #1xn nx2
            theta = theta - alpha/m*ddJ
        return theta

    def graph(self):
        theta = self.theta.copy()
        max = 100
        min = -100
        incr = .5
        x = np.zeros(int((max-min)/incr)**2)
        y = np.zeros(int((max-min)/incr)**2)
        z = np.zeros(int((max-min)/incr)**2)
        cnt = 0
        for i in np.arange(min,max,incr):
            theta[0]=i
            for j in np.arange(min,max,incr):
                theta[1]=j
                x[cnt] = theta[0]
                y[cnt] = theta[1]
                z[cnt] = self.cost(self.trainDataX,theta)
                cnt+=1
        # n = n_graph(x,y,z,z)
        # n.plot()
        return x,y,z
if __name__=='__main__':
    x = np.array([[.1,.2,.3,.4,1,2,3,4],[.1,.2,.3,.4,1,2,3,4]]).T.reshape((8,2))
    y = np.array([0,0,0,0,1,1,1,1]).T.reshape((8,1))

    l = log_reg(x,y)

    a = l.cost(x,l.theta)
    #sb = (l.graph())
    c = l.gradientDesc(.05,1000)
    print(c)
