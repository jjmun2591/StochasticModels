import numpy as np
#from n_graph import *
np.random.seed(1)

class lin_reg():
    
    def __init__(self,trainDataX,trainDataY):
        self.theta = np.random.rand(1,trainDataX.shape[1])
        self.trainDataX = trainDataX
        self.trainDataY = trainDataY
        
    def predict(self,X,theta):
        y = np.dot(X,theta.T)
        return y
    
    def cost(self,x,theta):
        y = self.trainDataY
        yhat = self.predict(x,theta)
        m = y.shape[1]
        delta = yhat-y
        return 1/2/m*np.dot(delta.T,delta)

    def gradientDesc(self,alpha,epochs,testing=False):
        theta = self.theta
        x = self.trainDataX
        y = self.trainDataY
        m = self.trainDataY.shape[0]

        
        theta_cost = np.zeros((epochs,x.shape[1]+1))
        for i in range(epochs):
            yhat = self.predict(x,theta)
            
            delta = np.add(yhat,-y) #nx1
            ddJ = np.dot(delta.T,x) #2xn nx1
            theta = theta - alpha/m*ddJ

            if testing:

                for j in range(x.shape[1]+1):
                    if j==x.shape[1]:
                        theta_cost[i,j] = self.cost(x,theta)
                    else:
                        theta_cost[i,j] = theta[:,j]
        # if testing:
        #     ng = n_graph(theta_cost[:,0].reshape((epochs,1)),
        #     theta_cost[:,1].reshape((epochs,1)),
        #     theta_cost[:,2].reshape((epochs,1)),
        #     np.arange(0,epochs))
        #     ng.plot()
        return theta
    
    def closedSolution(self):
        x = self.trainDataX
        y = self.trainDataY
        # theta = (X.T*X)^-1*X.T*y
        return np.dot(np.linalg.inv(np.dot(x.T,x)),np.dot(x.T,y))



if __name__ == '__main__':
    input = np.random.rand(100,10)
    output = np.random.rand(100,1)
    l = lin_reg(input,output)
    g = l.gradientDesc(.0005,1000,True)
    print(g)
    print(l.closedSolution())