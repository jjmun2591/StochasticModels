from mpl_toolkits.mplot3d import axes3d
from matplotlib import cm
import matplotlib.pyplot as plt
import numpy as np

class n_graph():
    def __init__(self,X,Y,Z,V):

        # Grab some test data.
        self.X = X
        self.Y = Y
        self.Z = Z
        self.V = V

    def plot(self):
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        cmhot = plt.get_cmap("hot")
        surf = ax.scatter(self.X, self.Y, self.Z,c=self.V)

        plt.show()