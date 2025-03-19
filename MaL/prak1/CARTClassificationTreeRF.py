# %%
import numpy as np
from binaryTree import tree

from typing import Union
from numpy.typing import NDArray

class bClassificationTree:    
    def _calGiniImpurity(self,y) -> float:
        unique, counts = np.unique(y, return_counts=True)
        N = counts/len(y)
        G = 1 - np.sum(N**2)
        return G

    def _bestSplit(self,X : np.ndarray,y : NDArray[np.bool_], feature) -> tuple[float, str]:
        G = 1
        bestSplit = np.inf
        XSort = np.unique(X[:,feature].round(self.xDecimals))
        XDiff = (XSort[1:] + XSort[:-1]) / 2
        for i in range(XDiff.shape[0]):
            index = X[:,feature] < XDiff[i]
            G1 = self._calGiniImpurity(y[index])
            G2 = self._calGiniImpurity(y[~index])
            GSplit = np.mean(index)*G1 + np.mean(~index)*G2
            if G > GSplit:
                G = GSplit
                bestSplit = XDiff[i]
        return bestSplit, G
    
    def _chooseFeature(self,X,y) -> tuple[float, float, int]:
        G = np.zeros(X.shape[1])
        bestSplit = np.zeros(X.shape[1])
        if self.max_features is None:
            feature = np.arange(X.shape[1])
        elif self.max_features == 'sqrt':
            feature = np.random.choice(X.shape[1],int(np.sqrt(X.shape[1])),replace=False)
        else:
            feature = np.random.choice(X.shape[1],self.max_features,replace=False)
        for i in feature:
            bestSplit[i], G[i] = self._bestSplit(X,y,i)
        smallest = np.argmin(G)
        return G[smallest], bestSplit[smallest], smallest
    
    def _ComputeValue(self,y):
        unique, counts = np.unique(y, return_counts=True)
        i = np.argmax(counts)
        return(unique[i])

    def __init__(self, max_features=None, threshold = 10**-8, xDecimals = 8, minLeafNodeSize=3):
        self.max_features = max_features
        self.bTree = None
        self.threshold = threshold
        self.xDecimals = xDecimals
        self.minLeafNodeSize = minLeafNodeSize

    def _GenTree(self,X,y,parentNode,branch) -> None:
        commonValue = self._ComputeValue(y)
        initG = self._calGiniImpurity(y)
        
        # Abbruchbedingung: 
        # threshold = Wert ab dem die Verbesserung zu aufwändig ist
        # minLeafNodeSize = Mindestanzahl an Daten, die in einem Blatt vorhanden sein müssen
        if  initG < self.threshold or X.shape[0] <= self.minLeafNodeSize:
            self.bTree.addNode(parentNode,branch,commonValue)
            return
        (G, bestSplit ,chooseA) = self._chooseFeature(X,y)
        if G > 0.98*initG:
            self.bTree.addNode(parentNode,branch,commonValue)
            return
        if parentNode == None: 
            self.bTree = tree(chooseA, bestSplit, '<')
            myNo = 0
        else: 
            myNo = self.bTree.addNode(parentNode,branch,bestSplit,operator='<',varNo=chooseA)
        
        # Aufteilung der Trainingsdaten in True & False Bereiche für das aktuelle Merkmal
        index = np.less(X[:,chooseA],bestSplit)
        XTrue  = X[index,:] 
        yTrue  = y[index]
        XFalse = X[~index,:]
        yFalse = y[~index]    
        if XTrue.shape[0] > self.minLeafNodeSize:
            self._GenTree(XTrue,yTrue,myNo,True)
        else:
            commonValue = self._ComputeValue(yTrue)
            self.bTree.addNode(myNo,True,commonValue)
        if XFalse.shape[0] > self.minLeafNodeSize:
            self._GenTree(XFalse,yFalse,myNo,False)
        else:
            commonValue = self._ComputeValue(yFalse)
            self.bTree.addNode(myNo,False,commonValue)
        return()

    def fit(self, X,y) -> None:
        self._GenTree(X,y,None,None)
    
    def predict(self, X):
        return(self.bTree.eval(X))
    
    def decision_path(self, X):
        return(self.bTree.trace(X))
        
    def weightedPathLength(self,X):
        return(self.bTree.weightedPathLength(X)) 
        
    def numberOfLeafs(self):
        return(self.bTree.numberOfLeafs())

if __name__ == '__main__':
    import os
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    from twoMoonsProblem import twoMoonsProblem

    np.random.seed(42)
    X, Y = twoMoonsProblem(SamplesPerMoon=1000)

    MainSet = np.arange(0,X.shape[0])
    # Trainingsset = 80% der Rohdaten
    Trainingsset = np.random.choice(X.shape[0], int(0.8*X.shape[0]), replace=False)
    # Testset = Restliche 20% der Rohdaten
    Testset = np.delete(MainSet,Trainingsset)

    errorFactor = 1 + 2*(np.random.rand(Trainingsset.shape[0]) - 0.5)*0.05 
    XTrain = X[Trainingsset,:]
    yTrain = Y[Trainingsset] * errorFactor 
    XTest = X[Testset,:]
    yTest = Y[Testset]
    
    myTree = bClassificationTree(xDecimals=3)
    myTree.fit(XTrain, yTrain)
    yPredict = myTree.predict(XTest)
    yDiff = np.abs(yPredict - yTest)


    import matplotlib.pyplot as plt
    fig = plt.figure(1)
    ax = fig.add_subplot(1,1,1)
    indexA = np.flatnonzero(Y>0.5)
    indexB = np.flatnonzero(Y<0.5)

    ax.scatter(X[indexA,0],X[indexA,1],color='red', marker='o')
    ax.scatter(X[indexB,0],X[indexB,1],color='black', marker='+')
    ax.set_xlabel('$x_0$')
    ax.set_ylabel('$x_1$')
    ax.set_ylim([-1,2])
    ax.set_ylim([-1,2])
    ax.set_title("Two Moons Set Training")



    fig = plt.figure(2)
    ax = fig.add_subplot(1,1,1)
    indexA = np.flatnonzero(yPredict>0.5)
    indexB = np.flatnonzero(yPredict<0.5)
    ax.scatter(XTest[indexA,0],XTest[indexA,1],color='red', marker='o')
    ax.scatter(XTest[indexB,0],XTest[indexB,1],color='black', marker='+')
    ax.set_xlabel('$x_0$')
    ax.set_ylabel('$x_1$')
    ax.set_ylim([-1,2])
    ax.set_ylim([-1,2])
    ax.set_title("Two Moons Set Testdata")

    plt.show()