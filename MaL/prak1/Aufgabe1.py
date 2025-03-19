import numpy as np
from binaryTree import tree

# Typisierung der Funktionen
from typing import Union
from numpy.typing import NDArray

class bClassificationTree:    
    def _calGiniImpurity(self,y) -> float:
        unique, counts = np.unique(y, return_counts=True)
        N = counts/len(y)
        G = 1 - np.sum(N**2)
        return G

    def _bestSplit(self,X:NDArray, y:NDArray[np.bool_], feature:int) -> tuple[float, str]:
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
        G = np.inf * np.ones(X.shape[1])
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
    
    def predict(self, X) -> NDArray[np.float64]:
        return(self.bTree.eval(X))
    
    def decision_path(self, X):
        return(self.bTree.trace(X))
        
    def weightedPathLength(self,X) -> int:
        return(self.bTree.weightedPathLength(X)) 
        
    def numberOfLeafs(self) -> int:
        return(self.bTree.numberOfLeafs())

def addSubPlot(plotNr:int, feature1:int, feature2:int, xLabel:str, yLabel:str) -> None:
    ax = fig.add_subplot(2,2,plotNr)
    D1 = np.array([x for x in dataset[:,(0,feature1,feature2)] if x[0] == 1])
    D2 = np.array([x for x in dataset[:,(0,feature1,feature2)] if x[0] == 2])
    D3 = np.array([x for x in dataset[:,(0,feature1,feature2)] if x[0] == 3])
    ax.scatter(D1[:,1], D1[:,2],c='red',  s=40)
    ax.scatter(D2[:,1], D2[:,2],c='green',s=40)
    ax.scatter(D3[:,1], D3[:,2],c='blue', s=40)
    ax.set_xlabel(xLabel)
    ax.set_ylabel(yLabel)
    ax.grid(True,linestyle='-',color='0.75')    

if __name__ == '__main__':
    import os
    os.chdir(os.path.dirname(os.path.abspath(__file__)))

    test = np.genfromtxt('Testset.csv', delimiter=',')
    train = np.genfromtxt('Trainingsset.csv', delimiter=',')

    XTrain = train[:,1:]
    yTrain = train[:,0]
    XTest = test[:,1:]
    yTest = test[:,0]
    
    myTree = bClassificationTree(xDecimals=5, threshold=0.1, minLeafNodeSize=3)
    myTree.fit(XTrain, yTrain)
    yPredict = myTree.predict(XTest)
    yDiff = np.abs(yPredict - yTest)
    print('yDiff: ', yDiff)
    unique, count = np.unique(yDiff, return_counts=True)
    print('Fehler: ', count[1] + count[2])

    dataset = np.genfromtxt('AllData.csv',delimiter=',')
    import matplotlib.pyplot as plt
    fig = plt.figure(1)
    addSubPlot(1, 8, 13, xLabel='Non flavanoid phenols', yLabel='Proline')
    addSubPlot(2, 7, 10, xLabel='Flavanoids', yLabel='Color Intensity')
    addSubPlot(3, 1, 7, xLabel='Alcohol', yLabel='Flavanoids')
    addSubPlot(4, 1, 10, xLabel='Alcohol', yLabel='Color Intensity')
    plt.show()
