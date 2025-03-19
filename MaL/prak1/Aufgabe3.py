import numpy as np
import matplotlib.pyplot as plt

from Aufgabe1 import bClassificationTree

class randomForestClassification:    
    def __init__(self, noOfTrees=10, max_features=None, threshold = 10**-8, xDecimals = 8, minLeafNodeSize=3, perc=1):
        self.perc = perc
        self.threshold = threshold
        self.xDecimals = xDecimals
        self.minLeafNodeSize = minLeafNodeSize
        self.bTree = []
        self.noOfTrees = noOfTrees
        for i in range(noOfTrees):
            tempTree = bClassificationTree(max_features=max_features, threshold = self.threshold, xDecimals = self.xDecimals , minLeafNodeSize=self.minLeafNodeSize)
            self.bTree.append(tempTree)

    def fit(self,X,y):
        self.samples = []
        for i in range(self.noOfTrees):
            bootstrapSample = np.random.randint(X.shape[0],size=int(self.perc*X.shape[0]))
            self.samples.append(bootstrapSample)     
            bootstrapX = X[bootstrapSample,:]
            bootstrapY = y[bootstrapSample]
            self.bTree[i].fit(bootstrapX,bootstrapY)
    
    def predict(self,X):
        ypredictTemp = np.zeros( (X.shape[0],self.noOfTrees) )
        ypredict = np.zeros( X.shape[0] )
        for i in range(self.noOfTrees):
            ypredictTemp [:,i] = self.bTree[i].predict(X)
        for j in range(X.shape[0]):
            unique, counts = np.unique(ypredictTemp[j,:], return_counts=True)
            i = np.argmax(counts)
            ypredict[j] = unique[i]
        return(ypredict)
    
if __name__ == '__main__':
    import os
    os.chdir(os.path.dirname(os.path.abspath(__file__)))

    dataset = np.genfromtxt('AllData.csv',delimiter=',')
    X = dataset[:,1:]
    Y = dataset[:,0]
    
    # reproducable results for random functions
    np.random.seed(42)

    MainSet = np.arange(0,X.shape[0])
    Trainingsset = np.random.choice(X.shape[0], int(0.8*X.shape[0]), replace=False)
    Testset = np.delete(MainSet,Trainingsset)
    XTrain = X[Trainingsset,:]
    yTrain = Y[Trainingsset]
    XTest = X[Testset,:]
    yTest = Y[Testset]

    errorPerTree = np.zeros(50)
    for i in range(1, 51):
        myForest = randomForestClassification(noOfTrees=i, minLeafNodeSize=3, xDecimals=5, threshold=0.1)
        myForest.fit(XTrain, yTrain)
        yPredict = myForest.predict(XTest)
        yDiff = yPredict - yTest
        errorPerTree[i-1] = np.count_nonzero(yDiff)
    x = np.arange(1,51,1)
    fig = plt.figure(1)
    ax = fig.add_subplot(1,1,1)
    ax.plot(x,errorPerTree,'o-')
    ax.set_xlabel('Baeume')
    ax.set_ylabel('Fehler')
    ax.grid(True,linestyle='-',color='0.75')
    plt.show()