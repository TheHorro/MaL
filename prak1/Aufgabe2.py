import numpy as np
import matplotlib.pyplot as plt 

from Aufgabe1 import bClassificationTree

def testTree(subplot:int, feature1:int, feature2:int, xLabel:str, yLabel:str) -> None:
    XTrain = train[:,(feature1,feature2)]
    yTrain = train[:,0]
    XTest = test[:,(feature1,feature2)]
    yTest = test[:,0]
    
    myTree = bClassificationTree(xDecimals=5, threshold=0.1, minLeafNodeSize=5)
    myTree.fit(XTrain, yTrain)
    yPredict = myTree.predict(XTest)
    yDiff = np.abs(yPredict - yTest)
    # print('yDiff: ', yDiff)
    
    print(f'Fehler({feature1}, {feature2}): {np.count_nonzero(yDiff)}')
    ax = fig.add_subplot(2,2,subplot)
    
    f1_min, f1_max = XTest[:, 0].min(), XTest[:, 0].max()
    f2_min, f2_max = XTest[:, 1].min(), XTest[:, 1].max()

    # Datenbereich etwas größer wählen als er eigentlich ist, damit der Plot komplett ausgefüllt ist
    XX, YY = np.mgrid[f1_min:1.1*f1_max:0.005, f2_min:1.1*f2_max:0.005]
    X = np.array([XX.ravel(), YY.ravel()]).T
    Z = myTree.predict(X)   # Jedem Punkt auf dem Canvas einen Wert zuweisen
    Z = Z.reshape(XX.shape) # vorhergesagtes in richtige Array-Form bringen
    ax.pcolormesh(XX, YY, Z, cmap=cmap_rgb,alpha=0.5)

    D1 = np.array([x for x in test[:,(0,feature1,feature2)] if x[0] == 1])
    D2 = np.array([x for x in test[:,(0,feature1,feature2)] if x[0] == 2])
    D3 = np.array([x for x in test[:,(0,feature1,feature2)] if x[0] == 3])
    ax.scatter(D1[:,1], D1[:,2], c='red')
    ax.scatter(D2[:,1], D2[:,2], c='green')
    ax.scatter(D3[:,1], D3[:,2], c='blue')

    ax.set_xlabel(xLabel)
    ax.set_ylabel(yLabel)
    print(f'Subplot {subplot} done')    


if __name__ == '__main__':
    import os
    os.chdir(os.path.dirname(os.path.abspath(__file__)))

    test = np.genfromtxt('Testset.csv', delimiter=',')
    train = np.genfromtxt('Trainingsset.csv', delimiter=',')

    from matplotlib.colors import LinearSegmentedColormap

    colors = ["#FF4040", # light red 
              "#40FF40", # light green
              "#4040FF"] # light blue
    cmap_rgb = LinearSegmentedColormap.from_list("my_cmap", colors)

    fig = plt.figure(1)
    testTree(1, 8, 13, xLabel='Non flavanoid phenols', yLabel='Proline')
    testTree(2, 7, 10, xLabel='Flavanoids', yLabel='Color Intensity')
    testTree(3, 1, 7, xLabel='Alcohol', yLabel='Flavanoids')
    testTree(4, 1, 10, xLabel='Alcohol', yLabel='Color Intensity')
    
    
    plt.show()