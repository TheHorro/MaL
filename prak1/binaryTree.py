import numpy as np

class tree:
    def __init__(self, varNo, value, operator):
        self.rootNode = treeNode(0, value, varNo=varNo, operator=operator)
        self.nodes = []
        self.nodes.append(self.rootNode)
        self.leafNodes = []
        self.leafNodes.append(0)

    def addNode(self, Parent, branch, value, operator='<', varNo=0):
        node = treeNode(len(self.nodes),value,Parent=Parent,operator=operator,varNo=varNo)
        self.leafNodes.append(node.number)
        self.nodes.append(node)

        parent = self.nodes[Parent]
        if branch is True:
            parent.leftTrue = node
        else:
            parent.rightFalse = node

        # Falls der Knoten ein Child hat, ist er kein Blatt mehr -> aus leafNodes löschen
        if parent.leftTrue is not None and parent.rightFalse is not None:
            toDelete = self.leafNodes.index(parent.number)
            del self.leafNodes[toDelete]
        return(node.number)

    def trace(self, x):
        traceRoute = self.rootNode.trace(x)[0]
        return traceRoute

    def eval(self, x):
        traceRoute = self.trace(x)
        y = np.zeros(len(traceRoute))
        for i in range(len(y)):
            y[i] = self.nodes[traceRoute[i][-1]]()            
        return(y)
        
    def weightedPathLength(self, X):
        traceroute = self.trace(X)
        sum = 0
        for i in range(len(traceroute)):
            sum = sum + len(traceroute[i]) -1
        return(sum)
        
    def numberOfLeafs(self):
        return(len(self.leafNodes))

    def print(self, maxlevels=-1):
        ongoingstring = "\\node {"+self.rootNode.conditionString()+" }\n" 
        if self.rootNode.leftTrue is not None:
            ongoingstring = self.rootNode.leftTrue.addMyString(ongoingstring, maxlevels, '  ')
        if self.rootNode.rightFalse is not None: 
            ongoingstring = self.rootNode.rightFalse.addMyString(ongoingstring, maxlevels, '  ')
        ongoingstring = ongoingstring + " ;"
        return(ongoingstring)

class treeNode:
    def __init__(self, number, value, Parent=None, operator='<', varNo=0):
        self.number     = number
        self.parent     = Parent    # Parent-Node
        self.leftTrue   = None
        self.rightFalse = None
        self.value      = value
        self.varNo      = varNo
        self.operator   = operator

    # Value speichert Bedingung für Knoten oder Ausgabewert für Leafs
    def __call__(self):
        return self.value

    # Ist dieser Knoten ein Leaf?
    def leafNode(self) -> bool:
        if self.leftTrue is not None and self.rightFalse is not None:
            return False
        else:
            return True

    # Auswertung der Bedingung des Knotens
    def evalCondition(self, x) -> bool:
        if self.operator == '=':
            cond = x[:, self.varNo] == self.value
        elif self.operator == '<':
            cond = x[:, self.varNo] < self.value
        else: # case >
            cond = x[:, self.varNo] > self.value
        return cond

    # Rekursive Funktion
    # Gibt alle Möglichen Kombinationen an Entscheidungen mit dem jeweiligen Ergebnis zurück
    def trace(self, x, index=None, traceRoute=None):
        if index is None:
            index = np.arange(len(x))
        if traceRoute is None:
            traceRoute = [[] for x in range(len(x))]

        for k in index: 
            traceRoute[k].append(self.number)

        if self.leafNode():
            return (traceRoute, index)

        cond = self.evalCondition(x[index])
        trueIndex  = index[cond]
        falseIndex = index[~cond]

        if self.leftTrue is not None and trueIndex.size != 0:
            traceRoute = self.leftTrue.trace(x, trueIndex, traceRoute)[0]
        if self.rightFalse is not None and falseIndex.size != 0:
            traceRoute =  self.rightFalse.trace(x, falseIndex, traceRoute)[0]
        return (traceRoute, index)

    def conditionString(self):
        if not self.leafNode():
            mystring = "$\\tiny %d \\mathrel{||} x[%d] %s %.2f$" % (self.number, self.varNo, self.operator, self.value)
        else:
            mystring = "$\\tiny %d \\mathrel{||} %.2f$" % (self.number, self.value)
        return(mystring)

    def addMyString(self, ongoingstring, levelsleft=-1, indent=''):
        if levelsleft == 0:
            return ongoingstring
        if not self.leafNode():
            ongoingstring = ongoingstring + indent + "child { node {"+self.conditionString()+" }\n"
        else:
            ongoingstring = ongoingstring + indent + "child { node[fill=gray!30] {"+self.conditionString()+" }\n"
        if self.leftTrue is not None:
            ongoingstring = self.leftTrue.addMyString(ongoingstring, levelsleft-1, indent + '  ')
        if self.rightFalse is not None: 
            ongoingstring = self.rightFalse.addMyString(ongoingstring, levelsleft-1, indent + '  ')
        ongoingstring = ongoingstring + indent + "}\n"
            
        return(ongoingstring)
        


if __name__ == '__main__':
    np.random.seed(3)

    bicycleTree = tree(0,1,'=')
    No = bicycleTree.addNode(0,False,1,varNo=1,operator='=')
    bicycleTree.addNode(No,False,0)
    bicycleTree.addNode(No,True,1)
    No = bicycleTree.addNode(0,True,1,varNo=2,operator='=')
    bicycleTree.addNode(No,True,0)
    No = bicycleTree.addNode(No,False,1,varNo=3,operator='=')
    bicycleTree.addNode(No,True,0)
    bicycleTree.addNode(No,False,1)
    import time
    x = np.array([True,False,False,False]).reshape(1,4)
    y = bicycleTree.eval(x)
    traceRoute = bicycleTree.trace(x)
    print(traceRoute)
    print(y)
    x = np.random.randint(2, size=(1000000,4)) 
    # t1 = time.clock()
    y = bicycleTree.eval(x)
    # t2 = time.clock()
    # print(t2-t1)
    traceRoute = bicycleTree.trace(x)
