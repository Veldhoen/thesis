import numpy as np
#import nltk
from nltk.tree import Tree

from Node import *

def initialize(d, words):
    global M1, M2, b1, b2, L, vocabulary
    M1 = np.random.rand(d, 2*d)  #construction weights
    b1 = np.random.rand(d)       #construction bias
    M2 = np.random.rand(2*d,d)   #reconstruction weights
    b2 = np.random.rand(2*d)     #reconstruction bias
    L = np.random.rand(d,len(words))
    vocabulary = {key: value for (key, value) in zip(words,range(len(words)))}
    return vocabulary

def parseSentence(sent):
    nodes = dict()
    for index in range(len(sent)):
        nodes[index] = Leaf(vocabulary[sent[index]],index, M1, b1, L)
#    for index, node in nodes.iteritems():
#        print node.startAt(), node.endAt(), node.word, node.representation(vocabulary)

    candidates = dict()
    while len(nodes) > 1:

#        print 'new iteration, ', nodes.keys()
        # Update candidate list
        for index, node in nodes.iteritems():
#            print index, 'spans:',node.startAt(), node.endAt()
            if node.endAt()+1 in nodes.keys():
                parent = Parent(node, nodes[node.endAt()+1],vocabulary,M1,b1)
                error = parent.reconstructionError(M2, b2)
                candidates[index] = (parent, error)

        # Find the candidate with the least error:
        bestCandidate = None
        for index, candidate in candidates.iteritems():
            if bestCandidate is None: bestCandidate = candidate
            else:
                if candidate[1] < bestCandidate[1]:
                    bestCandidate = candidate

        # Update nodes and candidates
        newNode = bestCandidate[0]
        index = newNode.startAt()
        nextIndex = newNode.rightChild.startAt()
#        print 'create new node spanning',index, newNode.rightChild.endAt()
        nodes[index] = newNode      # replace left child with new parent
        del nodes[nextIndex]        # delete right child
        del candidates[index]       # remove candidate belonging to left Child
        if nextIndex in candidates: # remove candidate belonging to right Child (if any)
           del candidates[nextIndex]

    return nodes[0]


def main():

    d = 10
    words = ['dog', 'cat', 'chases', 'the']
    initialize(d,words)

    sentence = ['the','dog','chases', 'the','cat']
    parse = parseSentence(sentence)
    print parse
    #print M1, b1

if __name__ == "__main__":
   main()