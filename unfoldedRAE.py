from __future__ import division
import numpy as np
#import nltk
from nltk.tree import Tree

from Node import *
from numericalGradient import *
from getEmbeddings import *
from readCorpus import *

def initializeReal():
    print 'Initializing...'
    global Wc, bc, Wr, br, L, vocabulary
    L,vocabulary = getSennaEmbs()
    v,d = np.shape(L)
    Wc = np.random.rand(d, 2*d)  #construction weights
    bc = np.random.rand(d)       #construction bias
    Wr = np.random.rand(2*d,d)   #reconstruction weights
    br = np.random.rand(2*d)     #reconstruction bias

    print 'Done.', v,'words, dimensionality:',d

def initialize(d, words):
    global Wc, bc, Wr, br, L, vocabulary
    Wc = np.random.rand(d, 2*d)  #construction weights
    bc = np.random.rand(d)       #construction bias
    Wr = np.random.rand(2*d,d)   #reconstruction weights
    br = np.random.rand(2*d)     #reconstruction bias
    L = np.random.rand(len(words),d)
    vocabulary = {key: value for (key, value) in zip(words,range(len(words)))}
    return vocabulary

def initializeOnes(d, words):
    global Wc, bc, Wr, br, L, vocabulary
    Wc = np.ones([d, 2*d],np.float32)  #construction weights
    bc = np.ones(d,np.float32)       #construction bias
    Wr = np.ones([2*d,d],np.float32)   #reconstruction weights
    br = np.ones(2*d,np.float32)     #reconstruction bias
    L = np.ones([len(words),d],np.float32)
    vocabulary = {key: value for (key, value) in zip(words,range(len(words)))}
    return vocabulary



def parseSentence(sent):
    nodes = dict()
    for index in range(len(sent)):
        word = sent[index]
        if word in vocabulary: wordIndex = vocabulary[word]
        else:                  wordIndex = vocabulary['UNK']
        newLeaf = Node([], index, index, wordIndex)
        newLeaf.forwardPass(Wc,bc,Wr,br,L,True)
        nodes[index] = newLeaf
    candidates = dict()
    while len(nodes) > 1:

#        print 'new iteration, ', nodes.keys()
        # Update candidate list
        for index, node in nodes.iteritems():
#            print index, 'spans:',node.startAt(), node.endAt()
            if node.end+1 in nodes.keys():
                sibling = nodes[node.end+1]
                parent = Node([node,sibling], node.start, sibling.end)
                error = parent.reconstructionError(Wc,bc,Wr,br,L)
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

        newNode.children[0].setLeft(True)
        newNode.children[1].setLeft(False)
#        print newNode

        index = newNode.start
        nextIndex = newNode.children[1].start
#        print 'create new node spanning',index, newNode.rightChild.endAt()
        nodes[index] = newNode      # replace left child with new parent
        del nodes[nextIndex]        # delete right child
        del candidates[index]       # remove candidate belonging to left Child
        if nextIndex in candidates: # remove candidate belonging to right Child (if any)
           del candidates[nextIndex]

    nodes[0].setLeft(True)
    return nodes[0]


def trySentence(sentence):
    print sentence
    parse = parseSentence(sentence)
    print parse
    diff =  numericalGradient(parse,Wc,bc,Wr,br,L)
    print 'difference numerical/ analytical:', diff

#    gradWc,gradBc,gradWr,gradBr = parse.backprop(np.zeros(d),Wc,bc,Wr,br,L)

def epoch(trees):
    print 'start training'
    delta = np.zeros(d)
    alpha = 0.1
    grads = []
    global Wc, bc, Wr, br
    for tree in trees:
        print 'training', tree
        tree.forwardPass(Wc,bc,Wr,br,L,)
        grads.append(tree.backprop(delta, Wc,bc,Wr,br,L))
    DWc, DBc, DWr, DBr = sum(grads)

    Wc -= alpha * (DWc/len(trees))
    bc -= alpha * (DBc/len(trees))
    Wr -= alpha * (DWr/len(trees))
    br -= alpha * (DBr/len(trees))

def main():
    global d
    d = 50
    words = ['dog', 'cat', 'chases', 'the','that', 'mouse']
    initialize(d,words)
    initializeReal()
#    trySentence(['the','dog','chases', 'the','cat','that','chases','the','mouse'])
#    trySentence(['the','dog','chases', 'the','cat'])
#    trySentence(['the','dog','chases'])
#    trySentence(['the','dog'])
#    trySentence(['the'])

    trees = readC(L,vocabulary)
    epoch(trees)
    #print M1, b1

if __name__ == "__main__":
   main()