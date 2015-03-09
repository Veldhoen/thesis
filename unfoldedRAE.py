from __future__ import division
import numpy as np
#import nltk
from nltk.tree import Tree
import warnings


from Node import *
from numericalGradient import *
from getEmbeddings import *
from readCorpus import *

def initializeReal():
    print 'Initializing...'
    global Wc, bc, Wr, br, L, vocabulary
    L,vocabulary = getSennaEmbs()
    v,d = np.shape(L)
    # todo: normal distribution around 0 with 0.01 stdd
    Wc = np.random.rand(d, 2*d)  #construction weights
    bc = np.random.rand(d)       #construction bias
    Wr = np.random.rand(2*d,d)   #reconstruction weights
    br = np.random.rand(2*d)     #reconstruction bias

    print 'Done.', v,'words, dimensionality:',d

def initialize(d, words):
    global Wc, bc, Wr, br, L, vocabulary
    Wc = np.random.rand(d, 2*d)*2-1  #construction weights
    bc = np.random.rand(d)*2-1       #construction bias
    Wr = np.random.rand(2*d,d)*2-1   #reconstruction weights
    br = np.random.rand(2*d)*2-1     #reconstruction bias
    L = np.random.rand(len(words),d)*2-1
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

#        newNode.children[0].setLeft(True)
#        newNode.children[1].setLeft(False)
#        print newNode

        index = newNode.start
        nextIndex = newNode.children[1].start
#        print 'create new node spanning',index, newNode.rightChild.endAt()
        nodes[index] = newNode      # replace left child with new parent
        del nodes[nextIndex]        # delete right child
        del candidates[index]       # remove candidate belonging to left Child
        if nextIndex in candidates: # remove candidate belonging to right Child (if any)
           del candidates[nextIndex]

    return nodes[0]


def trySentence(sentence):
    print sentence
    parse = parseSentence(sentence)
    print parse
    diff =  numericalGradient(parse,Wc,bc,Wr,br,L)
    print 'difference numerical/ analytical:', diff

#    gradWc,gradBc,gradWr,gradBr = parse.backprop(np.zeros(d),Wc,bc,Wr,br,L)

def epoch(trees):
    warnings.filterwarnings("error")

    print '\t Start training'
    global Wc, bc, Wr, br
    delta = np.zeros(d)
    alpha = 0.1
    DWc = np.zeros_like(Wc)
    DBc = np.zeros_like(bc)
    DWr = np.zeros_like(Wr)
    DBr = np.zeros_like(br)


    for tree in trees:
#        print 'training', tree
        tree.forwardPass(Wc,bc,Wr,br,L,)
        try:
            grWc, grBc, grWr, grBr = tree.backprop(delta, Wc,bc,Wr,br,L)
            DWc += grWc/len(trees)
            DBc += grBc/len(trees)
            DWr += grWr/len(trees)
            DBr += grBr/len(trees)
        except: 
            grWc, grBc, grWr, grBr = tree.backprop(delta, Wc,bc,Wr,br,L, True)
            break

    print '\t Update parameters'
    Wc -= alpha * (DWc)
    bc -= alpha * (DBc)
    Wr -= alpha * (DWr)
    br -= alpha * (DBr)

def main():
    global d

    d = 3
    words = ['dog', 'cat', 'chases', 'the','that', 'mouse']
    initialize(d,words)

    print  'Wc:', Wc
    print  'bc:', bc
    print  'Wr:', Wr
    print  'br:', br
    print  'L:', L
    print  'voc:', vocabulary
#
#    trySentence(['the','dog','chases', 'the','cat','that','chases','the','mouse'])
#    trySentence(['the','dog','chases', 'the','cat'])
    trySentence(['the','dog','chases'])
#    trySentence(['the','dog'])
#     trySentence(['the'])


#     d=50
#     initializeReal()
#     trees = readC(vocabulary)
#
#     for example in trees:
#         print example.toString(vocabulary)
#         example.forwardPass(Wc,bc,Wr,br,L)
#         diff = numericalGradient(example,Wc,bc,Wr,br,L)
#         print diff
#         break



#     [tree.forwardPass(Wc,bc,Wr,br,L) for tree in trees]
#     error = sum([tree.reconstructionError(Wc,bc,Wr,br,L) for tree in trees])
#     print 'Error:', error
#
#     for i in range(10):
#         print 'epoch', i
#         epoch(trees)
#         error = sum([tree.reconstructionError(Wc,bc,Wr,br,L) for tree in trees])
#         print 'Error:', error



    #print M1, b1

if __name__ == "__main__":
   main()