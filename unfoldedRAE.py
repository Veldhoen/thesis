from __future__ import division

from nltk import tree
import numpy as np
#from numericalGradient2 import *
'''
there are two kinds of nodes. Both node types have (0 or 2) children. There is no explicit link to a parent.
 - class Node: node in the RNN
   each parent node has a corresponding reconstruction, with the same activation.
 - class Reconstruction: node in the reconstruction of a tree
   if the original node is a parent, the reconstruction is also a parent.
   The children are reconstructions that again copy the structure of their 'original'

A Node keeps track of its span in the sentence (start and end).
This is only useful for the parsing bit, and it may be better to place this information outside of the class.

When activated, each Node and Reconstruction has values:
- z, its representation
- a, its activation f(z)
- ad, the derivative of its activation  f'(z)
 the function activate(vector) returns a and ad (either tanh or sigmoid, there is also a function activateIdentity(vector))

A forward pass can be called for the root, with the weights and biases Wc, bc, Wr, br and word matrix L.
The root's offspring is recursively activated, until the represenation of the root is computed.
The activation of any parent node causes its reconstruction to activate its 'offspring' as well.

A backward pass can be called for the root, with a zero delta vector of length d.
The delta of any parent node is the sum of the error backpropagated by its parent, and the error from its reconstruction.
The backward pass yields the gradients for gradWc, gradBc, gradWr, gradBr.
'''

class Node:
    def __init__(self, children, start, end, wordIndex = None):
        self.children = children
        self.start = start
        self.end = end
        self.wordIndex = wordIndex
        self.word = None # the field word is for pretty printing
        if len(self.children) > 0:
            self.reconstruction = Reconstruction(self)

    # create a network from an nltk tree object
    @classmethod
    def fromTree(self,thistree, voc,start = 0, end = 0):
        print thistree
        word = thistree.label().strip()
        if len(thistree) > 0:
            wordIndex = None
            children = [Node.fromTree(child, voc, start, end) for child in thistree]
        else:
            if word in voc: wordIndex = voc[word]
            else:           wordIndex = voc['UNK']
            children = []
        parent = Node(children, start, end, wordIndex)
        parent.setWord(word)
        return parent

    def forward(self,theta,recalculate = True):
        Wc, bc, Wr, br, L = unwrap(theta)
        if len(self.children)>0:
            if recalculate: childrena = [child.forward(theta,recalculate) for child in self.children]
            else:           childrena = [child.a for child in self.children]
            self.z = Wc.dot(np.concatenate(childrena))+bc
            self.reconstruction.forward(self.z,Wr,br)
            self.a,self.ad = activate(self.z)
        else: #leaf
            self.z = L[self.wordIndex]
            self.a, self.ad = activateIdentity(self.z)
        return self.a


    def backprop(self, delta,Wc,bc,Wr,br,L):
        if len(self.children)>0:
            # compute gradWr and gradBr and delta for this node's reconstruction
            gradWr, gradBr,recDelta = self.reconstruction.backprop(Wr,br)
            # increment delta with reconstruction delta
            delta += recDelta

            # compute gradWc and gradBc for this node's construction
            # (from this delta and the children's activation)
            gradWc = np.outer(delta, np.concatenate([child.a for child in self.children]))
            gradBc = delta

            # compute delta to backpropagate (i.e. children's deltas)
            childrenads = np.concatenate([child.ad for child in self.children])
            backpropdelta = np.split(np.multiply(np.transpose(Wc).dot(delta),childrenads),2)
            # compute gradWc,gradBc,gradWr,gradBr for children
            gradWcL,gradBcL,gradWrL,gradBrL = self.children[0].backprop(backpropdelta[0],Wc,bc,Wr,br,L)
            gradWcR,gradBcR,gradWrR,gradBrR = self.children[1].backprop(backpropdelta[1],Wc,bc,Wr,br,L)
            # add gradients from children
            gradWc += gradWcL + gradWcR
            gradBc += gradBcL + gradBcR
            gradWr += gradWrL + gradWrR
            gradBr += gradBrL + gradBrR
        else: #leaf
            gradWc = np.zeros_like(Wc)
            gradBc = np.zeros_like(bc)
            gradWr = np.zeros_like(Wr)
            gradBr = np.zeros_like(br)
        return gradWc,gradBc,gradWr,gradBr

    def originalLeafs(self):
        if len(self.children)>0: return np.concatenate([child.originalLeafs() for child in self.children])
        else:                    return self.a

    def error(self,theta,target = None):
        self.forward(theta, True)
        rootError = self.reconstructionError(theta)
        return rootError + sum([child.error(theta) for child in self.children])

    # compute reconstruction error for this node: predict leafs and see how different they are from the actual leafs
    def reconstructionError(self,theta, recalculate = True):
        if len(self.children) == 0:
           return 0
        self.forward(theta,recalculate)
        original = self.originalLeafs()
        reconstruction = self.reconstruction.predictLeafs()
        length = np.linalg.norm(original-reconstruction)
        return .5*length*length

    def setWord(self, word):
        self.word = word
    def __str__(self):
        if len(self.children) > 0:
            childrenStrings = [str(child) for child in self.children]
            return '['+childrenStrings[0]+','+childrenStrings[1]+']'
        else:
            if self.word: return self.word
            else:         return str(self.wordIndex)

class Reconstruction:
    def __init__(self,original):
        self.original = original
        self.children = [Reconstruction(child) for child in original.children]

    def forward(self,z,Wr,br):
        # determine this node's representation (z) and activation (and derivative thereof)
        self.z = z
        self.a, self.ad = activate(self.z)

        # activate successors ('children')
        if len(self.children) > 0:
            reconstructions = np.split(Wr.dot(self.a)+br,2)
            self.children[0].forward(reconstructions[0], Wr,br)
            self.children[1].forward(reconstructions[1], Wr,br)

    def backprop(self,Wr,br):
        if len(self.children) > 0:
            # call the children for backpropagation of gradients and deltas
            gradWrL,gradBrL, deltaL = self.children[0].backprop(Wr,br)
            gradWrR,gradBrR, deltaR = self.children[1].backprop(Wr,br)
            deltas = np.concatenate([deltaL, deltaR])

            # compute gradients
            gradWr = np.outer(deltas,self.a) + gradWrL + gradWrR
            gradBr = deltas + gradBrL + gradBrR

            # compute this node's delta
            delta = np.multiply(np.transpose(Wr).dot(deltas),self.ad)
        else: #leaf
            # gradients Wr and br: zeros, there is no reconstruction in this node
            gradWr = np.zeros_like(Wr)
            gradBr = np.zeros_like(br)
            # determine delta: difference with original node*activation derivative
            # TODO: scale original to appropriate range
            delta = np.multiply(-(self.original.a-self.a),self.ad)
        return gradWr, gradBr, delta

    def predictLeafs(self):
        if len(self.children) > 0 : return np.concatenate([child.predictLeafs() for child in self.children])
        else:                       return self.a

    def __str__(self):
        if len(self.children) > 0:
            childrenStrings = [str(child) for child in self.children]
            return '['+childrenStrings[0]+','+childrenStrings[1]+']'
        else: return str(self.original)+'\''

def activateIdentity(vector):
    return vector, np.ones_like(vector)

## sigmoid:
# def activate(vector):
#     act =  1/(1+np.exp(-1*vector))
#     der = np.multiply(act,1-act)
#     return act, der

#tanh:
def activate(vector):
    act = np.tanh(vector)
    der = 1- np.multiply(act,act)
    return act, der

def wrap(parameters):
    return np.concatenate([np.reshape(par, -1) for par in parameters])

def unwrap(theta):
    # Retrieve Wc, bc, Wr, br from flat theta
    left = 0
    right = left + d*2*d
    Wc = np.reshape(theta[left:right],(d,2*d))
    left = right
    right = left + d
    bc = theta[left:right]
    left = right
    right = left + 2*d*d
    Wr = np.reshape(theta[left:right],(2*d,d))
    left = right
    right = left + 2*d
    br = theta[left:right]
    left = right
    L = np.reshape(theta[left:],((len(theta)-left)//d,d))
    return Wc, bc, Wr, br,L

def compare(grad,numgrad):
    if np.array_equal(numgrad,grad):
       print 'numerical and analytical gradients are equal.'
       return None
    print 'Difference numerical and analytical gradients:', np.linalg.norm(numgrad-grad)/np.linalg.norm(numgrad+grad)

    names = ['Wc','bc', 'Wr','br','L']
    pairedGrads = zip(unwrap(numgrad),unwrap(grad))

    for i in range(len(names)):
        a,b = pairedGrads[i]
        a = np.reshape(a,-1)
        b = np.reshape(b,-1)
        print 'Difference '+names[i]+' :', np.linalg.norm(a-b)/np.linalg.norm(a+b)

def main():
    global d
    d = 3
    words = ['dog', 'cat', 'chases', 'the','that', 'mouse']
    Wc = np.random.rand(d, 2*d)*2-1  #composition weights
    bc = np.random.rand(d)*2-1       #composition bias
    Wr = np.random.rand(2*d,d)*2-1   #reconstruction weights
    br = np.random.rand(2*d)*2-1     #reconstruction bias
    L = np.random.rand(len(words),d)*2-1
    vocabulary = {key: value for (key, value) in zip(words,range(len(words)))}
    theta = wrap((Wc,bc,Wr,br,L))

#     sentence = "(the)"
#     sentence = "((the) (dog))"
    sentence = "(((the) (dog)) (chases) )"
#     sentence = "(((the) (dog)) (chases ((the)(cat))))"
#     sentence = "(((the) (dog)) (chases ((the)(cat)((that)((chases)((the) (mous)))))))"

    thistree = tree.Tree.fromstring(sentence)
    network = Node.fromTree(thistree,vocabulary)
    network.forward(theta, True)
    grad = wrap(network.backprop(np.zeros(d),Wc,bc,Wr,br,L)+(np.zeros_like(L),))
#    numgrad = numericalGradient(network, theta, None)
#    compare(grad,numgrad)

main()