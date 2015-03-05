from __future__ import division
import numpy as np

class Reconstruction:
    def __init__(self,original):
        self.original = original
        self.children = [Reconstruction(child) for child in original.children]

    def forwardPass(self,z,Wc,bc,Wr,br,L):
        # determine this node's representation and activation (and derivative thereof)
        self.z = z
        self.a, self.ad = activate(self.z)
        # activate successors ('children')
        if len(self.children) > 0:
            reconstructions = np.split(Wr.dot(self.a)+br,2)
            self.children[0].forwardPass(reconstructions[0], Wc,bc,Wr,br,L)
            self.children[1].forwardPass(reconstructions[1], Wc,bc,Wr,br,L)

    def backprop(self,Wr,br):
#       print 'reconstruction backprop', str(self)
        if len(self.children) == 0:
            # gradients Wr and br: zeros, there is no reconstruction in this node
            gradWr = np.zeros_like(Wr)
            gradBr = np.zeros_like(br)
            # determine delta: difference with original node
            delta = -(self.original.a-self.a)*(self.ad)
        else:
            # initialize gradients Wr and br from successors
            gradWrL,gradBrL, deltaL = self.children[0].backprop(Wr,br)
            gradWrR,gradBrR, deltaR = self.children[1].backprop(Wr,br)
            # compute this node's delta
            deltas = np.concatenate([deltaL, deltaR])
            delta = np.multiply(np.transpose(Wr).dot(deltas),self.ad)

            # increment gradWr and gradBr using this node's activation and its successors' deltas
            d = np.transpose(np.array([deltas]))
            a = np.array([self.a])
            gradWr = d.dot(a) + gradWrL + gradWrR
            gradBr = deltas + gradBrL + gradBrR
#        print gradWr, gradBr
#        print 'r',np.shape(gradWr), np.shape(gradBr), np.shape(delta)
        return gradWr, gradBr, delta

    def predictLeafs(self):
        if len(self.children) > 0 : return np.concatenate([child.predictLeafs() for child in self.children])
        else:                       return self.a

    def __str__(self):
        if len(self.children) > 0:
            childrenStrings = [str(child) for child in self.children]
            return '['+childrenStrings[0]+','+childrenStrings[1]+']'
        else: return str(self.original)+'\''

class Node:
    def __init__(self, children, start, end, wordIndex = None):
        self.children = children
        self.start = start
        self.end = end
        self.wordIndex = wordIndex
        if len(self.children) > 0:
            self.reconstruction = Reconstruction(self)

    def setLeft(self,whetherLeft):
        self.isLeft = whetherLeft

    def forwardPass(self,Wc,bc,Wr,br,L,recalculate = True):
        if len(self.children) == 0:
            self.z = L[self.wordIndex]
            self.a = self.z
            self.ad = self.a
#            self.ad = np.ones_like(self.a)
        else:
            if recalculate: childrenReps = [child.forwardPass(Wc,bc,Wr,br,L) for child in self.children]
            else:           childrenReps = [child.a for child in self.children]
            self.z = Wc.dot(np.concatenate(childrenReps))+bc
            self.reconstruction.forwardPass(self.z,Wc,bc,Wr,br,L)
            self.a,self.ad = activate(self.z)
        return self.a

    def backprop(self, delta,Wc,bc,Wr,br,L):
#        print 'composition backprop', str(self)
        if len(self.children)>0:
            # compute gradWr and gradBr for reconstruction
            gradWrP, gradBrP,recDelta = self.reconstruction.backprop(Wr,br)
            # increment delta with reconstruction delta
            delta += recDelta

            # compute gradWc and gradBc for this node (this delta, the children's activation)
            gradWc = np.transpose(np.array([delta])).dot(np.array([np.concatenate([child.a for child in self.children])]))
            gradBc = delta

            # compute gradWc,gradBc,gradWr,gradBr for children
            backpropdelta = np.split(np.transpose(Wc).dot(delta),2)
            gradWcL,gradBcL,gradWrL,gradBrL = self.children[0].backprop(backpropdelta[0],Wc,bc,Wr,br,L)
            gradWcR,gradBcR,gradWrR,gradBrR = self.children[1].backprop(backpropdelta[1],Wc,bc,Wr,br,L)
            # add gradients from children and reconstruction
            gradWc += gradWcL + gradWcR
            gradBc += gradBcL + gradBcR
            gradWr = gradWrP + gradWrL + gradWrR
            gradBr = gradBrP + gradBrL + gradBrR
        else:
            gradWc = np.zeros_like(Wc)
            gradBc = np.zeros_like(bc)
            gradWr = np.zeros_like(Wr)
            gradBr = np.zeros_like(br)
#        print 'c', np.shape(gradWc), np.shape(gradBc), np.shape(gradWr), np.shape(gradBr)
        return gradWc,gradBc,gradWr,gradBr

    def originalLeafs(self):
        if len(self.children)>0: return np.concatenate([child.originalLeafs() for child in self.children])
        else:                    return self.a


    def reconstructionError(self,Wc,bc,Wr,br,L, verbose = False):
        if len(self.children) == 0:
           return 0
        self.forwardPass(Wc,bc,Wr,br,L,recalculate = False)
        original = self.originalLeafs()
        reconstruction = self.reconstruction.predictLeafs()
        if verbose:
#           print 'activation:', self.a
#           print 'original:', original
#           print 'reconstruction:', reconstruction
           print 'difference:' ,original - reconstruction
        length = np.linalg.norm(reconstruction-original)
        return .5*length*length

    def __str__(self):
        if len(self.children) > 0:
            childrenStrings = [str(child) for child in self.children]
            return '['+childrenStrings[0]+','+childrenStrings[1]+']'
        else:
            return str(self.wordIndex)

#def activate(vector):
#    return vector, np.ones_like(vector)


def activate(vector):
    act = np.tanh(vector)
    der = np.add(1,-1*np.multiply(act,act))
    return act, der
