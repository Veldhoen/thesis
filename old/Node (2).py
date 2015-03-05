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
        print 'reconstruction backprop', str(self)
        if len(self.children) == 0:
            # gradients Wr and br: zeros, there is no reconstruction in this node
            gradWr = np.zeros_like(Wr)
            gradBr = np.zeros_like(br)
            # determine delta: difference with original node
            self.delta = -(self.original.a-self.a)*(self.ad)
        else:
            # initialize gradients Wr and br from successors
            grads = [child.backprop(Wr,br) for child in self.children]
            gradWr,gradBr = [sum(a,b) for (a,b) in zip(grads[0],grads[1])]
            # determine successors' deltas
            # compute this node's delta
            deltas = np.concatenate([child.delta for child in self.children])
            self.delta = np.multiply(np.transpose(Wr).dot(deltas),self.ad)

            # increment gradWr and gradBr using this node's activation and its successors' deltas
            d = np.transpose(np.array([deltas]))
            a = np.array([self.a])
            gradWr += d.dot(a)
            gradBr += deltas
#        print gradWr, gradBr
#        print np.shape(self.delta)
        return gradWr, gradBr

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

    def backprop(self, backpropagatedDelta,Wc,bc,Wr,br,L):
        print 'composition backprop', str(self)
        # compute delta from backpropagated delta and Wc
        interm = np.split(np.transpose(Wc).dot(backpropagatedDelta),2)
        if self.isLeft: self.delta = np.multiply(interm[0], self.ad)
        else:           self.delta = np.multiply(interm[1], self.ad)

        # If this node is a parent node...
        if len(self.children)>0:
            # compute gradWr and gradBr from reconstruction
            gradWrP, gradBrP = self.reconstruction.backprop(Wr,br)
            # increment delta with reconstruction delta
            self.delta += self.reconstruction.delta

            # compute gradWc,gradBc,gradWr,gradBr from children
            grads = [child.backprop(self.delta,Wc,bc,Wr,br,L) for child in self.children]
            gradWc,gradBc,gradWr,gradBr = [sum(a,b) for (a,b) in zip(grads[0],grads[1])]
            # add gradients from reconstruction
            gradWr += gradWrP
            gradBr += gradBrP
        else:
            gradWc = np.zeros_like(Wc)
            gradBc = np.zeros_like(bc)
            gradWr = np.zeros_like(Wr)
            gradBr = np.zeros_like(br)

        # update gradients Wc and bc for this node
        d = np.transpose(np.array([backpropagatedDelta]))
        a = np.array([self.a])
        addition = d.dot(a)
        if self.isLeft: addition = np.concatenate([addition,np.zeros_like(addition)],axis = 1)
        else:           addition = np.concatenate([np.zeros_like(addition), addition],axis = 1)
        gradWc += addition
        gradBc += backpropagatedDelta

        # return all gradients
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
