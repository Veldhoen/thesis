from __future__ import division
import numpy as np


class Parent:
   def __init__(self, left, right):
       self.leftChild = left
       self.rightChild = right
       self.leftRec = Reconstruction(left)
       self.rightRec = Reconstruction(right)
       
   def forwardPass(self,Wc,bc,Wr,br,L):
       left = self.leftChild.forwardPass(Wc,bc,Wr,br,L)
       right = self.rightChild.forwardPass(Wc,bc,Wr,br,L)
       self.z=Wc.dot(np.concatenate([left,right]))+b1   # apply weight matrix and bias
       self.a,self.ad = activate(self.representation)

       reconstruction = np.tanh(M2.dot(self.a)+b2)
       self.leftRec.forwardPass(reconstruction[0:d-1], M1,b1,M2,b2,L)
       self.rightRec.forwardPass(reconstruction[d:-1], M1,b1,M2,b2,L)

       return self.a

   def backprop(self, deltaP,Wc,bc,Wr,br,L):

       # compute the construction delta for this node
       deltaC = self.ad*deltaP*Wc

       # compute the reconstruction gradients from
       # the reconstruction of this node
       gradWrLR,gradbrLR, deltaRLR = leftRec.backprop()
       gradWrRR,gradbrRR, deltaRRR = rightRex.backprop()

       # compute the construction and reconstruction
       # gradients from the children
       gradWcLC,gradbcLC,gradWrLC,gradbrLC = self.leftChild.backprop(deltaC,Wc,bc,Wr,br,L)
       gradWcRC,gradbcRC,gradWrRC,gradbrRC = self.rightChild.backprop(deltaC,Wc,bc,Wr,br,L)

       # combine all computed gradients
       gradWr = [deltaLR,deltaRR]*self.a.T
       gradWr += gradWrLR + gradWrRR
       gradWr += gradWrLC + gradWrRC

       gradbr = [deltaLR,deltaRR]
       gradbr += gradbrLR + gradbrRR
       gradbr += gradbrLC + gradbrRC

       gradWc = gradWcLC + gradWcRC
       gradbc = gradbcLC + gradbcRC

       return gradWc, gradbc, gradWr, gradbr



class RecParent:
   def __init__(self, left, right):
       if isinstance(left, Leaf):
          self.leftChild = RecLeaf(left)
       else:
          self.leftChild = RecParent(left.leftChild, left.rightChild)
       if isinstance(right, Leaf):
          self.rightChild = RecLeaf(left)
       else:
          self.leftChild = RecParent(left.leftChild, left.rightChild)

  def forwardPass(self, representation, M2,b2):
      self.representation = representation
      self.activation = np.tanh(self.representation)
      reconstructedChildren = np.tanh(M2.dot(np.tanh(representation))+b2)        # determine reconstruction
      self.leftChild.forwardPass(reconstructedChildren[0],M2,b2)
      self.rightChild.forwardPass(reconstructedChildren[1],M2,b2)



  def backprop(self):
      delta =



class Root:
   def __init__(self, left, right):
       self.construction = Parent(left,right)
       self.reconstruction = RecParent(left,right)

   def forward(M1,b1,M2,b2):
       activation = self.construction.forwardPass(M1,b1,M2,b2)
       self.reconstruction.forwardPass(activation, M2, b2)

   def back(self):
       [gradM2, gradb2] = self.reconstruction.back()


       reconstructedLeafs = self.reconstruction.leafs
       trueLeafs =


       reconstructionError = self.reconstruction.reconstructionError(M2,b2)


       self.delta = activateDer(self.representation)
       gradM1 =
       gradb1 =
       gradM2 =
       gradb2 =



class Reconstruction:
   def __init__(self,tree):
       if instanceOf(tree, Leaf):
          True
          #bleh
       else:
          self.leftChild = Reconstruction(tree.leftChild)
          self.rightChild = Reconstruction(tree.rightChild)



#class Node:
#   def startAt(self):
#       return self.start
#   def endAt(self):
#       return self.end

class recLeaf:
   def __init__(self,original):
       self.original = original

   def error(self):
       return norm

   def forwardPass(self,reconstruction,M2,b2):
       self.reconstruction = reconstruction

class recParent:
   def __init__(self,left,right):
       leftPart = left.offspring
       rightPart = right.offspring
       self.original = leftPart+rightPart
       if isinstance(left,Leaf):
           self.leftChild = recLeaf(leftPart)
       else:
           self.leftChild = recParent(left.leftChild,left.rightChild)
       if isinstance(right,Leaf):
           self.rightChild = recLeaf(rightPart)
       else:
           self.rightChild = recParent(right.leftChild,right.rightChild)

   def forwardPass(self,activation, M2,b2):
       reconstructedChildren = np.tanh(M2.dot(activation)+b2)        # determine reconstruction
       self.leftChild.forwardPass(reconstructedChildren[0],M2,b2)
       self.rightChild.forwardPass(reconstructedChildren[1],M2,b2)

   def backProp(self,gradM1, gradb1, gradM2, gradb2):
       reconstruction = self.leftChild.reconstruction
       delta

class Parent:
   def __init__(self,left,right):
       self.span = (left.span(0),right.span(1))
       self.leftChild = left
       self.rightChild = right
       self.offspring = left.offspring + right.offspring
       self.reconstuction = recParent(left,right)

   def forwardPass(self,M1,b1,M2,b2, L):
       left  = self.leftChild.forwardPass(M1,b1,M2,b2,L)
       right = self.rightChild.forwardPass(M1,b1,M2,b2,L)
       self.representation=M1.dot(np.concatenate([leftRep,rightRep]))+b1   # apply weight matrix and bias
       self.activation = np.tanh(self.representation)
       self.reconstruction.forwardPass(self.activation)
       return self.activation

   def backProp(self,prevDelta, gradM1, gradb1, gradM2, gradb2):

       self.reconstructionError = self.reconstruction.backProp(gradM1, gradb1, gradM2, gradb2)
#       self.totalError =

class Leaf:
   def __init__(self,start,wordIndex):
       self.range = (start, end)          *
       self.word = wordIndex

   def forwardPass(self,M1,b1,L):
       self.representation = L[wordIndex]
       self.activation = self.representation
       return self.activation

def activate(vector):
    act = np.tanh(vector)
    der = np.add(1,-1*np.multiply(act,act))
    return act, der
