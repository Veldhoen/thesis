from __future__ import division
import numpy as np

class Node:
   def startAt(self):
       return self.start
   def endAt(self):
       return self.end

class Parent(Node):
   def __init__(self, left, right, M1, b1):
       self.leftChild = left
       self.rightChild = right
       self.start = left.startAt()
       self.end = right.endAt()
       self.offspring = left.offspring + right.offspring
       self.setRepresentation(vocabulary,M1,b1)

   def setRepresentation(self,vocabulary, M1, b1):
       leftRep = self.leftChild.representation           # obtain children's representations
       rightRep = self.rightChild.representation
       p=M1.dot(np.concatenate([leftRep,rightRep]))+b1   # apply weight matrix and bias
       self.representation =  p/np.linalg.norm(p)        # length normalization

   def __str__(self):
       return '['+str(self.leftChild)+' '+str(self.rightChild)+']'

   def reconstructionError(self, M2, b2):
       leftRep = self.leftChild.representation           # obtain children's representations
       rightRep = self.rightChild.representation
       n1= self.leftChild.offspring                      # determine children's offspring (weighted reconstruction)
       n2= self.rightChild.offspring
       childrenP = M2.dot(self.representation)+b2        # determine reconstruction
       leftError = (n1/(n1+n2))*np.linalg.norm(leftRep,childrenP[0])
       rightError = (n2/(n1+n2))*np.linalg.norm(rightRep,childrenP[1])

       return leftError+rightError

class Leaf(Node):
   def __init__(self, wordIndex ,index,M1,b1,L):
      self.start = index
      self.end = index
      self.word = word
      self.setRepresentation(vocabulary,M1,b1)
      self.offspring = [self.representation]

   def setRepresentation(self, vocabulary, M1, b1):
      self.representation =  vocabulary[self.word]
   def __str__(self):
       return self.word
