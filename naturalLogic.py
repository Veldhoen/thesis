from __future__ import division

from nltk import tree
import numpy as np
import re
import numericalGradient2 as ng
import sys

class Network:
  def __init__(self, comparisonlayer):
    self.comparison = comparisonlayer
  def forward(self, theta):
    M1,b1,V,M2,b2, M3, b3 = unwrap(theta)
    self.z = M3.dot(self.comparison.forward(M1,b1,V,M2,b2)) + b3
    self.a = self.z / sum(self.z)
    self.ad = np.ones(len(self.a))
    return self.a


  def backprop(self,trueRelation,theta):
      M1,b1,V,M2,b2, M3, b3 = unwrap(theta)
      target = np.zeros(len(self.a))
      target[trueRelation] = 1
#      delta = np.ones(len(self.a))
#      print np.shape(np.transpose(M3).dot(self.a-target))
#      delta = np.multiply(t1,np.ones(len(self.z)))
#      delta = np.multiply((self.a-target),self.ad)
     # delta = self.a-target
      
      delta = self.a
      delta[trueRelation] = 1-self.a[trueRelation]

#      delta = np.multiply((1-self.a).dot(self.a),self.a-target)
      # compute gradients
      gradM3 = np.outer(delta,self.comparison.a)
      gradb3 = delta

### Based on Bowman implementation
#       gradM3 = np.zeros_like(M3)
#       ac = self.comparison.a
#       for i in range(len(self.a)):
#           d = (trueRelation==i)-self.a[i]
#           print d
#           gradM3[i] = np.transpose(-(np.multiply(ac,d)))
#           gradb3[i] = np.transpose(-(np.multiply(1,d)))
###

      # compute delta to backpropagate
      deltaB = np.multiply(np.transpose(M3).dot(delta),self.comparison.ad)
      # backpropagate to retrieve other gradients
      gradM1, gradb1, gradV, gradM2, gradb2 = self.comparison.backprop(deltaB, M1,b1,V,M2,b2)

      return gradM1, gradb1, gradV, gradM2, gradb2, gradM3, gradb3

  def error(self, trueRelation, theta):
      self.forward(theta)
      return -np.log(self.a[trueRelation])


class RNN:
  def __init__(self,tree,words):
      self.children = [RNN(child,words) for child in tree]
      try:    self.index = words.index(tree.label())
      except: self.index = len(words)+1
  def forward(self, M,b,V):
      if len(self.children) > 0:
         self.z = M.dot(np.concatenate([child.forward(M,b,V) for child in self.children]))+b
         self.a, self.ad = tanh(self.z)
      else:
         self.z = V[self.index]
         self.a, self.ad = identity(self.z)
      return self.a
  def backprop(self, delta, M, b, V):

      if len(self.children) > 0:
          childrenas = np.concatenate([rnn.a for rnn in self.children])
          gradM = np.outer(delta,childrenas)
          gradb = delta

          childrenads = np.concatenate([rnn.ad for rnn in self.children])
          # compute delta to backpropagate
          deltaB = np.split(np.multiply(np.transpose(M).dot(delta),childrenads),2)
          # backpropagate to retrieve other gradients
          grad0M, grad0b, grad0V= self.children[0].backprop(deltaB[0], M,b,V)
          grad1M, grad1b, grad1V= self.children[1].backprop(deltaB[1], M,b,V)
          gradM += grad0M + grad1M
          gradb += grad0b + grad1b
          gradV  = grad0V  + grad1V

      else:
          gradM = np.zeros_like(M)
          gradb = np.zeros_like(b)
          gradV = np.zeros_like(V)
          gradV[self.index] = delta
      return gradM, gradb, gradV
class Comparisonlayer:
  def __init__(self,rnns):
    self.rnns = rnns
  def forward(self, M1,b1,V,M2,b2):
      self.z = M2.dot(np.concatenate([rnn.forward(M1,b1,V) for rnn in rnns]))+b2
      self.a, self.ad = ReLU(self.z)
      return self.a

  def backprop(self,delta, M1,b1,V,M2,b2):
      childrenas = np.concatenate([rnn.a for rnn in self.rnns])
      gradM2 = np.outer(delta,childrenas)
      gradb2 = delta

      childrenads = np.concatenate([rnn.ad for rnn in self.rnns])
      # compute delta to backpropagate
      deltaB = np.split(np.multiply(np.transpose(M2).dot(delta),childrenads),2)
      # backpropagate to retrieve other gradients
      grad0M1, grad0b1, grad0V= self.rnns[0].backprop(deltaB[0], M1,b1,V)
      grad1M1, grad1b1, grad1V= self.rnns[1].backprop(deltaB[1], M1,b1,V)
      gradM1 = grad0M1 + grad1M1
      gradb1 = grad0b1 + grad1b1
      gradV  = grad0V  + grad1V
      return gradM1, gradb1, gradV, gradM2, gradb2

# activation functions:
def identity(vector):
    act = vector
    der = np.ones(len(act))
    return act, der
def tanh(vector):
    act = np.tanh(vector)
    der = 1- np.multiply(act,act)
    return act, der
def ReLU(vector):
    act = np.array([max(x,0) for x in vector])
    der = np.array([1*(x>=0) for x in vector])
    return act, der
# def softmax(vector):
#     act = vector / sum(vector)
#     # kroneker delta: dij = 0 if i!= j, 1 if i == j
# 
#     delta = np.zeros_like(vector)#Kroneker delta
#     der = np.multiply(act,delta-act)
#     return act, der

def wrap((M1,b1,V,M2,b2,M3,b3)):
    theta = np.concatenate([np.reshape(w,-1) for w in [M1,b1,V,M2,b2,M3,b3]])
    return theta
def unwrap(theta):
    left = 0
    right = left + dwords*2*dwords
    M1 = np.reshape(theta[left:right],(dwords, 2*dwords))
    left = right
    right = left + dwords
    b1 = theta[left:right]
    left = right
    right = left+(nwords+1)*dwords
    V = np.reshape(theta[left:right],(nwords+1, dwords))

    left = right
    right = left + dcomparison*2*dwords
    M2 = np.reshape(theta[left:right],(dcomparison, 2*dwords))
    left = right
    right = left + dcomparison
    b2 = theta[left:right]
    left = right
    right = left + numrel*dcomparison
    M3 = np.reshape(theta[left:right],(numrel,dcomparison))
    left = right
    right = left + numrel
    b3 = theta[left:right]
    return M1,b1,V,M2,b2,M3,b3

def compare(grad,numgrad):
    if np.array_equal(numgrad,grad): 
       print 'numerical and analytical gradients are equal.'
       return None
    print 'Difference numerical and analytical gradients:', np.linalg.norm(numgrad-grad)/np.linalg.norm(numgrad+grad)

    names = ['M1','b1','V','M2','b2','M3','b3']
    pairedGrads = zip(unwrap(numgrad),unwrap(grad))

    for i in range(len(names)):
        a,b = pairedGrads[i]
        a = np.reshape(a,-1)
        b = np.reshape(b,-1)
        print 'Difference '+names[i]+' :', np.linalg.norm(a-b)/np.linalg.norm(a+b)



global nwords, dwords, dcomparison,numrel
relations = ['<','>','=','|','^','v','#']
words = ['all', 'no', 'hippo', 'bark']
dwords = 16
dcomparison = 45
numrel = len(relations)
nwords = len(words)
M1 = np.random.rand(dwords, 2*dwords)*.02-.01  #composition weights
b1 = np.random.rand(dwords)*.02-.01       #composition bias
M2 = np.random.rand(dcomparison,2*dwords)*.02-.01
b2 = np.random.rand(dcomparison)*.02-.01       #composition bias
M3 = np.random.rand(len(relations),dcomparison)
b3 = np.random.rand(len(relations))*.02-.01       #composition bias
V = np.random.rand(len(words)+1,dwords)*.02-.01

theta = wrap((M1,b1,V,M2,b2,M3,b3))



line= '|	( all hippo ) bark	( no hippo ) bark'
bits = line.split('\t')
if len(bits) == 3:
   relation, s1, s2 = bits
else: sys.exit()
target = relations.index(relation)
ss = ['('+re.sub(r"([^()\s]+)", r"(\1)", s)+')' for s in [s1,s2]]
rnns  = [RNN(tree.Tree.fromstring(s), words)  for s in ss]
network = Network(Comparisonlayer(rnns))
network.forward(theta)
grad = wrap(network.backprop(target,theta))
numgrad = ng.numericalGradient(network,theta,target)

compare(grad,numgrad)
