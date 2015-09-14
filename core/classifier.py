import NN, myIORNN, myRAE, myRNN, myTheta
# import sys
import numpy as np
# import pickle
# import nltk, re

# import os



class Classifier(NN.Node):
  def __init__(self,children, labels, fixed = True):
    if fixed: children = [NN.Leaf([],('word',),i) for i in range(children)]
    comparison = NN.Node(children, [self], ('comparison',),'tanh')
#    leafs = [NN.Leaf([comparison],('word',),i) for i in range(n)]
#    comparison.inputs=leafs
    NN.Node.__init__(self,[comparison], [], ('classify',),'softmax')
    self.labels = labels

  def replaceChildren(self,children, fixed):
    if fixed:
      for i in range(len(children)):
        self.inputs[0].inputs[i].key = children[i]
    else: self.inputs[0].inputs = children

  def train(self,theta,gradient,activate=True, target = None,fixed = True):
    if activate: self.forward(theta)
    delta = np.copy(self.a)
    true = self.labels.index(target)
    delta[true] = delta[true]-1
    self.backprop(theta, delta, gradient, addOut = False, moveOn=True, fixWords = fixed)
    error = self.error(theta,target,False)

#    print 'classifier.train: ',[leaf for leaf in self.inputs[0].inputs],target, ',error:', error
    return error


#  def forward(self,theta, activateIn = True, activateOut = False, signal=None):

  def error(self,theta, target, activate=True):
    if activate: self.forward(theta)

    try: err= -np.log(self.a[self.labels.index(target)])
    except: err= -np.log(0.00000000000001)
#    print 'cl.error', self.a, self.labels.index(target), err
    return err

  def evaluate(self, theta, children, gold, fixed = True):
    self.replaceChildren(children, fixed)

    loss = self.error(theta,gold,True)
    return loss









