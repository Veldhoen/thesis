import NN, myIORNN, myRAE, myRNN, myTheta
# import sys
import numpy as np
# import pickle
# import nltk, re

# import os



class Classifier(NN.Node):
  def __init__(self,children, labels, fixed):
#    print 'CLassifier.init', children
    if fixed: children = [NN.Leaf([],('word',),i) for i in range(children)]
    comparison = NN.Node(children, [self], ('comparison',),'ReLU')
#    leafs = [NN.Leaf([comparison],('word',),i) for i in range(n)]
#    comparison.inputs=leafs
    NN.Node.__init__(self,[comparison], [], ('classify',),'softmax')
    self.labels = labels

  def replaceChildren(self,children, fixed):
    if fixed:
      for i in range(len(children)):
        self.inputs[0].inputs[i].key = children[i]
    else: self.inputs[0].inputs = children

  def backprop(self, theta, delta, gradient, addOut = False, moveOn=False, fixWords = False,fixWeights=False):
    if fixWeights: #ignore fixWeights for the classifier weights
      NN.Node.backprop(self,theta, delta, gradient, addOut = addOut, moveOn=False, fixWords = True,fixWeights=False)
    NN.Node.backprop(self,theta, delta, gradient, addOut = addOut, moveOn=moveOn, fixWords = fixWords,fixWeights=fixWeights)

  def train(self,theta,gradient,activate, target,fixWords, fixWeights):
#    print str(self)
    if activate: self.forward(theta)
#    print self.a, self, target
#    print 'activated'
    delta = np.copy(self.a)
    true = self.labels.index(target)
    delta[true] -= 1
    self.backprop(theta, delta, gradient, addOut = False, moveOn=True, fixWords = fixWords, fixWeights=fixWeights)
    error = self.error(theta,target,False)

#    print 'classifier.train: ',[leaf for leaf in self.inputs[0].inputs],target, ',error:', error
    return error


#  def forward(self,theta, activateIn = True, activateOut = False, signal=None):

  def error(self,theta, target, activate=True):
    if activate: self.forward(theta)

    try: err= -np.log(self.a[self.labels.index(target)])
    except:
     # print self.a
      err = -np.log(1e-10)
#    print 'cl.error', self.a, self.labels.index(target), err
    return err

  def evaluate(self, theta, target, sample=1):
    return self.error(theta,target,True)

  def evaluate2(self, theta, children, gold, fixed = True):
    self.replaceChildren(children, fixed)

    loss = self.error(theta,gold,True)
    return loss

  def predict(self,theta,children=None, fixed = True, activate = True):
    if children is not None: self.replaceChildren(children, fixed)
    if activate: self.forward(theta)
    return self.labels[self.a.argmax(axis=0)]

  def __str__(self):
    return 'classify: '+', '.join([str(ch) for ch in self.inputs[0].inputs])






