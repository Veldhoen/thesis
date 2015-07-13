from __future__ import division
from NN import Node, Leaf
import random
import sys
import numpy as np

def this2Nodes(nltkTree):
#  print 'this2Nodes', nltkTree
  if nltkTree.height()>2:
    cat = 'composition'
    lhs = nltkTree.label()
    rhs = '('+ ', '.join([child.label() for child in nltkTree])+')'
    thisOuter = Node([], [], 'TMP', 'sigmoid')
#    Node(inputs, outputs, cat,nonlinearity)

    childrenNodes = [this2Nodes(child) for child in nltkTree]

    childrenOuter= [outer for inner, outer in childrenNodes]
    thisOuter.outputs =childrenOuter
    childrenInner= [inner for inner, outer in childrenNodes]
    thisInner = Node(childrenInner, [], (cat,lhs,rhs,'I'), 'sigmoid')

    childrenOuterInput = childrenInner[:]
    childrenOuterInput.append(thisOuter)
    # append thisOuter to the childrenInner
    # to obtain the input to the childrenOuter nodes

    for j in range(len(childrenOuter)):
      # set the inputs for the child's outer representation
      childJInner = childrenOuterInput.pop(j)
      childrenOuter[j].inputs = childrenOuterInput
      childrenOuter[j].cat = (cat,lhs,rhs,j,'O')
      # the category of an outside node is what happens above it.
      # it also needs to know j: which child it is
      childrenOuterInput.insert(j,childJInner) # reset childrenInner

  else: #at a preterminal
    cat = ('word',)
    word = nltkTree[0]
#    print 'word is:', word
    thisInner = Leaf([],('word',), word, 'identity')
    thisOuter = Node([], [], 'TMP', 'sigmoid')

    uNode = Node([thisOuter,thisInner],[],('u',),'sigmoid')
    scoreNode = Node([uNode],[],('score',),'sigmoid')
    uNode.outputs = [scoreNode]

    thisOuter.outputs = [uNode]
  return thisInner, thisOuter

def findScoreNodes(node):
  if node.cat==('score',): return [node]
  else: return [n for sublist in [findScoreNodes(c) for c in node.outputs] for n in sublist]

def activateScoreNW(uNode,wordNode,scoreNode,theta):
  wordNode.forward(theta,activateIn=False, activateOut=False)
  uNode.forward(theta,activateIn=False, activateOut=False)
  scoreNode.forward(theta,activateIn=False, activateOut=False)

def computeScore(scoreNode,theta,x, activate=True, reset = True):
  uNode = scoreNode.inputs[0]
  wordNode = [node for node in uNode.inputs if node.cat==('word',)][0]
  original = wordNode.key
  wordNode.key = x

  if activate:# locally recompute activations for candidate
    activateScoreNW(uNode,wordNode,scoreNode,theta)
    score = scoreNode.a[0][0]

  if reset:
    # restore observed node
    wordNode.key = original
    # locally recompute activations for original observed node
    activateScoreNW(uNode,wordNode,scoreNode,theta)
  return score, original

def trainWord(scoreNode, theta, gradients, target, vocabulary):
  print 'trainWord types:'
  print'\t theta',type(theta),'gradients',type(gradients)


  uNode = scoreNode.inputs[0]
  wordNode = [node for node in uNode.inputs if node.cat==('word',)][0]
  # pick a candidate x different from own index
  if target is None:
    original = wordNode.key
    x = original
    while x == original or x == 'UNKNOWN':  x = random.choice(vocabulary)
  else: x = target

  # compute error for chosen candidate
#  error = 1 - self.score(theta, recompute= False)+self.score(theta, x, False)

  if True: #error>1: # if the candidate scores too high: backpropagate error
    # backpropagate through observed node
    realScore = scoreNode.a
    delta = -1*scoreNode.ad
    print  np.shape(scoreNode.a),np.shape(scoreNode.ad), np.shape(delta)
    scoreNode.backprop(theta,delta, gradients)

    # backpropagate through candidate
    # save original settings
    original = wordNode.key
    wordNode.key = x

    # locally recompute activations for candidate
    activateScoreNW(uNode,wordNode,scoreNode,theta)

    candidateScore = scoreNode.a
    delta = scoreNode.ad
    scoreNode.backprop(delta, theta, gradients)

    # restore observed node
    wordNode.key = original
    # locally recompute activations for original observed node
    activateScoreNW(uNode,wordNode,scoreNode,theta)

  return 1-realScore+candidateScore

class IORNN():
  def __init__(self, nltkTree):
#    print 'IORNN.init', nltkTree

    self.rootI, self.rootO = this2Nodes( nltkTree)
#    (self,outputs,cat, word='', index=0,nonlinearity='identity'):
    self.rootO.__class__ = Leaf
    self.rootO.cat=('root',)
    self.rootO.key=''

    # = Leaf(rootO.outputs,'root',index= 0,nonlinearity = 'sigmoid')
#    self.rootO.inputs=[]
    self.scoreNodes = findScoreNodes(self.rootO)

#    print 'IORNN inside:', self.rootI
#    print 'IORNN outside:', self.rootO
  def __str__(self):
    return str(self.rootI)

  def setScoreNodes(self): self.scoreNodes = findScoreNodes(self.rootO)

  def activate(self,theta):
    print 'activate NW inside:'
    self.rootI.forward(theta,activateIn = True, activateOut = False)
    print 'activate NW outside:'
    self.rootO.forward(theta,activateIn = False, activateOut = True)
    print 'activated the network.'

  def trainWords(self, theta, gradients = None, activate=True, target = None):
    print 'trainWords types:'
#    print'\t theta',type(theta),'gradients',type(gradients)


    if activate: self.activate(theta)
    error = 0
    for scoreNode in self.scoreNodes:
      error += trainWord(scoreNode, theta, gradients, target, theta.lookup[('word',)])
    return gradients, error/ len(self.scoreNodes)

  def evaluateNAR(self,theta, vocabulary):
    ranks = 0
    num = 0
    nwords = len(theta[('word',)])
    self.activate(theta)
    if len(self.scoreNodes) <1:
      print 'this network has no score nodes?!', self
      return 0
    for scoreNode in self.scoreNodes:
#      print 'at a scoreNode'


      results = [computeScore(scoreNode,theta,x, True, False) for x in vocabulary]

      # reset the scoreNW
      computeScore(scoreNode,theta,results[0][0], True, False)
#      print 'original score:',score

      scores = [score for score, original in results]
      ranking = np.array(scores).argsort()[::-1].argsort()
      rank = ranking[vocabulary.index(originals[0])]
#      print 'rank:', rank
      ranks+= rank
    return ranks/(nwords*len(self.scoreNodes))

