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
    thisOuter = Node([], [], 'TMP', 'tanh')
#    Node(inputs, outputs, cat,nonlinearity)

    childrenNodes = [this2Nodes(child) for child in nltkTree]

    childrenOuter= [outer for inner, outer in childrenNodes]
    thisOuter.outputs =childrenOuter
    childrenInner= [inner for inner, outer in childrenNodes]
    thisInner = Node(childrenInner, [], (cat,lhs,rhs,'I'), 'tanh')

    childrenOuterInput = childrenInner[:]
    childrenOuterInput.append(thisOuter)
    # append thisOuter to the childrenInner
    # to obtain the input to the childrenOuter nodes
    for j in range(len(childrenOuter)):
      # set the inputs for the child's outer representation
      childrenOuter[j].inputs = childrenOuterInput[:j]+childrenOuterInput[j+1:]
      childrenOuter[j].cat = (cat,lhs,rhs,j,'O')
      # the category of an outside node is what happens above it.
      # it also needs to know j: which child it is

  else: #at a preterminal
    cat = ('word',)
    word = nltkTree[0]
#    print 'word is:', word
    thisInner = Leaf([],('word',), word, 'identity')
    thisOuter = Node([], [], 'TMP', 'tanh')

    uNode = Node([thisOuter,thisInner],[],('u',),'tanh')
    scoreNode = Node([uNode],[],('score',),'identity')
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

def computeError(scoreNode,theta,x,reset = True):
  candidateScore,original  = computeScore(scoreNode, theta, x, reset = True)
  realScore = scoreNode.a[0][0]
  return max(0,1-realScore+candidateScore)

'''compute the score for the given scoreNode if the wordNode's key is replaced by the target: x '''
def computeScore(scoreNode,theta,x, reset = True):
  uNode = scoreNode.inputs[0]
  wordNode = [node for node in uNode.inputs if node.cat==('word',)][0]

  original = wordNode.key  # save original key

  # locally recompute activations for candidate
  wordNode.key = x
  activateScoreNW(uNode,wordNode,scoreNode,theta)
  score = scoreNode.a[0][0]

  if reset: # restore observed node
    wordNode.key = original
    # locally recompute activations for original observed node
    activateScoreNW(uNode,wordNode,scoreNode,theta)
  return score, original

def trainWord(scoreNode, theta, gradients, target, vocabulary):
  uNode = scoreNode.inputs[0]
  wordNode = [node for node in uNode.inputs if node.cat==('word',)][0]
  # pick a candidate x different from own index
  original = wordNode.key
  if target is None:
    x = original
    while x == original or x == 'UNKNOWN':  x = random.choice(vocabulary)
  else: x = target

  print '\ntraining score node of word:',original,', target:', target

  if True: #error>1: # if the candidate scores too high: backpropagate error
    # backpropagate through observed node
    realScore = scoreNode.a[0]
#    print 'real score:', realScore, 'derivative:', scoreNode.ad[0]
    delta = -1*scoreNode.ad[0]
#    cDelta = delta[:]
    scoreNode.backprop(theta,delta, gradients)

    # backpropagate through candidate
    wordNode.key = x
    activateScoreNW(uNode,wordNode,scoreNode,theta) # locally recompute activations for candidate
    candidateScore = scoreNode.a[0]
#    print 'candidate score:', candidateScore,'derivative:', scoreNode.ad[0]
    delta = scoreNode.ad[0]
#    print 'ddelta:',delta+cDelta
    scoreNode.backprop(theta, delta, gradients)

    # restore observed node
    wordNode.key = original
    activateScoreNW(uNode,wordNode,scoreNode,theta) # locally recompute activations for original observed node
#  print '\nDone training score node of word:',wordNode.key,', target:', target
  return max(0,1-realScore+candidateScore)

class IORNN():
  def __init__(self, nltkTree):
#    print 'IORNN.init', nltkTree

    self.rootI, self.rootO = this2Nodes( nltkTree)
#    (self,outputs,cat, word='', index=0,nonlinearity='identity'):
    self.rootO.__class__ = Leaf
    self.rootO.cat=('root',)
    self.rootO.key=''
    self.rootO.nonlin = 'identity'
    self.scoreNodes = findScoreNodes(self.rootO)

#    print 'IORNN inside:', self.rootI
#    print 'IORNN outside:', self.rootO
  def __str__(self):
    return str(self.rootI)

  def setScoreNodes(self): self.scoreNodes = findScoreNodes(self.rootO)

  def words(self):
    words = []
    for scoreNod in self.scoreNodes:
      words.append(wordNode.key for wordNode in [node for node in scoreNode.inputs[0].inputs if node.cat==('word',)][0])
    return words

  def activate(self,theta):
#    print 'activate NW inside:'
    self.rootI.forward(theta,activateIn = True, activateOut = False)
#    print 'activate NW outside:'
    self.rootO.forward(theta,activateIn = False, activateOut = True)
#    print 'activated the network.'

  def trainWords(self, theta, gradient = None, activate=True, target = None):
    if gradient is None: gradient = theta.gradient()
    if activate: self.activate(theta)
    error = 0
    for scoreNode in self.scoreNodes:
      error += trainWord(scoreNode, theta, gradient, target, theta.lookup[('word',)])
    return gradient, error/ len(self.scoreNodes)

  def error(self,theta, target, activate=True):
    if activate: self.activate(theta)
    errors = [computeError(node,theta,target, reset = True) for node in self.scoreNodes]
    return sum(errors)

  def evaluateNAR(self,theta, vocabulary):
    if vocabulary is None: vocabulary = theta.lookup[('word',)]
    else:
      for word in self.words():
        if word not in vocabuary: vocabulary.append(word)

    ranks = 0
    num = 0

    self.activate(theta)

    for scoreNode in self.scoreNodes:
      results = [computeError(scoreNode,theta,x, reset=False) for x in vocabulary]
      # reset the scoreNW
      nothing = computeScore(scoreNode,theta,results[0][1], reset=True)
      scores = [score for score, original in results]
      ranking = np.array(scores).argsort()[::-1].argsort()
      rank = ranking[vocabulary.index(results[0][1])]
#      print 'rank:', rank
      ranks+= rank
    return ranks/(len(vocabulary)*len(self.scoreNodes))

