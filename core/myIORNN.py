from __future__ import division
from NN import Node, Leaf
import random
import sys
import numpy as np

def this2Nodes(nltkTree):
  if nltkTree.height()>2:
    cat = 'composition'
    lhs = nltkTree.label()
    rhs = '('+ ', '.join([child.label() for child in nltkTree])+')'
    thisOuter = Node([], [], (), 'tanh')
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
    thisInner = Leaf([],('word',), word, 'identity')
    thisOuter = Node([], [], 'TMP', 'tanh')

    uNode = Node([thisOuter,thisInner],[],('u',),'tanh')
    scoreNode = Node([uNode],[],('score',),'tanh')
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

'''compute the score for the given scoreNode if the wordNode's key is replaced by the target: x '''
def computeScore(scoreNode,theta,x=None, reset = True):

  uNode = scoreNode.inputs[0]
  wordNode = [node for node in uNode.inputs if node.cat==('word',)][0]

  original = wordNode.key  # save original key
  
  if x is None: score = scoreNode.a[0]
  else:
    # locally recompute activations for candidate
    wordNode.key = x
    activateScoreNW(uNode,wordNode,scoreNode,theta)
    score = scoreNode.a[0]
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
    while x == original:  x = random.choice(vocabulary)
  else: x = target

  if True: #error>1: # if the candidate scores too high: backpropagate error
    # backpropagate through observed node
    realScore = scoreNode.a[0]
    delta = -1*scoreNode.ad
    scoreNode.backprop(theta,delta, gradients)

    # backpropagate through candidate
    wordNode.key = x
    activateScoreNW(uNode,wordNode,scoreNode,theta) # locally recompute activations for candidate
    candidateScore = scoreNode.a[0]
    delta = scoreNode.ad
    scoreNode.backprop(theta, delta, gradients)

    # restore observed node
    wordNode.key = original
    activateScoreNW(uNode,wordNode,scoreNode,theta) # locally recompute activations for original observed node
  return max(0,1-realScore+candidateScore)

class IORNN():
  def __init__(self, nltkTree):
    self.rootI, self.rootO = this2Nodes( nltkTree)
    self.rootO.__class__ = Leaf
    self.rootO.cat=('root',)
    self.rootO.key=0
    self.rootO.nonlin = 'identity'
    self.scoreNodes = findScoreNodes(self.rootO)

  def __str__(self):
    return str(self.rootI)

  def maxArity(self,node=None):
    if node is None: node = self.rootI
    ars = [self.maxArity(c) for c in node.inputs]
    ars.append(len(node.inputs))
    return max(ars)

  def setScoreNodes(self): self.scoreNodes = findScoreNodes(self.rootO)

  def words(self):
    words = []
    for scoreNode in self.scoreNodes:
      uNode = scoreNode.inputs[0]
      wordNode = [node for node in uNode.inputs if node.cat==('word',)][0]
      words.append(wordNode.key)
    return words

  def activate(self,theta):
    self.rootI.forward(theta,activateIn = True, activateOut = False)
    self.rootO.forward(theta,activateIn = False, activateOut = True)

  def train(self, theta, gradient, activate=True, target = None):
    if activate: self.activate(theta)
    error = 0
    for scoreNode in self.scoreNodes:
      error += trainWord(scoreNode, theta, gradient, target, theta[('word',)].keys())
    return error/ len(self.scoreNodes)

  def error(self,theta, target, activate=True):
    if activate: self.activate(theta)
    errors = []
    for node in self.scoreNodes:
     originalscore,o = computeScore(node,theta,None)
     candidatescore,o = computeScore(node,theta,target, reset = True)
     errors.append(max(0,1-originalscore+candidatescore))
    return sum(errors)

  def evaluate(self,theta, sample=1):
    vocabulary = theta[('word',)].keys()
    if sample <1 and sample>0:
      vocabulary = random.sample(vocabulary, int(sample*len(vocabulary)))
      for word in self.words():
        if word not in vocabulary: vocabulary.append(word)
    ranks = 0
    num = 0
    self.activate(theta)
    for scoreNode in self.scoreNodes:
      results = [computeScore(scoreNode,theta,x, reset=False)for x in vocabulary]
      # reset the scoreNW
      nothing = computeScore(scoreNode,theta,results[0][1], reset=False)
      scores = [score for score, original in results]
      ranking = np.array(scores).argsort()[::-1].argsort()
      rank = ranking[vocabulary.index(results[0][1])]
      ranks+= rank
    return ranks/(len(vocabulary)*len(self.scoreNodes))

