from __future__ import division
from NN import Node, Leaf

import sys

def this2Nodes(nltkTree, vocabulary):
#  print 'this2Nodes', nltkTree
  if nltkTree.height()>2:
    cat = 'composition'
    lhs = nltkTree.label()
    rhs = '('+ ', '.join([child.label() for child in nltkTree])+')'
    thisOuter = Node([], [], 'TMP', 'sigmoid')

    childrenNodes = [this2Nodes(child,vocabulary) for child in nltkTree]

    childrenOuter= [outer for inner, outer in childrenNodes]
    thisOuter.outputs =childrenOuter
    childrenInner= [inner for inner, outer in childrenNodes]
    thisInner = Node(childrenInner, [], (cat,lhs,rhs,'I'), 'sigmoid')

    childrenOuterInput = childrenInner[:]
    childrenOuterInput.append(thisOuter)
    # append thisOuter to the childrenInner
    # to obtain the input to the childrenOuter nodes

    for j in range(len(nltkTree)):
      # set the inputs for the child's outer representation
      childJInner = childrenOuterInput.pop(j)
      childrenOuter[j].inputs = childrenOuterInput
      childrenOuter[j].cat = (cat,lhs,rhs,j,'O')
      # the category of an outside node is what happens above it.
      # it also needs to know j: which child it is
      childrenOuterInput.insert(j,childJInner) # reset childrenInner

  else: #at a preterminal
    cat = 'word'
    word = nltkTree[0]
#    print 'word is:', word
    try: index = vocabulary.index(word)
    except:
      if word.split('-')[-1] != 'UNK': word += '-UNK'
      index = vocabulary.index('UNK')
    thisInner = Leaf([],'word', word, index,'identity')
    thisOuter = Node([], [], '', 'sigmoid')
    
    uNode = Node([thisOuter,thisInner],[],'u','sigmoid')
    scoreNode = Node([uNode],[],'score','sigmoid')
    uNode.outputs = [scoreNode]

    thisOuter.outputs = [uNode]
  return thisInner, thisOuter

class IORNN():
  def __init__(self, nltkTree, vocabulary):
#    print 'IORNN.init', nltkTree

    self.rootI, rootO = this2Nodes( nltkTree, vocabulary)
    self.rootO = Leaf(rootO.outputs,'root', 0,'identity')
    self.rootO.inputs=[]
    self.scoreNodes = self.findScoreNodes(self.rootO)

  def __str__(self):
    return str(self.rootI)

  def trainWords(self, theta, gradients = None, target = None):
    for scoreNode in self.scoreNodes:
      trainWord(scoreNode, theta, gradients, target)


def findScoreNodes(node):
  if node.cat=='score': return [node]
  else: return [n for sublist in [findScoreNodes(c) for c in node.outputs] for n in sublist]

def trainWord(self, scoreNode, theta, gradients, target):
  nwords = len(theta['word'])
  uNode = scoreNode.inputs[0]
  wordNode = uNode.inputs[0]
  # pick a candidate x different from own index
  if target is None:
    x = myWord.index
    while x == myWord.index:  x = random.randint(0,nwords-1)
  else: x = target

  # compute error for chosen candidate

  error = 1 - self.score(theta, recompute= False)+self.score(theta, x, False)

  if True: #error>1: # if the candidate scores too high: backpropagate error
    # backpropagate through observed node
    delta = -1*scoreNode.Ad
    scoreNode.backprop(delta, theta, gradients)

    # backpropagate through candidate
    # save original settings
    original = myWord.index
    wordNode.index = x

    # locally recompute activations for candidate
    wordNode.forward(theta,activateIn=False, activateOut=False)
    uNode.forward(theta,activateIn=False, activateOut=False)
    scoreNode.forward(theta,activateIn=False, activateOut=False)

    delta = scoreNode.Ad
    scoreNode.backprop(delta, theta, gradients)

    # restore observed node
    self.index = original
    # locally recompute activations for original observed node
    wordNode.forward(theta,activateIn=False, activateOut=False)
    uNode.forward(theta,activateIn=False, activateOut=False)
    scoreNode.forward(theta,activateIn=False, activateOut=False)

  return error