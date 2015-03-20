import nltk
from nltk.tree import Tree
import numpy as np
from collections import defaultdict
global grammarBased
grammarBased = False


def rnnFromTree(tree, vocabulary):
  if tree.height() > 2:
    if grammarBased: cat = tree.label()+' -> '+' '.join([child.label() for child in tree])
    else: cat = 'composition'
    children = [rnnFromTree(child,vocabulary) for child in tree]
    return Node(children,cat)
  else: #preterminal node
    if grammarBased: cat = tree.label()
    else: cat = 'preterminal'
    word = tree[0]
    try: index = vocabulary.index(word)
    except: index = 0
    return Node([Leaf('word',index, word)],cat)

class Node:
  def __init__(self,children,cat):
    self.cat = cat
    self.children = children
    self.nonlin = 'tanh'
  def backprop(self,delta,parameters,gradients):
    childrenas = np.concatenate([child.a for child in self.children])
    childrenads = np.concatenate([child.ad for child in self.children])

    # compute delta to backprop and backprop it
    deltaB = np.split(np.multiply(np.transpose(parameters[self.cat]['M']).dot(delta),childrenads),len(self.children))
    [self.children[i].backprop(deltaB[i], parameters, gradients) for i in range(len(self.children))]

    # update gradients for this node
    pars = gradients[self.cat]
    if 'M' in pars:
      pars['M']+=np.outer(delta,childrenas)
      pars['b']+=delta
    else:
      pars['M']=np.outer(delta,childrenas)
      pars['b']=delta
  def forward(self,parameters):

    inputsignal = np.concatenate([child.forward(parameters) for child in self.children])
    M= parameters[self.cat]['M']
    b= parameters[self.cat]['b']

    self.z = M.dot(inputsignal)+b
#    self.z = parameters[self.cat]['M'].dot(inputsignal)+parameters[self.cat]['b']
    self.a, self.ad = activate(self.z,self.nonlin)
    return self.a
  def __str__(self):
    if self.cat == 'comparison': return '['+'] VS ['.join([str(child) for child in self.children])+']'
    elif self.cat == 'softmax': return ''.join([str(child) for child in self.children])
    else: return '('+' '.join([str(child) for child in self.children])+')'
class Leaf(Node):
  def __init__(self,cat, index = 0, word = ''):
    Node.__init__(self,[],cat)
    self.index = index
    self.word = word
    self.nonlin = 'identity'
    self.cat = cat
  def forward(self,parameters):
    self.z = parameters[self.cat][self.index]
    self.a, self.ad = activate(self.z,self.nonlin)
    return self.a
  def backprop(self,delta, parameters, gradient):
    True
  def __str__(self):
    return self.word
    
class Top(Node):
  def forward(self,parameters):
    Node.forward(self,parameters)
    self.a = np.exp(self.z) /    sum(np.exp(self.z))

  def backprop(self,parameters,target):
    gradients = defaultdict(dict)
    delta = self.a
    delta[target] = -1+delta[target]   # Phong said 1-delta[trueRel], but that did not work
    Node.backprop(self,delta,parameters, gradients)


def activate(vector, nonlinearity):
  if nonlinearity =='identity':
    act = vector
    der = np.ones(len(act))
  elif nonlinearity =='tanh':
    act = np.tanh(vector)
    der = 1- np.multiply(act,act)
  elif nonlinearity =='ReLU':
    act = np.array([max(x,0)+0.01*min(x,0) for x in vector])
    der = np.array([1*(x>=0) for x in vector])
#  elif nonlinearity =='sigmoid'
  else: 
    print 'no familiar nonlinearity.'
    act = vector
    der = np.ones(len(act))
  return act, der







# TODO: backprop for Leaf (words)
#       gradient check






p1='(ROOT (S (S (NP (NP (DT A) (NN group)) (PP (IN of) (NP (NNS kids)))) (VP (VBZ is) (VP (VBG playing) (PP (IN in) (NP (DT a) (NN yard)))))) (CC and) (S (NP (DT an) (JJ old) (NN man)) (VP (VBZ is) (VP (VBG standing) (PP (IN in) (NP (DT the) (NN background)))))) (. .)))'
p2 = '(ROOT (S (S (NP (NP (DT A) (NN group)) (PP (IN of) (NP (NP (NNS boys)) (PP (IN in) (NP (DT a) (NN yard)))))) (VP (VBZ is) (VP (VBG playing)))) (CC and) (S (NP (DT a) (NN man)) (VP (VBZ is) (VP (VBG standing) (PP (IN in) (NP (DT the) (NN background)))))) (. .)))'
trees = [Tree.fromstring(st) for st in [p1,p2]]
print trees[0]

[nltk.treetransforms.chomsky_normal_form(t) for t in trees]
[nltk.treetransforms.collapse_unary(t, collapsePOS = True,collapseRoot = True) for t in trees]

print trees[0]


vocabulary = []
nw = Top([Node([rnnFromTree(tree, vocabulary) for tree in trees],'comparison')],'softmax')
print nw

words = ['UNK']
relations = ['ENTAILMENT','NEUTRAL','CONTRADICTION']
dwords = 16
dint = 16
dcomp = 45
nrel = len(relations)

parameters = defaultdict(dict)
parameters['word'] = np.random.rand(len(words),dwords)*.02-.01           # Word matrix
parameters['preterminal']['M'] = np.random.rand(dint, dwords)*.02-.01    # lowest layer weights
parameters['preterminal']['b'] = np.random.rand(dint)*.02-.01            # lowest layer bias
parameters['composition']['M'] = np.random.rand(dint, 2*dint)*.02-.01    # composition weights
parameters['composition']['b'] = np.random.rand(dint)*.02-.01            # composition bias
parameters['comparison']['M'] = np.random.rand(dcomp,2*dint)*.02-.01       # comparison weights
parameters['comparison']['b'] = np.random.rand(dcomp)*.02-.01            # comparison bias
parameters['softmax']['M'] = np.random.rand(nrel,dcomp)*.02-.01          # softmax weights
parameters['softmax']['b'] = np.random.rand(nrel)*.02-.01                # softmax bias

nw.forward(parameters)
print nw.a
nw.backprop(parameters,relations.index('ENTAILMENT'))