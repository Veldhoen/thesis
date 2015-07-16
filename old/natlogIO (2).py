from __future__ import division

import sys, os, re
from nltk import tree
from collections import defaultdict
import numpy as np
import random

from IORNN import *
from training import *
from getEmbeddings import *

from params import *

def iornnFromTree(tree, vocabulary, wordReduction = False, grammarBased = False):
  if tree.height() > 2:
    if grammarBased: cat = tree.label()+' -> '+' '.join([child.label() for child in tree])
    else: cat = 'composition'
    children = [iornnFromTree(child,vocabulary,wordReduction) for child in tree]
    parent = Node(children,cat,'tanh')
    children[0].setRelatives(parent,children[1])
    children[1].setRelatives(parent,children[0])
    return parent
  else: #preterminal node
    words = tree.leaves()
    if len(words)== 1: word = words[0]
    else: 'Not exactly one leaf?!', tree
    try: index = vocabulary.index(word)
    except: index = 0
    leaf = Leaf('word',index, word)

    if wordReduction:
    # wordReduction adds an extra layer to reduce high-dimensional words
    # to the dimensionality of the inner representations
      if grammarBased: cat = tree.label()
      else: cat = 'preterminal'
      parent = Node([leaf],cat,'tanh')
      leaf.setRelatives(parent,None)
      return parent
    else: return leaf

def initialize(dwords, dint, dcomp, nrel, nwords = 1, V = None):
  # initialize all parameters randomly using a uniform distribution over [-0.1,0.1]
  types = []
#  types.append(('preterminalM','float64',(dint,dwords)))
#  types.append(('preterminalB','float64',(dint)))
  types.append(('compositionMI','float64',(dint,2*dint)))
  types.append(('compositionBI','float64',(dint)))
  types.append(('compositionMO','float64',(dint,2*dint)))
  types.append(('compositionBO','float64',(dint)))
#  types.append(('comparisonM','float64',(dcomp,2*dint)))
#  types.append(('comparisonB','float64',(dcomp)))
#  types.append(('classifyM','float64',(nrel,dcomp)))
#  types.append(('classifyB','float64',(nrel)))
  types.append(('uM', 'float64',(2*dint,2*dint)))
  types.append(('uB', 'float64',(2*dint)))
  types.append(('scoreM', 'float64',(2*dint,1)))
  types.append(('word','float64',(nwords,dwords)))
  theta = np.zeros(1,dtype = types)
  for name, t, size in types:
    if isinstance(size, (int,long)): theta[name] = np.random.rand(size)*.02-.01
    elif len(size) == 2: theta[name] = np.random.rand(size[0],size[1])*.02-.01
    else: print 'invalid size:', size
  if 0 in theta: print 'zero in theta!'
  return theta[0]

relations = ['<','>','=','|','^','v','#']
vocabulary = ['UNK', 'all','no','warthogs','walk','talk']
dwords = 25
dint = 25
dcomp = 75
nrel = len(relations)
nwords = len(vocabulary)
theta = initialize(dwords, dint, dcomp, nrel, nwords)
s1 = '( all warthogs ) walk'
t1 = nltk.tree.Tree.fromstring('('+re.sub(r"([^()\s]+)", r"(W \1)", s1)+')')
nw1 = iornnFromTree(t1, vocabulary)
print nw1.inner(theta)
print nw1.outer(theta)

def printReps(nw,theta):
  print nw, 'Inner:',nw.inner(theta), 'Outer:',nw.outer(theta)
  for child in nw.children: printReps(child,theta)

#printReps(nw1,theta)
nw1.forwardOuter(theta)
trainPredict(nw1, theta,vocabulary)
#line = '=	 ( all warthogs ) walk	 ( all warthogs ) walk'
#relation, s1, s2 = bits = line.split('\t')
#trees = [nltk.tree.Tree.fromstring('('+re.sub(r"([^()\s]+)", r"(W \1)", s)+')') for s in [s1,s2]],relations.index(relation)
#nw = Top([Node([iornnFromTree(tree, vocabulary) for tree in trees],'comparison','ReLU')],'classify','softmax')