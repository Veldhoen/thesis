from __future__ import division

import sys, os, re
from nltk import tree
from collections import defaultdict
import numpy as np
import random

from IORNN3 import *
from training import *
from getEmbeddings import *

from params import *

def iornnFromTree(tree, vocabulary, grammarBased = False):
  if tree.height() > 2:
    if grammarBased: cat = tree.label()+' -> '+' '.join([child.label() for child in tree])
    else: cat = 'composition'
    children = [iornnFromTree(child,vocabulary, grammarBased) for child in tree]
    parent = Node(children,cat,'tanh','tanh')
#    children[0].setRelatives(parent,children[1])
#    children[1].setRelatives(parent,children[0])
    return parent
  else: #preterminal node
    words = tree.leaves()
    if len(words)== 1: word = words[0]
    else: 'Not exactly one leaf?!', tree
    try: index = vocabulary.index(word)
    except: index = 0
    leaf = Leaf('word',index, 'tanh',word)
    return leaf

def initialize(dwords, dint, dcomp, nrel, nwords = 1, V = None):
  # initialize all parameters randomly using a uniform distribution over [-0.1,0.1]
  types = []
#  types.append(('preterminalM','float64',(dint,dwords)))
#  types.append(('preterminalB','float64',(dint)))
  types.append(('compositionIM','float64',(dint,2*dint)))
  types.append(('compositionIB','float64',(dint)))
  types.append(('compositionOM','float64',(dint,2*dint)))
  types.append(('compositionOB','float64',(dint)))
#  types.append(('comparisonM','float64',(dcomp,2*dint)))
#  types.append(('comparisonB','float64',(dcomp)))
#  types.append(('classifyM','float64',(nrel,dcomp)))
#  types.append(('classifyB','float64',(nrel)))
  types.append(('wordIM','float64',(nwords,dwords)))
  types.append(('wordOM', 'float64',(2*dint,2*dint)))
  types.append(('wordOB', 'float64',(2*dint)))
  types.append(('uOM', 'float64',(1,2*dint)))
#  types.append(('uBO', 'float64',(1)))
  types.append(('uOB', 'float64',(1,1))) #matrix with one value, a 1-D array with only one value is a float and that's problematic with indexing

  theta = np.zeros(1,dtype = types)
  for name, t, size in types:
    if isinstance(size, (int,long)): theta[name] = np.random.rand(size)*.02-.01
    elif len(size) == 2: theta[name] = np.random.rand(size[0],size[1])*.02-.01
    else: print 'invalid size:', size
  if 0 in theta: print 'zero in theta!'
  print 'created Theta:', theta.dtype.names
  return theta[0]

relations = ['<','>','=','|','^','v','#']
vocabulary = ['UNK', 'all','no','warthogs','hippos','dogs','bark','walk','talk']
dwords = 5
dint = 5
dcomp = 75
nrel = len(relations)
nwords = len(vocabulary)
theta = initialize(dwords, dint, dcomp, nrel, nwords)
#s1 = '( all warthogs ) warthogs'
s1 = '( all warthogs ) walk'
t1 = nltk.tree.Tree.fromstring('('+re.sub(r"([^()\s]+)", r"(W \1)", s1)+')')
nw1 = iornnFromTree(t1, vocabulary)
#print nw1.inner(theta)
#print nw1.outer(theta)

#def printReps(nw,theta):
#  print nw, 'Inner:',nw.inner(theta), 'Outer:',nw.outer(theta)
#  for child in nw.children: printReps(child,theta)

#printReps(nw1,theta)
#print theta.dtype
nw1.inner(theta)
nw1.outer(theta)
gradients = np.zeros_like(theta)             
trainPredict(nw1, theta,vocabulary, gradients)
# for name in gradients.dtype.names:
#   theta[name] = theta[name] + gradients[name]

#line = '=	 ( all warthogs ) walk	 ( all warthogs ) walk'
#relation, s1, s2 = bits = line.split('\t')
#trees = [nltk.tree.Tree.fromstring('('+re.sub(r"([^()\s]+)", r"(W \1)", s)+')') for s in [s1,s2]],relations.index(relation)
#nw = Top([Node([iornnFromTree(tree, vocabulary) for tree in trees],'comparison','ReLU')],'classify','softmax')