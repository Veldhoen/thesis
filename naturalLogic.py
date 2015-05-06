from __future__ import division

import sys, os, re
from nltk import tree
from collections import defaultdict
import numpy as np
import random
import pickle

import NN, IORNN #from NN import *
#from IORNN import *
from training import *
#from params import *


def types4IO(dwords, dint, nrel, nwords):
  types = []
#  types.append(('preterminalM','float64',(dint,dwords)))
#  types.append(('preterminalB','float64',(dint)))
  types.append(('compositionIM','float64',(dint,2*dint)))
  types.append(('compositionIB','float64',(dint)))
  types.append(('compositionOM','float64',(dint,2*dint)))
  types.append(('compositionOB','float64',(dint)))
  types.append(('wordIM','float64',(nwords,dwords)))
  types.append(('wordIB', 'float64',(dwords)))
  types.append(('wordOM', 'float64',(2*dint,2*dint)))
  types.append(('wordOB', 'float64',(2*dint)))
  types.append(('relIM','float64',(nrel,dwords)))
  types.append(('relOM', 'float64',(2*dint,2*dint)))
  types.append(('relOB', 'float64',(2*dint)))
  types.append(('uOM', 'float64',(1,2*dint)))
  types.append(('uOB', 'float64',(1,1))) #matrix with one value, a 1-D array with only one value is a float and that's problematic with indexing
  return types

def types4RNN(dwords, dint, dcomp, nrel, nwords):
  # initialize all parameters randomly using a uniform distribution over [-0.1,0.1]
  types = []
  types.append(('preterminalM','float64',(dint,dwords)))
  types.append(('preterminalB','float64',(dint)))
  types.append(('compositionM','float64',(dint,2*dint)))
  types.append(('compositionB','float64',(dint)))
  types.append(('comparisonM','float64',(dcomp,2*dint)))
  types.append(('comparisonB','float64',(dcomp)))
  types.append(('classifyM','float64',(nrel,dcomp)))
  types.append(('classifyB','float64',(nrel)))
  types.append(('wordIM','float64',(nwords,dwords)))
  return types

def initialize(style, dwords, dint, dcomp, nrel=1, nwords = 1, V = None):
  if style == 'IORNN': types = types4IO(dwords, dint, nrel, nwords)
  elif style == 'RNN': types = types4RNN(dwords, dint, dcomp, nrel, nwords)
  else: print 'PROBLEM'
  # initialize all parameters randomly using a uniform distribution over [-0.1,0.1]
  theta = np.zeros(1,dtype = types)
  for name, t, size in types:
    if name == 'wordIM' and not V is None: theta[name]=V
    if isinstance(size, (int,long)): theta[name] = np.random.rand(size)*.02-.01
    elif len(size) == 2: theta[name] = np.random.rand(size[0],size[1])*.02-.01
    else: print 'invalid size:', size
  return theta[0]




def rnnFromTree(tree, vocabulary, wordReduction = False, grammarBased = False):
  if tree.height() > 2:
    if grammarBased: cat = tree.label()+' -> '+' '.join([child.label() for child in tree])
    else: cat = 'composition'
    children = [rnnFromTree(child,vocabulary,wordReduction) for child in tree]
    return NN.Node(children,cat,'tanh')
  else: #preterminal node
    words = tree.leaves()
    if len(words)== 1: word = words[0]
    else: 'Not exactly one leaf?!', tree
    try: index = vocabulary.index(word)
    except: index = 0
    leaf = NN.Leaf('wordIM',index, word)

    if wordReduction:
    # wordReduction adds an extra layer to reduce high-dimensional words
    # to the dimensionality of the inner representations
      if grammarBased: cat = tree.label()
      else: cat = 'preterminal'
      return NN.Node([leaf],cat,'tanh')
    else: return leaf

def glueNW(trees,rel,reli,voc):
  nws = [iornnFromTree(t, voc) for t in trees]
  relLeaf = IORNN.Leaf('rel',reli, 'tanh',rel)
  cat = 'composition'
  im = IORNN.Node([nws[0],relLeaf],cat,'tanh','tanh')
  return IORNN.Node([im,nws[1]],cat,'tanh','tanh')

def iornnFromTree(tree, vocabulary, grammarBased = False):
#  print tree
  if tree.height() > 2:
    if grammarBased: cat = tree.label()+' -> '+' '.join([child.label() for child in tree])
    else: cat = 'composition'
    children = [iornnFromTree(child,vocabulary, grammarBased) for child in tree]
    parent = IORNN.Node(children,cat,'tanh','tanh')
    return parent
  else: #preterminal node
    words = tree.leaves()
    if len(words)== 1: word = words[0].lower()
    else: print 'Not exactly one leaf?!', tree
    try: index = vocabulary.index(word)
    except: index = 0
    leaf = IORNN.Leaf('word',index, 'tanh',word)
    return leaf

def printParams():
  print 'Network hyperparameters:'
  print '\tword size:', dwords
  print '\tinternal representation size:', dint
  print '\tdcomparison size:', dcomp
  print '\tgrammarbased:', grammarBased
  print 'Training hyperparameters:'
  print '\talpha :', alpha
  print '\tlambdaL2 :', lambdaL2
  print'\tbatch size :',bsize
  print '\tnumber of epochs :', epochs

def main(args):
  kind = args[0]
  style = args[1]

  if kind == 'sickData':
    relations = ['NEUTRAL','ENTAILMENT','CONTRADICTION']
    source = './data/sick.pik'
    embSrc = './data/senna.pik'
  else:
    relations = ['<','>','=','|','^','v','#']
    embSrc = None
    if kind == 'artData15': source = './data/bowman15.pik'
    if kind == 'artData14': source = './data/bowman14.pik'
    else: print 'choose datasource: sickData, artData14 or artData15'
  if not os.path.exists(source):
    print 'No data found at', source
    sys.exit()
#  try:
  if True:
    with open(source, 'rb') as f:
      trainData, testData, trialData, vocabulary = pickle.load(f)
   #   if style == 'IORNN': vocabulary.extend(relations)
    print 'examples loaded'
    if embSrc:
      print'loading embs'
      with open(embSrc, 'rb') as f:
        V,voc = pickle.load(f)
      print'loaded embs'
      for i in range(len(voc)):
        if voc[i] not in vocabulary and voc[i] not in relations and voc[i]!= 'UNK':
          np.delete(V,i,0)
          del(voc[i])
    else: V = None
#   except:
#     print 'Problem loading data'
#     sys.exit()

  treeset = [[],[],[]]
  i=0
  for set in trainData, testData, trialData:
    for (trees,target) in set:
      targeti = relations.index(target)
      if style == 'RNN':
        nw = NN.Top([NN.Node([rnnFromTree(tree, vocabulary,wordReduction=True) for tree in trees],'comparison','ReLU')],'classify','softmax')
      elif style == 'IORNN':
        nw = glueNW(trees,target,targeti,vocabulary)
#        target = None
#      elif style == 'RAE':

      else: print style, 'is not a familiar architecture'
      treeset[i].append((nw,targeti))
    i+=1
  # there is probably a neater way to do this:
  trainData = treeset[0]
  testData = treeset[1]
  trialData = treeset[2]

  printParams()

  print 'There are',len(trainData),'training examples and',len(testData),'test examples. Vocabulary size:', len(vocabulary)
  theta = initialize(style, dwords,dint,dcomp,len(relations),len(vocabulary), V)
  print 'Parameters initialized. Theta norm:',thetaNorm(theta)



#   accuracy, confusion = evaluate(theta,testData)
#   print confusionString(confusion, relations)
#  testcases = random.sample(trainData, 1)
#   for network, target in testcases:
#     print network
#     gradientCheck(theta,network, target)
  bowmanSGD(lambdaL2, alpha, epochs, np.copy(theta), trainData, testData,relations,batchsize = bsize)

if __name__ == "__main__":
    main(sys.argv[1:])