from __future__ import division

import sys, os, re
#from nltk import tree
from collections import defaultdict
import numpy as np
import random
import pickle

import NN, IORNN #from NN import *
#from IORNN import *
from trainingParallel import *
#from training import *
#from params import *


def rnnFromTree(tree, vocabulary, wordReduction = False, grammarBased = False):
  if tree.height() > 2:
    if grammarBased: cat = tree.label()+' -> '+' '.join([child.label() for child in tree])
    else: cat = 'composition'
    children = [rnnFromTree(child,vocabulary,wordReduction) for child in tree]
    return NN.Node(children,cat,'sigmoid')
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
      return NN.Node([leaf],cat,'sigmoid')
    else: return leaf

def glueNW(trees,rel,reli,voc):
  nws = [iornnFromTree(t, voc) for t in trees]
  relLeaf = IORNN.Leaf('rel',reli, 'sigmoid',rel)
  cat = 'composition'
  im = IORNN.Node([nws[0],relLeaf],cat,'sigmoid','sigmoid')
  return IORNN.Node([im,nws[1]],cat,'sigmoid','sigmoid')

def iornnFromTree(tree, vocabulary, grammar= None):
#  try:  print tree
#  except: print 'unprintable tree'
  if tree.height() > 2:
    if len(tree)!=2:
      print 'a non-binary tree!', tree
      sys.exit
    cat = 'composition'
    if grammar:
      lhs = tree.label()
      rhs = '('+ ', '.join([child.label() for child in tree])+')'
      rule = lhs+'->'+rhs
      if rule in grammar[0]:
        cat += '-'+rule+'-'
      elif lhs in grammar[1]:
        cat += '-'+lhs+'-'
    children = [iornnFromTree(child,vocabulary, grammar) for child in tree]
    parent = IORNN.Node(children,cat,'sigmoid','sigmoid')
    return parent
  else: #preterminal node
    words = tree.leaves()
    if len(words)== 1: word = words[0].strip('\"').lower()
    else:
      print 'Not exactly one leaf?!', tree
      word = 'UNK'
    try: index = vocabulary.index(word)
    except:
      try:
        pos = tree.label().split('+')[0]
        index = vocabulary.index('POS-'+pos)
        word += '-'+pos
      except:
        index = 0
        word += '-UNK'
    leaf = IORNN.Leaf('word',index, 'sigmoid',word)
    return leaf

def printParams():
  print 'Network hyperparameters:'
  print '\tword size:', dwords
  print '\tinternal representation size:', din
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
      for i in xrange(len(voc)):
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
  theta = initialize(style, dwords,din,dcomp,len(relations),len(vocabulary), V)
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
