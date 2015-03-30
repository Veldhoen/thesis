from __future__ import division

import sys, os, re
from nltk import tree
from collections import defaultdict
import numpy as np
import random

from NN import *
from training import *
from getEmbeddings import *

def initialize(dwords, dint, dcomp, nrel, nwords = 1, V = None):
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
  types.append(('word','float64',(nwords,dwords)))
  theta = np.zeros(1,dtype = types)
  for name, t, size in types:
    if isinstance(size, (int,long)): theta[name] = np.random.rand(size)*.02-.01
    elif len(size) == 2: theta[name] = np.random.rand(size[0],size[1])*.02-.01
    else: print 'invalid size:', size
  if 0 in theta: print 'zero in theta!'
  return theta[0]


def rnnFromTree(tree, vocabulary, wordReduction = False, grammarBased = False):
  if tree.height() > 2:
    if grammarBased: cat = tree.label()+' -> '+' '.join([child.label() for child in tree])
    else: cat = 'composition'
    children = [rnnFromTree(child,vocabulary,wordReduction) for child in tree]
    return Node(children,cat,'tanh')
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
      return Node([leaf],cat,'tanh')
    else: return leaf


def artData(source, relations):

  # make a list of files to open
  if os.path.isdir(source):
    toOpen = [os.path.join(source,f) for f in os.listdir(source)]
    toOpen = [f for f in toOpen if os.path.isfile(f)]
  elif os.path.isfile(source): toOpen = [source]


  examples = []
  vocabulary = ['UNK']
  for f in toOpen:
    with open(f,'r') as f:
      for line in f:
        bits = line.split('\t')
        if len(bits) == 3:
          relation, s1, s2 = bits
          # add unknown words to vocabulary
          for word in s1.split()+s2.split():
            if word !=')' and word != '(' and word not in vocabulary:
              vocabulary.append(word)
          # add training example to set
          examples.append(([nltk.tree.Tree.fromstring('('+re.sub(r"([^()\s]+)", r"(W \1)", s)+')') for s in [s1,s2]],relations.index(relation)))
  # Now that the vocabulary is established, create neural networks from the examples
  networks = []
  for (trees, target) in examples:
    nw = Top([Node([rnnFromTree(tree, vocabulary) for tree in trees],'comparison','ReLU')],'classify','softmax')
    networks.append((nw,target))
  np.random.shuffle(networks)
  nTest = len(networks)//5
  trainData = networks[:4*nTest]
  testData = networks[4*nTest:]
  trialData = []


  return trainData, testData, trialData, vocabulary

def sickData(source, relations):

  # make a list of files to open
  if os.path.isdir(source):
    toOpen = [os.path.join(source,f) for f in os.listdir(source)]
    toOpen = [f for f in toOpen if os.path.isfile(f)]
  elif os.path.isfile(source): toOpen = [source]
  examples = []
  vocabulary = ['UNK']
  for f in toOpen:
    with open(f,'r') as f:
      next(f) # skip header
      for line in f:
        bits = line.split('\t')
        s1 = bits[2]
        s2 = bits[4]
        relation = bits[5]
        kind = bits[-1].strip()
        for word in s1.split()+s2.split():
          if word !=')' and word != '(' and word not in vocabulary:
            vocabulary.append(word)
          # add training example to set
        examples.append(([nltk.tree.Tree.fromstring('('+re.sub(r"([^()\s]+)", r"(W \1)", s)+')') for s in [s1,s2]],relations.index(relation),kind))
  # Now that the vocabulary is established, create neural networks from the examples
  networks = defaultdict(list)
  for (trees, target,kind) in examples:
    try:
      [nltk.treetransforms.chomsky_normal_form(t) for t in trees]
      [nltk.treetransforms.collapse_unary(t, collapsePOS = True,collapseRoot = True) for t in trees]
      nw = Top([Node([rnnFromTree(tree, vocabulary,wordReduction=True) for tree in trees],'comparison','tanh')],'classify','softmax')
      networks[kind].append((nw,target))
    except:
      print 'problem with trees', trees
  return networks['TRAIN'], networks['TEST'], networks['TRIAL'], vocabulary



def main(args):
  kind = args[0]
  if kind == 'sickData':
    source = './data/sick/SICKcorpusSample.txt'
    embSrc = './data/senna'
  if kind == 'artData':
    source = './data/bowman15'
  if not os.path.exists(source):
    print 'No data found at', source
    sys.exit()


# network hyperparameters
  dwords = 16
  dint = 16
  dcomp = 45
  grammarBased = False
  # training hyperparameters
  alpha = 0.2
  lambdaL2 = 0.0002


  epochs = 20
  batches = 10
  rounds = epochs//batches


  print 'Reading corpus...'
  if kind == 'artData':
    relations = ['<','>','=','|','^','v','#']
    trainData, testData, trialData, vocabulary = artData(source, relations)
    V = None
  else:
    relations = ['NEUTRAL', 'ENTAILMENT', 'CONTRADICTION']
    trainData, testData, trialData, vocabulary = sickData(source, relations)
    V,voc = getSennaEmbs(embSrc, vocabulary)

  print 'Done. Retrieved ',len(trainData),'training examples and',len(testData),'test examples. Vocabulary size:', len(vocabulary)
  theta = initialize(dwords,dint,dcomp,len(relations),len(vocabulary), V)#dwords, dint, dcomp, nrel, nwords = 1, V = None
  print 'Parameters initialized. Theta norm:',thetaNorm(theta)

#  testcases = random.sample(trainData, 1)
#  for network, target in testcases:
#    print network
#    gradientCheck(theta,network, target)
  for i in range(rounds):
    print 'Training round', i
    thetaSGD = SGD(lambdaL2, alpha, batches, np.copy(theta), trainData, batchsize = 32)
    accuracy, confusion = evaluate(thetaSGD,testData)
    print '\tAccuracy:', accuracy
  print confusionString(confusion, relations)
  print 'Accuracy:', accuracy
if __name__ == "__main__":
   main(sys.argv[1:])