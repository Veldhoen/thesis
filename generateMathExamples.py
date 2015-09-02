import random
import nltk
import core.myIORNN as myIORNN
import core.myRAE as myRAE
import core.myTheta as myTheta
import core.trainingRoutines as trainingRoutines
import math
import numpy as np
import sys

operators = ['plus','minus']#,times,div]


def createTree(length):
  if length < 1: print 'whatup?'
  if length == 1:
    return nltk.Tree('digit',[str(random.randint(-9, 9))])
  else:
    left = random.randint(1, length-1)
    right = length - left
    children = [createTree(l) for l in [left,right]]
    operator = random.choice(operators)
    children.insert(1,nltk.Tree('operator',[operator]))
    return nltk.Tree(operator,children)


def solveTree(tree):

  if tree.height()==2: 
    try: return int(tree[0])
    except DeprecationWarning:
      print tree[0]
      sys.exit()
  else:
    children = [solveTree(c) for c in [tree[0],tree[2]]]
    if tree.label()== 'plus':
      return children[0]+children[1]
    elif tree.label()== 'minus':
      return children[0]-children[1]
#    elif tree.label()== 'minus':
#      return children[0]-children[1]

    else: print 'huh',tree.label()

def bigTree(length,voc):
  tree =createTree(length)
  answer =nltk.Tree('digit',[str(solveTree(tree))])
  if answer[0] not in voc:
#    print answer[0], 'BAD tree'
    return bigTree(length,voc)
  else:  return nltk.Tree('is', [tree,answer])

def randomEmbeddings(voc):
  d = math.log(len(voc),2)
  embeddings = []
  random.shuffle(voc)
  for i in range(len(voc)):
    e = bin(i)[2:]
    e = '0'*(int(math.ceil(d))-len(e))+e
    embeddings.append(np.array([float(c) for c in e]))
  return embeddings

class Treebank():
  def __init__(self,voc, kind):
    self.voc = voc
    self.kind = kind
  def getExamples(self, n=1000):
    nws = []
    for i in range(n):
      tree = bigTree(random.randint(1,5),self.voc)
      if self.kind == 'IORNN': nws.append(myIORNN.IORNN(tree))
      elif self.kind == 'RAE': nws.append(myRAE.RAE(tree))
    return nws

def main(args):
  outDir = args[0]
  specialize = args[1]=='wS'
  fixWords = args[2]=='fE'
  kind = args[3]
  if kind not in ['RAE', 'IORNN']:
    print 'invalid kind:', kind
    sys.exit()

  grammar = {'plus':{('digit, operator, digit'):5},'minus':{('digit, operator, digit'):5},'is':{'digit, digit':5}}

  voc= [str(i) for i in range(-30,30)]
#  voc= [str(i) for i in range(-60,60)]
  voc.append('is')
  voc.append('UNKNOWN')
  voc.extend(operators)
  embeddings = randomEmbeddings(voc)
  d = len(embeddings[0])
  print 'dimensionality:', d
  dims = {'inside':d,'outside':d,'word':d, 'maxArity':3}
  theta = myTheta.Theta(kind, dims, grammar, embeddings = embeddings, vocabulary= voc)

  if specialize: theta.specializeHeads()
  hyperParams = {'nEpochs':5,'lambda':0.0001,'alpha':0.01,'bSize':50,'fixWords':fixWords}

  trainingRoutines.plainTrain(Treebank(voc,kind), Treebank(voc,kind), hyperParams, True, theta, outDir)

if __name__ == '__main__':
  main(sys.argv[1:])