import random
import nltk
import core.myIORNN as myIORNN
import core.myTheta as myTheta
import core.trainingRoutines as trainingRoutines

operators = ['plus','minus']#,times,div]


def createTree(length):
  if length < 1: print 'whatup?'
  if length == 1:
    return nltk.Tree('digit',[random.randint(0, 9)])
  else:
    left = random.randint(1, length-1)
    right = length - left
    children = [createTree(l) for l in [left,right]]
    operator = random.choice(operators)
    children.insert(1,nltk.Tree('operator',[operator]))
    return nltk.Tree(operator,children)


def solveTree(tree):

  if tree.height()==2: return tree[0]
  else:
    children = [solveTree(c) for c in tree]
    if tree.label()== 'plus':
      return children[0]+children[2]
    elif tree.label()== 'minus':
      return children[0]-children[2]
    else: print 'huh'

def bigTree(length):
  tree =createTree(length)
  answer =nltk.Tree('digit',[solveTree(tree)])
  if answer[0] not in xrange(0,25):
#    print answer[0], 'BAD tree'
    return bigTree(length)
  else:  return nltk.Tree('is', [tree,answer])



class Treebank():
  def getExamples(dontBother):
    nws = []
    for i in range(1000):
      tree = bigTree(random.randint(1,5))
      nws.append(myIORNN.IORNN(tree))
    return nws

if __name__ == '__main__':
  dims = {'inside':5,'outside':5,'word':5, 'maxArity':3}
  grammar = {'plus':{('digit, operator, digit'):5},'minus':{('digit, operator, digit'):5},'is':{'digit, digit':5}}

  voc= range(0,25)
  voc.append('UNKNOWN')
  voc.extend(operators)
  theta = myTheta.Theta('IORNN', dims, grammar, embeddings = None, vocabulary= voc)

  theta.specializeHeads()
  hyperParams = {'nEpochs':5,'lambda':0.00001,'alpha':0.01,'bSize':50}
  outDir = 'models/testMath'
  trainingRoutines.plainTrain(Treebank(), Treebank(), hyperParams, True, theta, outDir)
