import random
from nltk import Tree
import core.myIORNN as myIORNN
import core.myRAE as myRAE
import core.myTheta as myTheta
import core.trainingRoutines as trainingRoutines
import math
import numpy as np
import sys, argparse, os


class mathExpression(Tree):
  def __init__(self,length, operators):
    if length < 1: print 'whatup?'
    if length == 1:
      Tree.__init__(self,'digit',[str(random.randint(-9, 9))])
    else:
      left = random.randint(1, length-1)
      right = length - left
      children = [mathExpression(l,operators) for l in [left,right]]
      operator = random.choice(operators)
      children.insert(1,nltk.Tree('operator',[operator]))
      Tree.__init__(self,operator,children)
  def solve(self):
    if self.height()==2:
      try: return int(self[0])
      except DeprecationWarning:
        print self
        sys.exit()
    else:
      children = [c.solve() for c in [self[0],self[2]]]
      operator = self.label()
      if operator== 'plus':
        return children[0]+children[1]
      elif operator== 'minus':
        return children[0]-children[1]
      elif operator== 'times':
        return children[0]*children[1]
      elif operator== 'div':
        try: return children[0]/children[1]
        except: return 1000000
      else:
        raise Exception('Cannot deal with operator '+str(operator))

class mathTree(Tree):
  def __init__(self,length,voc,operators):
    answer = [None]
    while answer[0] not in voc:
      tree =mathExpression(length,operators)
      answer =Tree('digit',[str(tree.solve())])
      Tree.__init__(self,'is', [tree,answer])

def randomOrthEmbeddings(voc):
  d = math.log(len(voc),2)
  embeddings = []
  random.shuffle(voc)           #random order, no structure in the embeddings
  for i in range(1,len(voc)+1): #don't use the binary encoding of zero, as it is not orthogonal to anything!
    e = bin(i)[2:]
    e = '0'*(int(math.ceil(d))-len(e))+e
    embeddings.append(np.array([float(c) for c in e]))
  return embeddings



class txtTreebank():
  def __init__(self):
    dets = ['one', 'two']
    nouns = ['horse','pig','rabbit','mol','swan','duck','cricket','donkey','cow','peackock','dove','pigeon','animal','caterpillar','butterfly','turtle','hippo']
    iverbs = ['eat','sleep','fly','think','wait','cry']
    tverbs= ['love','hate', 'adore','crave','help','want', 'dislike','like']

    DPs = [Tree('DT',[det]) for det in dets]

    NP1 = [Tree('NP',[DPs[0],Tree('N',[noun])]) for noun in nouns]
    NP2 = [Tree('NP',[DPs[1],Tree('N',[noun+'s'])]) for noun in nouns]
    
    
    IVP1 = [Tree('VP',[Tree('V',[verb+'s'])]) for verb in iverbs]
    IVP2 = [Tree('VP',[Tree('V',[verb])]) for verb in iverbs]
    TVP1 = [Tree('VP',[Tree('V',[verb+'s']),np]) for np in NP1+NP2 for verb in tverbs]
    TVP2 = [Tree('VP',[Tree('V',[verb]),np]) for np in NP1+NP2 for verb in tverbs]
    
    SI1 = [Tree('S',[np,vp]) for np in NP1 for vp in IVP1]
    SI2 = [Tree('S',[np,vp]) for np in NP2 for vp in IVP2]
    ST1 = [Tree('S',[np,vp]) for np in NP1 for vp in TVP1]
    ST2 = [Tree('S',[np,vp]) for np in NP2 for vp in TVP2]

    self.voc = dets+nouns+iverbs+tverbs
    self.voc.append('UNKNOWN')
    self.grammar = {'S':{'NP, VP':5},'NP':{'DT, N':5},'VP':{'V':5,'V, NP':5}}
    self.sentences = SI1+SI2+ST1+ST2

  def getVoc(self):
    return self.voc

  def getGrammar(self):
    return self.grammar

  def getExamples(self):
    nws = [myIORNN.IORNN(tree) for tree in random.sample(self.sentences,500)]
    return nws






class mathTreebank():
  def __init__(self, kind, complexity):
    self.kind = kind
    self.operators = ['plus','minus']
    if complexity == 'complex': self.operators.extend(['times','div'])

    self.grammar = {'plus':{('digit, operator, digit'):5},'minus':{('digit, operator, digit'):5},'is':{'digit, digit':5}}
    self.voc= [str(i) for i in range(-29,29)]
  #  voc= [str(i) for i in range(-60,60)]
    self.voc.append('is')
    self.voc.append('UNKNOWN')
    self.voc.extend(self.operators)

  def getVoc(self):
    return self.voc

  def getGrammar(self):
    return self.grammar

  def getExamples(self, n=1000):
    nws = []
    for i in range(n):
      tree = mathTree(random.randint(1,5),self.voc,self.operators)
      if self.kind == 'IORNN': nws.append(myIORNN.IORNN(tree))
      elif self.kind == 'RAE': nws.append(myRAE.RAE(tree))
    return nws


def main(args):

  if args['datatype'] == 'math':  tb = mathTreebank(args['kind'], args['complexity'])
  elif args['datatype'] == 'txt': tb = txtTreebank()
  else: sys.exit()

  voc = tb.getVoc()
  if args['orthogonal']: embeddings = randomOrthEmbeddings(voc)
  else: embeddings = None
  try: d = len(embeddings[0])
  except: d = 10
  print 'dimensionality:', d

  grammar = tb.getGrammar()
  dims = {'inside':d,'outside':d,'word':d, 'maxArity':3}
  theta = myTheta.Theta(args['kind'], dims, grammar, embeddings, voc)

  if args['specialize']: theta.specializeHeads()
  hyperParams = {'nEpochs':5,'lambda':0.0001,'alpha':0.01,
                 'bSize':50,'fixWords':args['fixEmb']}





  trainingRoutines.storeTheta(theta, os.path.join(args['out'],'initialTheta.pik'))
  trainingRoutines.plainTrain(tb, tb, hyperParams, True, theta, args['out'])

def mybool(string):
  if string in ['F', 'f', 'false', 'False']: return False
  if string in ['T', 't', 'true', 'True']: return True
  raise Exception('Not a valid choice for arg: '+string)


if __name__ == "__main__":
  parser = argparse.ArgumentParser(description='Train RAE/ IORNN unsupervised on artificial data')
  parser.add_argument('-d','--datatype', type=str, choices = ['math','txt'],help='Kind of training (math or txt)', required=True)
  parser.add_argument('-k','--kind', type=str, choices = ['IORNN','RAE'], help='Model type (IORNN or RAE)', required=True)
  parser.add_argument('-o','--out', type=str, help='Output dir to store pickled theta', required=True)
  parser.add_argument('-s','--specialize', type=mybool, help='Whether the parameter are specialized for rule head(True/False)', required=True)
  parser.add_argument('-f','--fixEmb', type=mybool, help='Whether the embeddings must be kept fixed (True/False)', required=True)
  parser.add_argument('-orth','--orthogonal', type=mybool, help='Whether the embeddings must be initialized orthogonal', required=True)
  parser.add_argument('-c','--complexity', type=str, choices = ['simple','complex'], help='Type of artithmetics (simple/complex)', required=False)
  args = vars(parser.parse_args())
  main(args)