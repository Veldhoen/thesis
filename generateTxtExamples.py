import random
from nltk import Tree
import core.myIORNN as myIORNN
import core.myTheta as myTheta
import core.trainingRoutines as trainingRoutines

class Treebank():
  def getExamples(dontBother):
    nws = [myIORNN.IORNN(tree) for tree in random.sample(sentences,500)]
    return nws


dets = ['one', 'two']
nouns = ['horse','pig','rabbit','mol','swan','duck','cricket','donkey','cow','peackock','dove','pigeon','animal','caterpillar','butterfly','turtle','hippo']
iverbs = ['eat','sleep','fly','think','wait','cry']
tverbs= ['love','hate', 'adore','crave','help','want', 'dislike','like']



DPs = [Tree('DP',[det]) for det in dets]

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

sentences = SI1+SI2+ST1+ST2


if __name__ == '__main__':
  dims = {'inside':5,'outside':5,'word':5, 'maxArity':2}
  grammar = {'plus':{('digit, operator, digit'):5},'minus':{('digit, operator, digit'):5},'is':{'digit, digit':5}}

  voc= dets + nouns + [w+'s' for w in nouns] + iverbs + [w+'s' for w in iverbs] + tverbs + [w+'s' for w in tverbs]
  voc.append('UNKNOWN')
  theta = myTheta.Theta('IORNN', dims, grammar, embeddings = None, vocabulary= voc)

#  theta.specializeHeads()
  hyperParams = {'nEpochs':5,'lambda':0.00001,'alpha':0.01,'bSize':50}
  outDir = 'models/testMath'
  trainingRoutines.plainTrain(Treebank(), Treebank(), hyperParams, True, theta, outDir)
