import NN, myIORNN, myRAE, myRNN, myTheta
import sys
import numpy as np
import pickle
import nltk, re
import random


class Classifier(NN.Node):
  def __init__(self,n, labels):
    comparison = NN.Node([], [self], ('comparison',),'tanh')
    leafs = [NN.Leaf([comparison],('word',),i) for i in range(n)]
    comparison.inputs=leafs
    NN.Node.__init__(self,[comparison], [], ('classify',),'softmax')
    self.labels = labels

  def setInputs(self, keys):
    for i in range(len(keys)):
      self.inputs[0].inputs[i].key = keys[i]


  def train(self,theta,gradient,activate=True, target = None):
    if activate: self.forward(theta)
    delta = np.copy(self.a)
    true = self.labels.index(target)
    delta[true] = delta[true]-1
    self.backprop(theta, delta, gradient, addOut = False, moveOn=True, fixWords = True)
    error = self.error(theta,target,False)

#    print 'classifier.train: ',[leaf for leaf in self.inputs[0].inputs],target, ',error:', error
    return error


#  def forward(self,theta, activateIn = True, activateOut = False, signal=None):

  def error(self,theta, target, activate=True):
    if activate: self.forward(theta)

    try: err= -np.log(self.a[self.labels.index(target)])
    except: err= -np.log(0.00000000000001)
#    print 'cl.error', self.a, self.labels.index(target), err
    return err

  def evaluate(self, theta, keys, gold):
    self.setInputs(keys)
    return self.error(theta,gold,True)




def getFromIFile(fromFile):
  data = {}
  with open(fromFile, 'r') as f:
    header = next(f)
    for line in f:
      try: gold_label, s1_binParse, s2_binParse, s1_parse, s2_parse, sentence1, sentence2, captionID, pairID, l0,l1,l2,l3,l4 = line.split('\t')
      except:
        print len( line.split('\t'))
        sys.exit()
      data[pairID] = ([s1_parse,s2_parse], gold_label)
  return data

def install(thetaFile):
  allData = {}
  src = 'C:/Users/Sara/AI/thesisData/snli_1.0/snli_1.0_'
  for dset in ['train','dev','test']:
    allData[dset] = getFromIFile(src+dset+'.txt')

  with open(thetaFile, 'rb') as f:
    theta = pickle.load(f)
  labels = set()
  embeddings=[]
  vocabulary=[]
  for data in allData.values():
    unable = 0
    for pairID, (parses, gold_label) in data.items():
      labels.add(gold_label)
      ts = [ nltk.Tree.fromstring(p.lower()) for p in parses]
      for t in ts:
        for leafPos in t.treepositions('leaves'):
          word = t[leafPos]
          digit = True
          bits = re.split(',|\.',word)
          for b in bits:
            if not b.isdigit(): digit = False
          if digit: t[leafPos]= '0'

      nws = [myRNN.RNN(t) for t in ts]
      try:embs = [nw.activate(theta) for nw in nws]
      except:
        unable+=1
        continue
      embeddings.extend(embs)
      vocabulary.extend(pairID+ch for ch in ['A','B'])

    print 'discarded', unable,'examples.'
  embeddings.insert(0,np.zeros_like(embeddings[0]))
  vocabulary.insert(0,'UNKNOWN')
  return allData, embeddings, vocabulary, list(labels)

def train(theta, allData, hyperParams):
  batchsize = hyperParams['bSize']
  if hyperParams['ada']: histGrad = theta.gradient()
  else: histGrad = None
  examples = allData['train'].keys()
  print len(examples)
  classifier = Classifier(arity, labels)
  for epoch in range(5):
    print '\tIteration',epoch
    # randomly split the data into parts of batchsize
    random.shuffle(examples)
    trainLoss = 0
    # train
    nBatches = (len(examples)+batchsize-1)//batchsize
    print nBatches
    for batch in xrange(nBatches):
      minibatch = [(i, allData['train'][i]) for i in examples[batch*batchsize:(batch+1)*batchsize]]
      gradient, avError = trainBatch(minibatch, theta, hyperParams['lambda'])
      trainLoss += avError
#      theta.regularize(hyperParams['alpha']/len(examples), hyperParams['lambda'])
      theta.add2Theta(gradient, hyperParams['alpha'], histGrad)
      if batch%100 == 0:
        print '\t\tBatch', batch, ', average error:',avError , ', theta norm:', theta.norm()
    # evaluate
    print '\tComputing performance ('+str(len(allData['dev']))+' examples)...'
    error = 0
    for pairID, (ts, gold_label) in allData['dev'].iteritems():
      error += classifier.evaluate(theta,[pairID+'A', pairID+'B'], gold_label)
    print '\tTraining error:', trainLoss/(nBatches), ', Estimated performance:', error/len(allData['dev'])



def trainBatch(tData, theta, lambdaL2):
  classifier = Classifier(arity, labels)
  grads = theta.gradient()
  error = 0
  for pairID, (ts, gold_label) in tData:
    classifier.setInputs([pairID+'A', pairID+'B'])
    error += classifier.train(theta,grads, True, gold_label)
  grads /= len(tData)
  return grads, error/len(tData)

def classifyInference(thetaFile):
  global labels
  allData, embeddings, vocabulary, labels = install(thetaFile)
  dims = {'comparison':75}
  dims['din']=len(embeddings[0])
  dims['nClasses']=len(labels)
  global arity
  arity = len(allData['train'].values()[0])
  dims['arity']= arity
  hyperParams={'bSize':50,'lambda': 0.00001,'alpha':0.01,'ada':True}

  theta = myTheta.Theta('classifier', dims, None, embeddings,  vocabulary)


  train(theta, allData,hyperParams)

