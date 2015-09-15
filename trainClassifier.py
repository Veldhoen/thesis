from __future__ import division
import core.classifier as cl
import sys,os
import core.trainingRoutines as tr
import core.SNLI as SNLI
import core.natlog as natlog
from collections import defaultdict, Counter
import random


def confusionS(matrix):
  s = ''
  for label in labels:
    s+='\t'+label
  s+='\n'
  for t in labels:
    s+= t
    for p in labels:
      s+= '\t'+str(matrix[t][p])
    s+='\n'
  return s


def evaluate(classifier,testData,theta):
  error = 0
  true = 0
  confusion = defaultdict(Counter)

  for pairID, (ts, gold_label) in testData.iteritems():
    if fixed:
      error += classifier.evaluate(theta,[pairID+'A', pairID+'B'], gold_label, True)
      prediction= classifier.predict(theta,[pairID+'A', pairID+'B'], True,False)
    else: 
      error += classifier.evaluate(theta,ts, gold_label, False)
      prediction = classifier.predict(theta,ts, False,False)
    confusion[gold_label][prediction] += 1
    if prediction == gold_label:
      true +=1
  accuracy = true/len(testData)
  loss = error/len(testData)
  return loss, accuracy, confusion

def train(theta, allData, hyperParams):
  batchsize = hyperParams['bSize']
  if hyperParams['ada']: histGrad = theta.gradient()
  else: histGrad = None
  examples = allData['train'].keys()
  classifier = cl.Classifier(theta.dims['arity'], labels, fixed)
  print '\tComputing performance ('+str(len(allData['dev']))+' examples)...'
  error = 0
  loss, accuracy, confusion =  evaluate(classifier,allData['dev'],theta)
  print '\tInitial training error: - , Estimated performance:',loss,', Accuracy:',accuracy, ', Confusion:'
  print confusionS(confusion)

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
      gradient, avError = trainBatch(classifier, minibatch, theta, hyperParams['lambda'])
      trainLoss += avError
#      theta.regularize(hyperParams['alpha']/len(examples), hyperParams['lambda'])
      theta.add2Theta(gradient, hyperParams['alpha'], histGrad)
      if batch%100 == 0:
        print '\t\tBatch', batch, ', average error:',avError , ', theta norm:', theta.norm()
    # evaluate
    print '\tComputing performance ('+str(len(allData['dev']))+' examples)...'
    error = 0
    loss, accuracy, confusion =  evaluate(classifier,allData['dev'],theta)
    print '\tTraining error:', trainLoss/(nBatches), ', Estimated performance:',loss,', Accuracy:',accuracy, ', Confusion:'
    print confusionS(confusion)



def trainBatch(classifier, tData, theta, lambdaL2):
  grads = theta.gradient()
  error = 0
  for pairID, (ts, gold_label) in tData:
    if isinstance(ts[0],str): classifier.replaceChildren([pairID+'A', pairID+'B'], True)
    else: classifier.replaceChildren(ts, False)
    error += classifier.train(theta,grads, True, gold_label)
  grads /= len(tData)
  return grads, error/len(tData)


def main(thetaFile,src, outFile):
  if not os.path.isfile(thetaFile):
    print 'no file containing theta:', thetaFile
#    sys.exit()
  if not os.path.isdir(src):
    print 'no src:', src
    sys.exit()
  
  global labels, fixed
#  theta, allData, labels = SNLI.install(thetaFile,src)     \
#  fixed = True
  theta, allData, labels = natlog.install(src)
  fixed = False

  hyperParams={'bSize':50,'lambda': 0.00001,'alpha':0.01,'ada':True}
  train(theta, allData, hyperParams)
  tr.storeTheta(theta, outFile)



if __name__ == "__main__": main(sys.argv[1],sys.argv[2],sys.argv[3])