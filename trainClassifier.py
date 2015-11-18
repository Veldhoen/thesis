from __future__ import division
import core.classifier as cl
import sys,os
import core.trainingRoutines as tr
import core.SNLI as SNLI
import core.natlog as natlog
import core.math as math
from collections import defaultdict, Counter
import random
import argparse


def confusionS(matrix):
  if len(labels)<15:
    s = ''
    for label in labels:
      s+='\t'+label
    s+='\n'
    for t in labels:
      s+= t
      for p in labels:
        s+= '\t'+str(matrix[t][p])
      s+='\n'

  else: #compacter representations
    s = 'target: (prediction,times)\n'
    for t,ps in matrix.items():
      s+=str(t)+':'
      for p, v in ps.items():
        s+= ' ('+p+','+str(matrix[t][p])+')'
      s+='\n'
  return s


def evaluate(classifier,testData,theta,verbose = False):
  print '\tComputing performance ('+str(len(testData))+' examples)...'
#  sys.exit()
  error = 0
  true = 0
  confusion = defaultdict(Counter)
  answers = defaultdict(Counter)
  for pairID, (ts, gold_label) in testData.iteritems():
    if fixed:
#      error += classifier.evaluate(theta,[pairID+'A', pairID+'B'], gold_label, True)
#      prediction= classifier.predict(theta,[pairID+'A', pairID+'B'], True,False)

      error += classifier.evaluate(theta,[pairID], gold_label, True)
      prediction= classifier.predict(theta,[pairID], True,False)
    else:
      error += classifier.evaluate(theta,ts, gold_label, False)
      prediction = classifier.predict(theta,ts, False,False)
    confusion[gold_label][prediction] += 1
    if prediction == gold_label:
      true +=1
#       answers[str(ts[1])][True]+=1
#     elif verbose:
#       print 'wrong:', ts, gold_label
#       answers[str(ts[1])][False]+=1
  accuracy = true/len(testData)
  loss = error/len(testData)


#   if verbose:
#     print [(answers[str(i)][True],answers[str(i)][False]) for i in range(-60,60)]
  return loss, accuracy, confusion

def train(theta, allData, hyperParams):
  batchsize = hyperParams['bSize']
  if hyperParams['ada']: histGrad = theta.gradient()
  else: histGrad = None
  examples = []
  for e in allData['train'].values():
#    print e[0], e[1]
    examples.append((cl.Classifier(e[0], labels, False),e[1]))

#  examples = [(cl.Classifier([e[0]], labels, False),e[1]) for e in allData['train'].values()]
  classifier = cl.Classifier(theta.dims['arity'], labels, fixed)
#  error = 0
  loss, accuracy, confusion =  evaluate(classifier,allData['dev'],theta)
  print '\tInitial training error: - , Estimated performance:',loss,', Accuracy:',accuracy, ', Confusion:'
  print confusionS(confusion)

  for epoch in range(hyperParams['nEpochs']):
    print '\tIteration',epoch,',', len(examples), 'examples.'
    # randomly split the data into parts of batchsize
    random.shuffle(examples)


    trainLoss = tr.trainOnSet(hyperParams, examples, theta, histGrad, cores=1)
#
#     trainLoss = 0
#     # train
#     nBatches = (len(examples)+batchsize-1)//batchsize
#     print nBatches
#     for batch in xrange(nBatches):
#       minibatch = [(i, allData['train'][i]) for i in examples[batch*batchsize:(batch+1)*batchsize]]
#       gradient, avError = trainBatch(classifier, minibatch, theta, hyperParams['lambda'])
#       trainLoss += avError
#       theta.regularize(hyperParams['alpha']/len(examples), hyperParams['lambda'])
#       theta.add2Theta(gradient, hyperParams['alpha'], histGrad)
#       if batch%100 == 0:
#         print '\t\tBatch', batch, ', average error:',avError , ', theta norm:', theta.norm()
#     # evaluate
#     print '\tComputing performance ('+str(len(allData['dev']))+' examples)...'
#     error = 0
    loss, accuracy, confusion =  evaluate(classifier,allData['dev'],theta)
    print '\tTraining error:', trainLoss, ', Estimated performance:',loss,', Accuracy:',accuracy, ', Confusion:'
    print confusionS(confusion)
 # print confusionS(confusion)
  #loss, accuracy, confusion =  evaluate(classifier,allData['dev'],theta, True)



def trainBatch(classifier, tData, theta, lambdaL2):
  grads = theta.gradient()
  error = 0
  for pairID, (ts, gold_label) in tData:

#    if fixed: classifier.replaceChildren([pairID+'A', pairID+'B'], True)
    if fixed: classifier.replaceChildren([pairID], True)
    else: classifier.replaceChildren(ts, False)
    error += classifier.train(theta,grads, True, gold_label, False)
#    print classifier, gold_label
  grads /= len(tData)
  return grads, error/len(tData)


def main(args):
  global labels, fixed
  if args['kind'] == 'snli':
    thetaFile = args['pars']
    src = args['src']
    if not os.path.isdir(src):
      print 'no src:', src
      sys.exit()

    if not os.path.isfile(thetaFile):
      print 'no file containing theta:', thetaFile
      sys.exit()
    theta, allData, labels = SNLI.install(thetaFile,src)
    fixed = True

  elif args['kind'] == 'natlog':
    src = args['src'] 
    if not os.path.isdir(src):
      print 'no src:', src
      sys.exit()
    theta, allData, labels = natlog.install(src)
    fixed = False
  elif args['kind'] == 'math':
    thetaFile = args['pars']
    if not os.path.isfile(thetaFile):
      print 'no file containing theta:', thetaFile
     # sys.exit()
    theta, allData, labels = math.install(thetaFile)
    fixed = True

  hyperParams={k:args[k] for k in ['bSize','lambda','alpha','ada','nEpochs']}
  train(theta, allData, hyperParams)
#  tr.plainTrain(allData['train'], allData['dev'], hyperParams, theta, args['out'], cores=1)
#  tr.storeTheta(theta, args['out'])

def mybool(string):
  if string in ['F', 'f', 'false', 'False']: return False
  if string in ['T', 't', 'true', 'True']: return True
  raise Exception('Not a valid choice for arg: '+string)

if __name__ == "__main__":
  parser = argparse.ArgumentParser(description='Train classifier')
 # data:
  parser.add_argument('-exp','--experiment', type=str, help='Identifier of the experiment', required=True)
  parser.add_argument('-m','--model', choices=['RNN','IORNN','RAE'], default='RNN', required=False)
  parser.add_argument('-k','--kind', choices=['natlog','snli','math'], required=True)
  parser.add_argument('-s','--src', type=str, help='Directory with training data', required=False)
  parser.add_argument('-o','--out', type=str, help='Output file to store pickled theta', required=True)
  parser.add_argument('-p','--pars', type=str, default='', help='File with pickled theta', required=False)
  # network hyperparameters:
  parser.add_argument('-din','--inside', type=int, help='Dimensionality of inside representations', required=False)
  parser.add_argument('-dwrd','--word', type=int, help='Dimensionality of leaves (word nodes)', required=False)
  parser.add_argument('-dout','--outside', type=int, help='Dimensionality of outside representations', required=False)
  # training hyperparameters:
  parser.add_argument('-n','--nEpochs', type=int, help='Maximal number of epochs to train per phase', required=True)
  parser.add_argument('-b','--bSize', type=int, default = 50, help='Batch size for minibatch training', required=False)
  parser.add_argument('-l','--lambda', type=float, help='Regularization parameter lambdaL2', required=True)
  parser.add_argument('-a','--alpha', type=float, help='Learning rate parameter alpha', required=True)
  parser.add_argument('-ada','--ada', type=mybool, help='Whether adagrad is used', required=True)
  args = vars(parser.parse_args())

  main(args)

