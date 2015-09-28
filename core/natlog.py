from classifier import *
import sys
import numpy as np
import pickle
import nltk, re
import random
import os
import myRNN



class natlogTB():
  def __init__(self,data,labels):
    self.labels = labels
    self.examples=[(Classifier(nws, labels, False),label) for nws,label in data]
    self.n = len(self.examples)
  def getExamples(self, n=0):
    if n == 0: n = self.n
    return self.examples


def getFromIFile(fromFile):
  data = []
  with open(fromFile, 'r') as f:
    header = next(f)
    i = 0
    for line in f:
      i+=1
      #if i>1000: break
      try: gold_label, s1_parse, s2_parse = line.split('\t')
      except:
        print len( line.split('\t'))
        print line
        sys.exit()
      parses = ['('+re.sub(r"([^()\s]+)", r"(W \1)", p.strip().lower())+')' for p in s1_parse, s2_parse]
      #parses = ['('+p.strip()+')' for p in s1_parse, s2_parse]
      data.append((parses, gold_label))
  return data

def process(rawdata,activation):
  vocabulary = set(['UNKNOWN'])
  labels = set()
  data = []
  for i in range(len(rawdata)):
    parses, gold_label = rawdata[i]
    labels.add(gold_label)
    ts = [ nltk.Tree.fromstring(p.lower()) for p in parses]
    [vocabulary.update(t.leaves()) for t in ts]
    data.append(([myRNN.RNN(t, activation).root for t in ts],gold_label))
  return vocabulary, labels, data

def install(source, kind = 'RNN', d=0, additive = False):
  print 'additive:', additive

  if d == 0: d = 16

  if not os.path.isdir(source):
      print 'no src:', source
      sys.exit()



  if os.path.isdir(source):
    toOpen = [os.path.join(source,f) for f in os.listdir(source)]
    toOpen = [f for f in toOpen if os.path.isfile(f)]

  vocabulary = set(['UNKNOWN'])
  labels = set()

  if additive: activation = 'identity'
  else: activation = 'tanh'

  if len(toOpen)>2:
    trainData = []
    testData =[]

    for f in toOpen:
      voc, lab, dat = process(getFromIFile(f), activation)
      vocabulary.update(voc)
      labels.update(lab)
      random.shuffle(dat)
      p10 = len(dat)//20
      trainData.extend(dat[:p10*17])
      testData.extend(dat[p10*17:])
    labels=list(labels)
    ttb = natlogTB(trainData,labels)
    dtb = natlogTB(testData,labels)
  else:
    data = []
    for f in sorted(toOpen):
      print f
      voc, lab, dat = process(getFromIFile(f), activation)
      vocabulary.update(voc)
      labels.update(lab)
      data.append(dat)

    labels=list(labels)
    dtb = natlogTB(data[0],labels)
    ttb = natlogTB(data[1],labels)



  vocabulary = list(vocabulary)
  dims = {'inside': d, 'word':d,'maxArity':2,'arity':2}
  theta = myTheta.Theta(kind, dims, None, None,  vocabulary=vocabulary)
                      #    style, dims, grammar, embeddings = None,  vocabulary = ['UNKNOWN'])
  if additive: theta.additiveComposition()
#   allData = {}
#   p10 = len(data)//10
#   allData['train']=dict(zip(range(p10*8),data[:p10*8]))
#   allData['dev']=dict(zip(range(p10),data[p10*8:p10*9]))
#   allData['test']=dict(zip(range(p10),data[p10*9:]))

  dims['comparison']=3*d
  dims['nClasses']=len(labels)
  thetaC = myTheta.Theta('classifier', dims, None, None,  vocabulary)
  for key, value in thetaC.items():
    theta.newMatrix(key, value)

  return theta, ttb,dtb
