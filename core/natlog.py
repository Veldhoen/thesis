from classifier import *
import sys
import numpy as np
import pickle
import nltk, re
import random
import os
import myRNN


def getFromIFile(fromFile):
  data = []
  with open(fromFile, 'r') as f:
    header = next(f)
    i = 0
    for line in f:
      i+=1
      if i>1000: break
      try: gold_label, s1_parse, s2_parse = line.split('\t')
      except:
        print len( line.split('\t'))
        print line
        sys.exit()


      parses = ['('+re.sub(r"([^()\s]+)", r"(W \1)", p.strip())+')' for p in s1_parse, s2_parse]
      #parses = ['('+p.strip()+')' for p in s1_parse, s2_parse]

      data.append((parses, gold_label))
  return data

def install(source):
  rawdata = []

  if os.path.isdir(source):
    toOpen = [os.path.join(source,f) for f in os.listdir(source)]
    toOpen = [f for f in toOpen if os.path.isfile(f)]

  for f in toOpen:
    rawdata.extend(getFromIFile(f))



  vocabulary = ['UNKNOWN','all','growl','lt_three','lt_two','mammals','most','move','no','not','not_all','not_most','pets','reptiles','some','swim','three','turtles','two','walk','warthogs']

  dims = {'inside': 25, 'word':25,'maxArity':2,'arity':len(rawdata[0][0])}
  theta = myTheta.Theta('RNN', dims, None, None,  vocabulary)

  labels = set()
  data = []
  for i in range(len(rawdata)):
    parses, gold_label = rawdata[i]
    labels.add(gold_label)
    ts = [ nltk.Tree.fromstring(p.lower()) for p in parses]
    data.append(([myRNN.RNN(t) for t in ts],gold_label))

  random.shuffle(data)
  allData = {}
  p10 = len(data)//10
  allData['train']=dict(zip(range(p10*8),data[:p10*8]))
  allData['dev']=dict(zip(range(p10),data[p10*8:p10*9]))
  allData['test']=dict(zip(range(p10),data[p10*9:]))

  dims['comparison']=75
  dims['nClasses']=len(labels)
  thetaC = myTheta.Theta('classifier', dims, None, None,  vocabulary)
  for key, value in thetaC.items():
    theta.newMatrix(key, value)

  return theta, allData, list(labels)
