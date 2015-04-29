import pickle, os
import nltk, re
import numpy as np
import random

def getSennaEmbs(source, voc = ['UNK']):
  print '\tObtaining embeddings...'
  getVoc = (voc == ['UNK'])
  L = [0]*len(voc)
  with open(source+'/words.lst','r') as words:
    with open(source+'/embeddings.txt','r') as embs:
      for w,e in zip(words,embs):
        word = w.strip().strip('\'')
        emb = [float(e) for e in e.strip().split()]
#        print word, emb
        if getVoc:
          voc.append(word)
          L.append(emb)
        else:
          if w in voc: L[voc.index(word)] = emb
  d = len(emb)
  for i in range(len(L)):
    if L[i]==0: L[i] = np.random.rand(d)*0.02-0.01
  V = np.array(L)
  with open(os.path.join('data','sennaV.pik'), 'wb') as f:
    print 'opened file'
    pickle.dump([voc], f, -1)
#    pickle.dump([V,voc], f, -1)
    print 'dumped info' 
  print 'file should be closed now'



def artData(name):
  source = os.path.join('data',name)
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

          examples.append(([nltk.tree.Tree.fromstring('('+re.sub(r"([^()\s]+)", r"(W \1)", s)+')') for s in [s1,s2]],relation))
  random.shuffle(examples)
  nTest = len(examples)//5
  trainData = examples[:4*nTest]
  testData = examples[4*nTest:]
  trialData = []

  with open(os.path.join('data',name+'.pik'), 'wb') as f:
    pickle.dump([trainData, testData, trialData, vocabulary], f, -1)

def sickData(name):
  source = os.path.join('data',name)
  # make a list of files to open
  if os.path.isdir(source):
    toOpen = [os.path.join(source,f) for f in os.listdir(source)]
    toOpen = [f for f in toOpen if os.path.isfile(f)]
  elif os.path.isfile(source): toOpen = [source]
  examples = {'TRAIN':[],'TEST':[],'TRIAL':[]}
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
        examples[kind].append(([nltk.tree.Tree.fromstring('('+re.sub(r"([^()\s]+)", r"(W \1)", s)+')') for s in [s1,s2]],relation))
  with open(os.path.join('data',name+'.pik'), 'wb') as f:
    pickle.dump([examples['TRAIN'],examples['TEST'],examples['TRIAL'],vocabulary], f, -1)


name = 'bowman14'
artData(name)

name = 'bowman15'
artData(name)


# source = 'data/senna'
# getSennaEmbs(source)
# with open('data/sennaV.pik', 'wb') as f:
#   V, voc =   pickle.load(f)
# 
# for i in range(10):
#   print voc[i], V[i]


# name = 'sickSample'
# #sickData(name)
# #source = os.path.join('data',name)
# with open(os.path.join('data',name+'.pik'), 'rb') as f:
#     trainData, testData, trialData, vocabulary = pickle.load(f)
# print len(trainData), len(testData), len(trialData), len(vocabulary)
#for e in testData: print e[0][0], e[0][1], e[1]

