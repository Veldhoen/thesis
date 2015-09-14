from classifier import *
import sys
import numpy as np
import pickle
import nltk, re
# import random
import os


def getFromIFile(fromFile):
  data = {}
  with open(fromFile, 'r') as f:
    header = next(f)
    i = 0
    for line in f:
      i+=1
      if i>100: break
      try: gold_label, s1_binParse, s2_binParse, s1_parse, s2_parse, sentence1, sentence2, captionID, pairID, l0,l1,l2,l3,l4 = line.split('\t')
      except:
        print len( line.split('\t'))
        sys.exit()
      data[pairID] = ([s1_parse,s2_parse], gold_label)
  return data

def install(thetaFile,src):
  allData = {}
  for dset in ['train','dev','test']:
    allData[dset] = getFromIFile(os.path.join(src,dset+'.txt'))

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
  dims = {'comparison':75}
  dims['inside']=len(embeddings[0])
  dims['nClasses']=len(labels)
  dims['arity']= len(allData['train'].values()[0])

  theta = myTheta.Theta('classifier', dims, None, embeddings,  vocabulary)

  return theta, allData, list(labels)
