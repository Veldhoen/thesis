import numpy as np
import os
from nltk.tree import Tree
import nltk
import NLRNN
from naturalLogic import *


def loadVocabulary(inFile):
  vocabulary = ['UNK']
  with open(inFile, 'r') as f:
    for line in f:
      vocabulary.append(line.strip())
  return vocabulary 

def initializeWordMatrix(vocabulary, embedFiles = None, d = 16):
  if embedFiles is None:
    V = np.random.rand(len(vocabulary),d)*.02-.01
  else:

    embs, wordList = embedFiles
    
    indexes = [0]*len(vocabulary)
    with open(wordList, 'r') as f:
      i = 0
      for line in f:
        w = line.strip()
        if w in vocabulary:
          indexes[vocabulary.index(w)] = i
        i +=1

    with open(embs, 'r') as f:
      first = f.readline()
      embedding = np.array([float(v) for v in first.split()])
      d = len(embedding)
      V = np.zeros((len(vocabulary), d))
      V[0] = np.random.rand(d)*.02-.01 # a random vector for 'UNK'
      V[1] = embedding
      i = 1
      for line in f:
        if i in indexes:
          embedding = np.array([float(v) for v in line.split()])
          V[indexes.index(i)] = embedding
        i +=1
  return V

def getData(inFile, relations, vocabulary):
  train=[]
  test=[]
  trial=[]
  with open(inFile, 'r') as f:
    heading = f.readline()
    for line in f:
      parts = line.split('\t')

      # obtain datapoint
      parse_A = parts[2]
      tree_A = Tree.fromstring(parse_A)
      parse_B = parts[4]
      tree_B = Tree.fromstring(parse_B)
      # binarize the trees and collapse unary productions
      [nltk.treetransforms.chomsky_normal_form(t) for t in [tree_A, tree_B]]
      [nltk.treetransforms.collapse_unary(t) for t in [tree_A, tree_B]]
      #create network
      nw = NLRNN.Network.fromTrees([tree_A, tree_B],vocabulary)

      #set label
      entailment = parts[5]
      entailment_AB = parts[7].split('_')[1]
      entailment_BA = parts[8].split('_')[1]
      relatedness = parts[6]
      target = relations.index(entailment)
      targets = (entailment,entailment_AB,entailment_BA,relatedness)

      # add example to appropriate set
      kind = parts[13].strip()
      if kind == 'TRAIN': train.append((nw,target))
      elif kind == 'TEST': test.append((nw,target))
      elif kind == 'TRIAL': trial.append((nw,target))
      else: print 'non kind!', kind#line
  return train, test, trial

def main(args):
  if len(args)== 0: datadir = 'C:/Users/Sara/AI/thesisData'
  else: datadir = args[0]
  # set paths
  vocFile = os.path.join(datadir,'SICK/vocabulary.txt')
  embedFiles = [os.path.join(datadir, spec) for spec in ['senna/embeddings/embeddings.txt', 'senna/hash/words.lst']]
  corpusFile = os.path.join(datadir,'SICK/corpus.txt')
  relations = ['NEUTRAL','CONTRADICTION','ENTAILMENT']



  print 'Initializing vocabulary and parameters...'
  vocabulary = loadVocabulary(vocFile)
  V = initializeWordMatrix(vocabulary, embedFiles)
  trainData,testData,trialData = getData(corpusFile, relations, vocabulary)



  dcomparison = 45
  dint = 16
  nwords, dwords = np.shape(V)
  [[nw[0].sethyperparams(nwords, dwords, dint, dcomparison, len(relations)) for nw in set] for set in [trainData,testData,trialData]]


  theta = initialize(nwords,dint, dcomparison, len(relations), V)
  print 'Done.', nwords, 'words of dimensionality', dwords, '. Comparison layer has', dcomparison,'nodes.'


  alpha = 0.2
  lambdaL2 = 0.0002
  epochs = 5
  thetaSGD = SGD(lambdaL2, alpha, epochs, np.copy(theta), trainData)
  evaluate(thetaSGD,testData)


if __name__ == "__main__":
   main(sys.argv[1:])