from __future__ import division
from IORNN import *
from training import *
from params import *
import pickle

import naturalLogic

def main():
#  source = 'data/sickSample.pik'
  source = 'data/flickr.pik'
  with open(source, 'rb') as f:
    examples, vocabulary = pickle.load(f)
  for kind, trees in examples.iteritems():
    for i in xrange(len(trees)):
#      print trees[i]
      examples[kind][i] = naturalLogic.iornnFromTree(trees[i][0][0], vocabulary)
  print 'Loaded data.',len(examples['TRAIN']), 'training examples, and',len(examples['TEST']), 'test examples.'

  source = 'data/senna.pik'
  with open(source, 'rb') as f:
    V,voc = pickle.load(f)
  V = np.vstack(tuple([V[i] for i in [voc.index(w) if w in voc else 0 for w in vocabulary]]))
  nwords = len(vocabulary)
  print 'Loaded embeddings. Vocabulary size is', nwords

  dwords = len(V[0])
  dint = dwords
  theta = naturalLogic.initialize('IORNN', dwords, dint, dcomp, 0, nwords, V)
  print 'Initialized theta.'

  print 'Starting training...'
  SGD(lambdaL2, alpha, epochs, theta, examples['TRAIN'], examples['TEST'], [], bsize)
  #bowmanSGD(lambdaL2, alpha, epochs, theta, examples['TRAIN'], examples['TEST'], [], bsize)
  with open(os.path.join('models','flickrIO.pik'), 'wb') as f:
    pickle.dump(theta, f, -1)


def evaluateNW(nw,theta):
  nwords = len(theta['wordIM'])

  for leaf in nw:
    scores = np.zeros(nwords)
    for x in xrange(nwords):
      scores[x] = nw.score(theta,x)
      rank = scores.argsort().argsort()[leaf.index]
main()