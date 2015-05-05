from __future__ import division
from IORNN import *
from training import *
from params import *
import pickle

import naturalLogic


# def types(d, nwords):
#   types = []
#   types.append(('compositionIM','float64',(dint,2*dint)))
#   types.append(('compositionIB','float64',(dint)))
#   types.append(('compositionOM','float64',(dint,2*dint)))
#   types.append(('compositionOB','float64',(dint)))
#   types.append(('wordIM','float64',(nwords,dwords)))
#   types.append(('wordOM', 'float64',(2*dint,2*dint)))
#   types.append(('wordOB', 'float64',(2*dint)))
#   types.append(('uOM', 'float64',(1,2*dint)))
#   types.append(('uOB', 'float64',(1,1))) #matrix with one value, a 1-D array with only one value is a float and that's problematic with indexing
#   return types
# 
# def initialize(style, d, nwords = 1, V = None):
#   types = types(dint, nwords)
#   # initialize all parameters randomly using a uniform distribution over [-0.1,0.1]
#   theta = np.zeros(1,dtype = types)
#   for name, t, size in types:
#     if name == 'wordIM': theta[name]=V
#     elif isinstance(size, (int,long)): theta[name] = np.random.rand(size)*.02-.01
#     elif len(size) == 2: theta[name] = np.random.rand(size[0],size[1])*.02-.01
#     else: print 'invalid size:', size
#   return theta[0]

def main():
  source = 'data/flickr.pik'
  with open(source, 'rb') as f:
    examples, vocabulary = pickle.load(f)
  for kind, trees in examples.iteritems():
    for i in range(len(trees)):
#      print trees[i]
      examples[kind][i] = naturalLogic.iornnFromTree(trees[i][0], vocabulary)
  print 'loaded data'

  source = 'data/senna.pik'
  with open(source, 'rb') as f:
    V,voc = pickle.load(f)
  print 'loaded embeddings'

  V = np.vstack(tuple([V[i] for i in [voc.index(w) for w in vocabulary]])
  print 'removed unnecessary embeddings'

  theta = naturalLogic.initialize('IORNNUS', dwords, dint, dcomp, 0, len(voc), V)
  print 'initialized theta'

  print 'starting training...'
  bowmanSGD(lambdaL2, alpha, epochs, theta, examples['TRAIN'], examples['TEST'], relations, batchsize)
  with open(os.path.join('models','flickrIO.pik'), 'wb') as f:
    pickle.dump(theta, f, -1)


def evaluateNW(nw,theta):
  nwords = len(theta['wordIM'])
  
  for leaf in nw:
    scores = np.zeros(nwords)
    for x in range(nwords):
      scores[x] = nw.score(theta,x)
      rank = scores.argsort().argsort()[leaf.index]
main()