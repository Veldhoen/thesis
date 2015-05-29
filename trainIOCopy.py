from __future__ import division
from IORNN import *
#from training import *
from trainingParallel import *
#from params import *
import pickle
import argparse
import naturalLogicCopy as naturalLogic
import myTheta

def main(args):
  # get treebank
  source = args['trees']
  if os.path.isdir(source): toOpen = [f for f in [os.path.join(source,f) for f in os.listdir(source)] if os.path.isfile(f)]
  elif os.path.isfile(source): toOpen = [source]
  else: 
    print source,'is not a file nor a dir'
    sys.exit()
  for f in [n for n in toOpen if 'TREES' in n]:
    with open(f, 'rb') as f:
      examples = pickle.load(f)
  # get vocabulary
  source = args['voc']
  if os.path.isdir(source): toOpen = [f for f in [os.path.join(source,f) for f in os.listdir(source)] if os.path.isfile(f)]
  elif os.path.isfile(source): toOpen = [source]
  else:
    print source,'is not a file nor a dir'
    sys.exit()

  vocabulary = set()
  for f in [n for n in toOpen if 'VOC' in n ]:
    with open(f, 'rb') as f: vocabulary.update(pickle.load(f))
  vocabulary = list(vocabulary)
  # create networks
  for kind, trees in examples.iteritems():
    for i in xrange(len(trees)):
      examples[kind][i] = naturalLogic.iornnFromTree(trees[i][0][0], vocabulary)
  print 'Loaded data.',len(examples['TRAIN']), 'training examples, and',len(examples['TEST']), 'test examples.'

  if args['pars']:
    with open(args['pars'], 'rb') as f:
      theta = pickle.load(f)
  else:
    dims = dict((k, args[k]) for k in ['inside','outside'])
    if args['emb']:
      with open(args['emb'], 'rb') as f:
        V,voc = pickle.load(f)
      V = np.vstack(tuple([V[i] for i in [voc.index(w) if w in voc else 0 for w in vocabulary]]))
      nwords = len(vocabulary)
      print 'Loaded embeddings. Vocabulary size is', nwords
      dims['word'] = len(V[0])
    else:
      V = None
      dims['word'] = args['outside']


    if not dims['inside']:  dims['inside'] = dims['word']
    if not dims['outside']:  dims['outside'] = dims['word']
    dims['nwords']=len(vocabulary)
    theta = myTheta.Theta('IORNN', dims, V)
    print 'Initialized theta.'

  hyperParams = dict((k, args[k]) for k in ['nEpochs','bSize','lambda','alpha'])
  cores = args['cores']


  print 'Starting training...'
  theta = SGD(theta, hyperParams, examples, [], cores)
  with open(os.path.join(args['out'] ), 'wb') as f:
    pickle.dump(theta, f, -1)



def evaluateNW(nw,theta):
  nwords = len(theta['wordIM'])

  for leaf in nw:
    scores = np.zeros(nwords)
    for x in xrange(nwords):
      scores[x] = nw.score(theta,x)
      ranking = scores.argsort().argsort()[leaf.index]

if __name__ == "__main__":
  parser = argparse.ArgumentParser(description='Train IORNN on a treebank')
  # data:
  parser.add_argument('-t','--trees', type=str, help='File or directory with pickled treebank', required=True)
  parser.add_argument('-v','--voc', type=str, help='File with pickled vocabulary', required=True)
  parser.add_argument('-e','--emb', type=str, help='File with pickled embeddings', required=False)
  parser.add_argument('-o','--out', type=str, help='Output file to store pickled theta', required=True)
  parser.add_argument('-p','--pars', type=str, help='File with pickled theta to initialize with', required=False)
  # network hyperparameters:
  parser.add_argument('-din','--inside', type=int, help='Dimensionality of inside representations', required=False)
  parser.add_argument('-dwrd','--outside', type=int, help='Dimensionality of leaves (word nodes)', required=False)
  parser.add_argument('-dout','--word', type=int, help='Dimensionality of outside representations', required=False)
  # training hyperparameters:
  parser.add_argument('-n','--nEpochs', type=int, help='Number of epochs to train', required=True)
  parser.add_argument('-b','--bSize', type=int, default = 0, help='Batch size for minibatch training', required=False)
  parser.add_argument('-l','--lambda', type=float, help='Regularization parameter lambdaL2', required=True)
  parser.add_argument('-a','--alpha', type=float, help='Learning rate parameter alpha', required=True)
  # computation:
  parser.add_argument('-c','--cores', type=int, default=1, help='Number of cores to use for parallel processing', required=False)
  args = vars(parser.parse_args())

  main(args)
