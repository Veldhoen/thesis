from __future__ import division
from IORNN import *
#from training import *
from trainingParallel import *
#from params import *
import pickle
import argparse
import naturalLogicCopy as naturalLogic
import myTheta
from collections import defaultdict, Counter

def getGrammar(grammars,option):
#  print 'option:', option
  if option[0] == 'None':
    print 'Grammar-based parameter selection off.'
    grammar = None
  else:
    if len(option)>1: n=option[1]
    else: n=0


    rules = defaultdict(Counter)
    if len(grammars)<1: print 'No RULES.pik file found. Grammar-based parameter selection off.'
    for m in grammars:
      with open(m, 'rb') as f:
        rules.update(pickle.load(f))

    rulesC = Counter()
    lhss = Counter()
    for LHS, RHSS in rules.iteritems():
      lhss[LHS.split('|')[0].split('+')[0]] += sum(RHSS.values())
      for RHS, count in RHSS.iteritems():
        rulesC[LHS+'->'+RHS]+=count


    if n<1: n = len(rulesC)

#    print len(lhss),'heads'
#    print list(lhss.most_common())[-10:]

    grammar= lhss.keys()

    if option[0] == 'LHS':
      print 'Grammar-based parameter selection at LHS-level.', len(lhss),'specializations.'

    elif option[0] == 'Rules':
      print 'Grammar-based parameter selection on. Initializing '+str(n)+' most frequent grammar rules + ',len(lhss),' LHS-level specializations.'
      grammar = [rule for rule, count in rulesC.most_common(n)]
      # rules is a (default)dict with the LHS as key and of Counters of RHS's
      # get the n most frequent rules:

  sys.exit()
  return grammar

def main(args):
  print 'Start (part of) experiment '+ args['experiment']

  source = args['source']
  if os.path.isdir(source):
    files = [f for f in [os.path.join(source,f) for f in os.listdir(source)] if os.path.isfile(f)]
    treebanks = [f for f in files if 'TREES.pik' in f]
    vocabularies = [f for f in files if 'VOC.pik' in f]
    grammars = [f for f in files if 'RULES.pik' in f]
  else:
    print 'no valid source directory:',source
    sys.exit()

  # get vocabulary
  print 'Loading vocabulary.'
  vocabulary = set()
  for m in vocabularies:
    with open(m, 'rb') as f:
      vocabulary.update(pickle.load(f))
  vocabulary = list(vocabulary)
  vocabulary.insert(0, vocabulary.pop(vocabulary.index('UNK')))
  print 'There are',len(vocabulary),'words.'

  # get grammar
  grammar = getGrammar(grammars, args['grammar'])

  # initialize theta
  print 'Initializing theta.'
  if args['pars']:
    with open(args['pars'], 'rb') as f:
      theta = pickle.load(f)
    print 'Retrieved Theta from disk.'
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
      dims['word'] = args['word']
    if not dims['inside']:  dims['inside'] = dims['word']
    if not dims['outside']:  dims['outside'] = dims['word']
    dims['nwords']=len(vocabulary)
    theta = myTheta.Theta('IORNN', dims, V, grammar)

    print 'Model dimensionality:'
    for dim, value in dims.iteritems():
      print '\t',dim, '-' , value
  hyperParams = dict((k, args[k]) for k in ['nEpochs','bSize','lambda','alpha'])
  cores = args['cores']
  print 'Hyper parameters:'
  for param, value in hyperParams.iteritems():
    print '\t',param, '-' ,value
  print 'number of cores -', cores

  ada = True
  if ada: print 'Adagrad is on.'
  else: print 'Adagrad is off.'

  if len(treebanks)>1:
    nEpochs = hyperparams['nEpochs']
    hyperparams['nEpochs'] = 1
    # there are multiple files with trees. Open them one at a time and do one epoch on each
    for epoch in xrange(nEpochs):
      for m in treebanks:
        print 'Loading treebank.'
        with open(m, 'rb') as f:
          examples = pickle.load(f)
        # create networks
        print 'Initializing networks.'
        for kind, trees in examples.iteritems():
          for i in xrange(len(trees)):
            examples[kind][i] = naturalLogic.iornnFromTree(trees[i][0][0], vocabulary,grammar)
        print 'Loaded data.',len(examples['TRAIN']), 'training examples, and',len(examples['TEST']), 'test examples.'
        print 'Start training...'
        theta = SGD(theta, hyperParams, examples, [], cores, adagrad = ada)
  else:
    # get treebank
    print 'Loading treebank.'
    with open(treebanks[0], 'rb') as f:
      examples = pickle.load(f)
    # create networks
    print 'Initializing networks.'
    for kind, trees in examples.iteritems():
      for i in xrange(len(trees)):
        examples[kind][i] = naturalLogic.iornnFromTree(trees[i][0][0], vocabulary,grammar)
    print 'Loaded data.',len(examples['TRAIN']), 'training examples, and',len(examples['TEST']), 'test examples.'
    print 'Start training...'
    theta = SGD(theta, hyperParams, examples, [], cores, adagrad = ada)




#
#
#   print 'Writing model to file.'
#   sentences = []
#   for nw in examples['TEST']:
#     nw.activateNW(theta)
#     sentences.append((' '.join([str(l) for l in nw.leaves()]),nw.innerA))
# #  sentences= [(str(nw),nw.innerA) for nw in examples['TEST']]
#   with open(os.path.join(args['out']+'SENTENCES.pik' ), 'wb') as f:
#     pickle.dump(sentences, f, -1)
#
#
#   with open(os.path.join(args['out']+'VOC.pik' ), 'wb') as f:
#     pickle.dump(vocabulary, f, -1)
#
#   with open(os.path.join(args['out']+'THETA.pik' ), 'wb') as f:
#     pickle.dump(theta, f, -1)

class ValidateGrammar(argparse.Action):
  def __call__(self, parser, args, values, option_string=None):
    valid_subjects = ['None','LHS','Rules']
    kind = values[0]
    if kind not in valid_subjects:
      raise ValueError('invalid grammar-option {s!r}'.format(s=kind))

    if len(values)==2:
      n = int(values[1])
    else: n = 0
    if len(values)>2:
      print '-g grammar options',values[2:], 'are ignored'
#    kind, n = values

    Credits = ('Credits', 'subject required')
    setattr(args, self.dest, (kind,n))

if __name__ == "__main__":
  parser = argparse.ArgumentParser(description='Train IORNN on a treebank')
  # data:
  parser.add_argument('-exp','--experiment', type=str, help='Identifier of the experiment', required=True)
  parser.add_argument('-s','--source', type=str, help='Directory with pickled treebank(s) and vocabulary(s)', required=True)
  parser.add_argument('-e','--emb', type=str, help='File with pickled embeddings', required=False)
  parser.add_argument('-g','--grammar', action=ValidateGrammar, help='Kind of parameter specialization', nargs='+',required=True)
#  parser.add_argument('-g','--grammar', type=str, help='File with pickled grammar', required=False)
  parser.add_argument('-o','--out', type=str, help='Output file to store pickled theta', required=True)
  parser.add_argument('-p','--pars', type=str, help='File with pickled theta to initialize with', required=False)
  # network hyperparameters:
  parser.add_argument('-din','--inside', type=int, help='Dimensionality of inside representations', required=False)
  parser.add_argument('-dwrd','--word', type=int, help='Dimensionality of leaves (word nodes)', required=False)
  parser.add_argument('-dout','--outside', type=int, help='Dimensionality of outside representations', required=False)
  # training hyperparameters:
  parser.add_argument('-n','--nEpochs', type=int, help='Number of epochs to train', required=True)
  parser.add_argument('-b','--bSize', type=int, default = 0, help='Batch size for minibatch training', required=False)
  parser.add_argument('-l','--lambda', type=float, help='Regularization parameter lambdaL2', required=True)
  parser.add_argument('-a','--alpha', type=float, help='Learning rate parameter alpha', required=True)
  # computation:
  parser.add_argument('-c','--cores', type=int, default=1, help='Number of cores to use for parallel processing', required=False)
  args = vars(parser.parse_args())

  main(args)
  

