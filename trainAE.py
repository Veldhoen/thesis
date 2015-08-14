import argparse
import core.myRAE as myRAE
import core.myTheta as myTheta
import core.trainingRoutines as training
from collections import defaultdict, Counter
import sys, os, pickle
import numpy as np



def getVocabulary(vocabularies):
  print 'Loading vocabulary.'
  vocabulary = set()
  for m in vocabularies:
    with open(m, 'rb') as f:
      vocabulary.update(pickle.load(f))
  vocabulary = list(vocabulary)
  print '\tThere are',len(vocabulary),'words in the combined treebanks.'
  try: vocabulary.pop(vocabulary.index('UNKNOWN'))
  except: True
  vocabulary.insert(0,'UNKNOWN')
  return vocabulary

def getGrammar(option, grammars):
  rules = defaultdict(Counter)
  print 'Grammar-based parameter selection:', option
  if len(grammars)<1:
    print 'No RULES.pik file found. Exit program'
    sys.exit()
  for m in grammars:
    with open(m, 'rb') as f:
      newRules =pickle.load(f)
      for lhs, rhss in newRules.iteritems():
        for rhs, count in rhss.iteritems():
          rules[lhs][rhs]+= count
  return rules

def initializeTheta(args,vocabulary, grammar,maxArity):
  print 'Initializing theta.'
  if args['pars']:
    with open(args['pars'], 'rb') as f:
      theta = pickle.load(f)
    print 'Retrieved Theta from disk.'
  else:
    dims = dict((k, args[k]) for k in ['inside','outside'])
    dims['maxArity']=maxArity
    if args['emb']:
      with open(args['emb'], 'rb') as f:
        V,voc = pickle.load(f)
      if 'UNKNOWN' not in voc: voc.insert('UNKNOWN',0)
      vocabulary = [w for w in vocabulary if w in voc]

      V = np.vstack(tuple([V[i] for i in [voc.index(w) for w in vocabulary]]))
      dims['word'] = len(V[0])
    else:
      V = None
      dims['word'] = args['word']
      if dims['word'] is None:
        print 'Either embeddings or dword must be specified. Stop program.'
        sys.exit()
    if not dims['inside']:  dims['inside'] = dims['word']
    if not dims['outside']:  dims['outside'] = dims['word']
    dims['nwords']=len(vocabulary)
    theta = myTheta.Theta('RAE', dims, grammar, V, vocabulary)

  theta.printDims()
  return theta

def main(args):
  print 'Start (part of) experiment '+ args['experiment']

  source = args['sourceTrain']
  if os.path.isdir(source):
    files = [f for f in [os.path.join(source,f) for f in os.listdir(source)] if os.path.isfile(f)]
    treebanksTrain = [f for f in files if 'RAES' in f]
    vocabularies = [f for f in files if 'VOC.pik' in f]
    grammars = [f for f in files if 'RULES.pik' in f]
  else:
    print 'no valid source directory:',source
    sys.exit()

  source = args['sourceValid']
  if os.path.isdir(source):
    files = [f for f in [os.path.join(source,f) for f in os.listdir(source)] if os.path.isfile(f)]
    treebanksValid = [f for f in files if 'RAES' in f]
    vocabularies.extend([f for f in files if 'VOC.pik' in f])
    grammars.extend([f for f in files if 'RULES.pik' in f])
  else:
    print 'no valid source directory:',source
    sys.exit()

  if len(vocabularies)<2:
    print 'no two vocabulary files.'
    sys.exit()

  if len(grammars)<2:
    print 'no two grammar files.'
    sys.exit()


  if len(treebanksTrain)<1 or len(treebanksValid)<1:
    print 'no training or validation data obtained. Abort execution.'
    sys.exit()

  vocabulary = getVocabulary(vocabularies)

  style =args['grammar'][0]
  grammar = getGrammar(style, grammars)
  maxArity=6

  theta=initializeTheta(args,vocabulary, grammar,maxArity)

  hyperParams = dict((k, args[k]) for k in ['nEpochs','bSize','lambda','alpha'])
  cores = max(1,args['cores']-1) # keep one core free for optimal efficiency


  if len(args['grammar'])>1: hyperParams['nRules']=args['grammar'][1]
  else: hyperParams['nRules']=200


  print 'Hyper parameters:'
  for param, value in hyperParams.iteritems():
    print '\t',param, '-' ,value


  print '\tnumber of cores -', cores

  ada = True
  if ada: print 'Adagrad is on.'
  else: print 'Adagrad is off.'

  outDir = args['out']
  if not os.path.isdir(outDir):
    print 'Not a valid output directory:', outDir
    sys.exit()


  tTreebank = training.Treebank(treebanksTrain,maxArity)
  vTreebank = training.Treebank(treebanksValid[:1],maxArity)
  print vTreebank.files


  training.storeTheta(theta, os.path.join(outDir,'initialTheta.pik'))
  # training...

  if style == 'beginSmall': training.beginSmall(tTreebank, vTreebank, hyperParams, ada, theta, outDir, cores)
  elif style == 'None': training.plainTrain(tTreebank, vTreebank, hyperParams, ada, theta, outDir, cores)
  elif style == 'LHS':
    theta.specializeHeads()
    training.plainTrain(tTreebank, vTreebank, hyperParams, ada, theta, outDir, cores)
  elif style == 'Rules':
    theta.specializeRules(hyperParams['nRules'])
    training.plainTrain(tTreebank, vTreebank, hyperParams, ada, theta, outDir, cores)








class ValidateGrammar(argparse.Action):
  def __call__(self, parser, args, values, option_string=None):
    valid_subjects = ['None','LHS','Rules','beginSmall']
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
  parser.add_argument('-st','--sourceTrain', type=str, help='Directory with pickled treebank(s), grammar(s) and vocabulary(s) for training', required=True)
  parser.add_argument('-sv','--sourceValid', type=str, help='Directory with pickled treebank(s), grammar(s) and vocabulary(s) for validation', required=True)
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
  parser.add_argument('-n','--nEpochs', type=int, help='Maximal number of epochs to train per phase', required=True)
  parser.add_argument('-b','--bSize', type=int, default = 50, help='Batch size for minibatch training', required=False)
  parser.add_argument('-l','--lambda', type=float, help='Regularization parameter lambdaL2', required=True)
  parser.add_argument('-a','--alpha', type=float, help='Learning rate parameter alpha', required=True)
  # computation:
  parser.add_argument('-c','--cores', type=int, default=1, help='Number of cores to use for parallel processing', required=False)
  args = vars(parser.parse_args())

  main(args)


