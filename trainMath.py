from __future__ import division

from collections import defaultdict, Counter
import core.math as math
import core.natlog as natlog
import core.trainingRoutines as tr
import argparse



def confusionS(matrix,labels):
  if True:#len(labels)<15:
    s = ''
    for label in labels:
      s+='\t'+label
    s+='\n'
    for t in labels:
      s+= t
      for p in labels:
        s+= '\t'+str(matrix[t][p])
      s+='\n'

  else: #compacter representations
    s = 'target: (prediction,times)\n'
    for t,ps in matrix.items():
      s+=str(t)+':'
      for p, v in ps.items():
        s+= ' ('+p+','+str(matrix[t][p])+')'
      s+='\n'
  return s

def evaluate(ttb,theta):
  n=ttb.n
  print 'Evaluating on ',n, 'examples.'
  error = 0
  true = 0
  confusion = defaultdict(Counter)
  for nw, target in ttb.getExamples():
    error+=nw.evaluate(theta,target)
    prediction = nw.predict(theta,None, False,False)
    confusion[target][prediction] += 1
    if prediction == target: true +=1
  accuracy = true/n
  loss = error/n

  print 'Loss:', loss,'Accuracy:', accuracy, 'Confusion:'
  print confusionS(confusion, ttb.labels)


  return loss, accuracy, confusion


def main(args):
  hyperParams={k:args[k] for k in ['bSize','lambda','alpha','ada','nEpochs','fixEmb','fixW']}

  if args['kind'] == 'math': theta, ttb, dtb = math.install(args['pars'], kind='RNN',d=args['word'])
  elif args['kind'] == 'natlog': theta, ttb, dtb = natlog.install(args['src'], kind='RNN',d=args['word'])
  tr.plainTrain(ttb, dtb, hyperParams, theta, args['outDir'], args['cores'])
  print 'evaluation on held-out data:'
  loss, accuracy, confusion=evaluate(dtb, theta)
  print 'evaluation on train data:'
  loss, accuracy, confusion=evaluate(ttb, theta)




def mybool(string):
  if string in ['F', 'f', 'false', 'False']: return False
  if string in ['T', 't', 'true', 'True']: return True
  raise Exception('Not a valid choice for arg: '+string)
  hyperParams={k:args[k] for k in ['bSize','lambda','alpha','ada','nEpochs']}


if __name__ == "__main__":
  parser = argparse.ArgumentParser(description='Train classifier')
 # data:
  parser.add_argument('-exp','--experiment', type=str, help='Identifier of the experiment', required=True)
  parser.add_argument('-m','--model', choices=['RNN','IORNN','RAE'], default='RNN', required=False)
  parser.add_argument('-k','--kind', choices=['natlog','snli','math'], required=True)
  parser.add_argument('-s','--src', type=str, default='',help='Directory with training data', required=False)
  parser.add_argument('-o','--outDir', type=str, help='Output dir to store pickled theta', required=True)
  parser.add_argument('-p','--pars', type=str, default='', help='File with pickled theta', required=False)
  # network hyperparameters:
  parser.add_argument('-dwrd','--word', type=int, default = 0, help='Dimensionality of leaves (word nodes)', required=False)
  # training hyperparameters:
  parser.add_argument('-n','--nEpochs', type=int, help='Maximal number of epochs to train per phase', required=True)
  parser.add_argument('-b','--bSize', type=int, default = 50, help='Batch size for minibatch training', required=False)
  parser.add_argument('-l','--lambda', type=float, help='Regularization parameter lambdaL2', required=True)
  parser.add_argument('-a','--alpha', type=float, help='Learning rate parameter alpha', required=True)
  parser.add_argument('-ada','--ada', type=mybool, help='Whether adagrad is used', required=True)
  parser.add_argument('-c','--cores', type=int, default=1,help='The number of parallel processes', required=False)
  parser.add_argument('-fw','--fixEmb', type=mybool, default=False, help='Whether the word embeddings are fixed', required=False)
  parser.add_argument('-fc','--fixW', type=mybool, default=False, help='Whether the composition function is fixed', required=False)

  args = vars(parser.parse_args())

  main(args)

