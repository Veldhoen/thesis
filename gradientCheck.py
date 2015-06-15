import naturalLogicCopy as nl
from nltk import tree
import myTheta
import numpy as np
from scipy import sparse
import sys

def gradientCheck(theta, network, target):
  #network.activateNW(theta)

  # compute analyticial and numerical gradient
  #grad = network.backprop(theta, target)
  print 'computing analytical gradient'
  grad, err = network.train(theta, None, target)
#   if err<=1:
#     print 'no backpropagation.'
#     return
#  print 'computing numerical gradient'
  numgrad = network.numericalGradient(theta,target)




  print 'Comparing analytical numerical gradient'
  # flatten gradient objects and report difference
  gradflat = np.array([])

  numgradflat = np.array([])
  for name in theta.keys():
    ngr = np.reshape(numgrad[name],-1)
    if sparse.issparse(grad[name]): grad[name] = grad[name].todense()
    gr = np.reshape(grad[name],-1)
    if np.array_equal(gr,ngr): diff = 0
    else: diff = np.linalg.norm(ngr-gr)/(np.linalg.norm(ngr)+np.linalg.norm(gr))
    print 'Difference '+name+' :', diff
    if False:#'composition'in name: #diff>0:
      print 'gr\t\tngr\t\td'
      for i in range(len(gr)):
        print str(gr[i])+'\t'+str(ngr[i])+'\t'+str(abs((gr[i]-ngr[i])/(gr[i]+ngr[i])))
    gradflat = np.append(gradflat,gr)
    numgradflat = np.append(numgradflat,ngr)
  print 'Difference overall:', np.linalg.norm(numgradflat-gradflat)/(np.linalg.norm(numgradflat)+np.linalg.norm(gradflat))

def checkIORNN():
  voc = ['UNK','most','large', 'hippos','bark','chase','dogs']
#  voc = ['UNK','most','hippos']
  d = 15
  dims = {'inside':d,'outside':d,'word':d,'nwords':len(voc)}
  theta = myTheta.Theta('IORNN', dims)

#  s = '(S (NP (Q most) (N hippos)) (VP (V chase) (NP (A big) (N dogs))))'
  s = '(S (NP (Q most) (N (A big) (N hippos))) (VP (V chase) (NP (A big) (N dogs))))'
#  s = '(NP (Q most) (N hippos))'
#  s = '(Q most)'
  t = tree.Tree.fromstring(s)
  nw = nl.iornnFromTree(t, voc, grammarBased = False)
#  nw.activateNW(theta)
  gradientCheck(theta, nw, 0)

checkIORNN()

