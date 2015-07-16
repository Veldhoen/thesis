import core.myTheta as myTheta
from nltk import Tree
import core.myIORNN as myIORNN
import numpy as np
from scipy import sparse
import sys

def numericalGradient(nw, theta, target = None):
#    nw.activateNW(theta)
    print '\n\nComputing numerical gradient for target', target
    epsilon = 0.0001
    numgrad = theta.gradient()

    for name in theta.keys():
        print '\t',name
    # create an iterator to iterate over the array, no matter its shape
        it = np.nditer(theta[name], flags=['multi_index'])
        
        while not it.finished:
#          nw.recomputeNW(theta)
          i = it.multi_index
          old = theta[name][i]
          theta[name][i] = old + epsilon
 #         errorPlus = max(0,1-score0+nw.score(theta,target))
          errorPlus=nw.error(theta,target,True)
          theta[name][i] = old - epsilon
#          errorMin = max(0,1-score0+nw.score(theta,target))
          errorMin=nw.error(theta,target,True)
          d =(errorPlus-errorMin)/(2*epsilon)
#          if d!=0: print '\t\tchange gradient',i,', diff:', d
#          else: print '\t\tnot change gradient',i
          numgrad[name][i] = d
          theta[name][i] = old  # restore theta
          it.iternext()
    return numgrad



def gradientCheck(theta, network, target):
  #network.activateNW(theta)

  # compute analyticial and numerical gradient
  #grad = network.backprop(theta, target)
  print 'computing analytical gradient'
  grad = theta.gradient()
  network.trainWords(theta, grad, activate=True, target=target)
#  grad, err = network.train(theta, None, target)
#   if err<=1:
#     print 'no backpropagation.'
#     return
#  print 'computing numerical gradient'
#  numgrad = network.numericalGradient(theta,target)
  numgrad = numericalGradient(network, theta, target)

  print 'unsparse angrad:'
  grad.unSparse()
  print 'unsparse numgrad:'
  numgrad.unSparse()


  print 'Comparing analytical to numerical gradient'
  # flatten gradient objects and report difference
  gradflat = np.array([])
  numgradflat = np.array([])
  for name in theta.keys():
    ngr = np.reshape(numgrad[name],-1)
    gr = np.reshape(grad[name],-1)
    if np.array_equal(gr,ngr): diff = 0
    else: diff = np.linalg.norm(ngr-gr)/(np.linalg.norm(ngr)+np.linalg.norm(gr))
    print 'Difference '+str(name)+' :', diff
    if diff>0.01:
      print '    ','gr\t\tngr\t\td'
      for i in range(len(gr)):
        if gr[i]==0 and ngr[i]==0: v = str(0)
        else: v= str(abs((gr[i]-ngr[i])/(gr[i]+ngr[i])))
        try: print '    ',theta.lookup[name][i//len(grad[name][0])], str(gr[i])+'\t'+str(ngr[i])+'\t'+v
        except: print '    ',str(gr[i])+'\t'+str(ngr[i])+'\t'+v
    gradflat = np.append(gradflat,gr)
    numgradflat = np.append(numgradflat,ngr)
  print 'Difference overall:', np.linalg.norm(numgradflat-gradflat)/(np.linalg.norm(numgradflat)+np.linalg.norm(gradflat))

def checkIORNN():
  voc = ['UNKNOWN','most','large', 'hippos','bark','chase','dogs']
  gram = {'S':{'(NP,VP)':2},'NP':{'(Q,N)':2}}
#  gram = ['S->(NP,VP)']
  heads = ['NP']
#  gram = []
#  heads = []

#  voc = ['UNK','most','hippos']
  d = 3
  dims = {'inside':d,'outside':d,'word':d,'nwords':len(voc)}

  theta = myTheta.Theta('IORNN', dims,gram,None,voc)
#  s = '(S (NP (Q most) (N hippos)) (VP (V chase) (NP (A big) (N dogs))))'
#  s = '(S (NP (Q most) (N (A big) (N hippos))) (VP (V chase) (NP (A big) (N dogs))))'
  s = '(Top (S (VP (NP (Q most) (N hippos)) (V bark))))'
#  s = '(NP (Q most) (N hippos))'
#  s = '(Top (Q most))'
#  s = '(Q most)'
  tree = Tree.fromstring(s)
  print tree
  nw = myIORNN.IORNN(tree)
  nw.activate(theta)
#  gradientCheck(theta, nw, 'dogs')
#  print nw


  theta.specializeHeads()
  theta.specializeRules()
checkIORNN()
