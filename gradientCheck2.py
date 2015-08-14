import core.myTheta as myTheta
from nltk import Tree
import core.myIORNN as myIORNN
import core.myRAE as myRAE
import numpy as np
from scipy import sparse
import sys

def numericalGradient(nw, theta, target = None):
#    nw.activateNW(theta)
    print '\n\nComputing numerical gradient for target', target
    epsilon = 0.0001
    numgrad = theta.gradient()

    for name in theta.keys():
      if name == ('word',): #True
        for word in theta[name].keys():
          for i in range(len(theta[name][word])):
            old = theta[name][word][i]
            theta[name][word][i] = old + epsilon
            errorPlus=nw.error(theta,target,True)
            theta[name][word][i] = old - epsilon
            errorMin=nw.error(theta,target,True)
            d =(errorPlus-errorMin)/(2*epsilon)
            numgrad[name][word][i] = d
            theta[name][word][i] = old  # restore theta
      else:
    # create an iterator to iterate over the array, no matter its shape
        it = np.nditer(theta[name], flags=['multi_index'])

        while not it.finished:
          i = it.multi_index
#          print '\n\t',i
          old = theta[name][i]
          theta[name][i] = old + epsilon
          errorPlus=nw.error(theta,target,True)
          theta[name][i] = old - epsilon
          errorMin=nw.error(theta,target,True)
          d =(errorPlus-errorMin)/(2*epsilon)
          numgrad[name][i] = d
          theta[name][i] = old  # restore theta
          it.iternext()
    return numgrad



def gradientCheck(theta, network, target=None):
  #network.activateNW(theta)

  # compute analyticial and numerical gradient
  print 'computing analytical gradient'
  grad = theta.gradient()
  network.train(theta, grad, activate=True, target=target)
  numgrad = numericalGradient(network, theta, target)



  print 'Comparing analytical to numerical gradient'
  # flatten gradient objects and report difference
  gradflat = np.array([])
  numgradflat = np.array([])
  for name in theta.keys():
    if name == ('word',): #True
      ngr = np.concatenate([numgrad[name][word] for word in theta[name].keys()]) #reshape(numgrad[name],-1)
      gr = np.concatenate([grad[name][word] for word in theta[name].keys()]) #np.reshape(grad[name],-1)

    else:
      ngr = np.reshape(numgrad[name],-1)
      gr = np.reshape(grad[name],-1)
    if np.array_equal(gr,ngr): diff = 0
    else: diff = np.linalg.norm(ngr-gr)/(np.linalg.norm(ngr)+np.linalg.norm(gr))
    print 'Difference '+str(name)+' :', diff
    if diff>0.001:
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
  theta = myTheta.Theta('IORNN', dims,gram,None,voc)
  nw = myIORNN.IORNN(tree)

  gradientCheck(theta, nw, 'dogs')
  theta.specializeHeads()
  nw.activate(theta)
  gradientCheck(theta, nw, 'dogs')
  theta.specializeRules()
  gradientCheck(theta, nw, 'dogs')

def checkRAE():
  theta = myTheta.Theta('RAE', dims,gram,None,voc)
  nw = myRAE.RAE(tree)

#  gradientCheck(theta, nw)
#   theta.specializeHeads()
#  nw.activate(theta)
  gradientCheck(theta, nw)
#   theta.specializeRules()
#   gradientCheck(theta, nw)


voc = ['UNKNOWN','most','large', 'hippos','bark','chase','dogs']
#voc = ['UNKNOWN','most']
gram = {'S':{'(NP, VP)':2},'NP':{'(Q, N)':2,'(Q, A, N)':1}}
#  gram = ['S->(NP,VP)']
heads = ['NP']
#gram = {}
#  heads = []

#  voc = ['UNK','most','hippos']
d = 3
dims = {'inside':d,'outside':d,'word':d,'nwords':len(voc)}

#s = '(S (NP (Q most) (N hippos)) (VP (V chase) (NP (A big) (N dogs))))'
s = '(S (NP (Q most) (N (A big) (N hippos))) (VP (V chase) (NP (A big) (N dogs))))'
#s = '(Top (S (NP (Q most) (N hippos)) (VP (VP (V chase)) (N dogs))))'
#s = '(Top (S (NP (Q most) (N hippos)) (VP (V bark))))'
s = '(NP (Q most) (A big) (N hippos))'
#s = '(TOP (NP (Q most) (N hippos)))'
#s = '(NP (Q most) (N hippos))'
#s = '(Top (Q most))'
#s = '(Q most)'
tree = Tree.fromstring(s)
print tree
print tree.productions()

#checkIORNN()
checkRAE()
