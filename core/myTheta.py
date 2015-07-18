from __future__ import division
import numpy as np
import sys
from scipy import sparse
from collections import Counter

class Theta(dict):

  def __init__(self, style, dims, grammar, embeddings = None,  vocabulary = ['UNKNOWN']):
    if dims is None:
      print 'No dimensions for initialization of theta'
      sys.exit()
    self.nwords = dims['nwords']
    self.dwords = dims['word']
    self.din = dims['inside']
    self.dout = dims['outside']


    self.grammar2cats(grammar)
    if style == 'IORNN':
      self.forIORNN(embeddings, vocabulary)
    elif style == 'RNN': self.forRNN(embeddings)
    else:
      print 'Style not supported for theta initialization:', style
      sys.exit()

  def grammar2cats(self, grammar):
    if grammar is None: grammar = Counter(Counter())
    self.heads = grammar.keys()
    maxArity = 2
    rulesC = Counter()
    for LHS, RHSS in grammar.iteritems():
      for RHS, count in RHSS.iteritems():
        maxArity = max(maxArity,len(RHS.split(', ')))
        rulesC[(LHS,RHS)]+=count
    self.maxArity = maxArity
    self.rules = [rule for rule, c in rulesC.most_common()]

  def __missing__(self, key):
#    if key[1]=='S' : print 'Theta missing', key
    for fakeKey in generalizeKey(key):
      if fakeKey in self.keys():
#        print fakeKey
        return self[fakeKey]
        break
    else:
      print key, 'not in theta.'
      return None
      #sys.exit()

  def forIORNN(self, embeddings, vocabulary ):
    print 'create composition matrices'
    for arity in xrange(1,self.maxArity+1):
      cat = 'composition'
      lhs = '#X#'
      rhs = '('+', '.join(['#X#']*arity)+')'
#      print lhs,rhs
      self.newMatrix((cat,lhs,rhs,'I','M'), None, (self.din,arity*self.din))
      self.newMatrix((cat,lhs,rhs,'I','B'),None,(self.din))
      for j in xrange(arity):
        self.newMatrix((cat,lhs,rhs,j,'O','M'),None,(self.dout,(arity-1)*self.din+self.dout))
        self.newMatrix((cat,lhs,rhs,j,'O','B'),None,(self.dout))
    print 'create lookup tables'
    self.lookup={('word',):vocabulary,('root',): ['']}
    self.newMatrix(('word',),embeddings,(self.nwords,self.dwords))
    self.newMatrix(('root',),None,(1,self.dout))


    print 'create score matrices'
    self.newMatrix(('u','M'), None,(self.dout,self.din+self.dout))
    self.newMatrix(('u','B'),None,(self.dout)) #matrix with one value, a 1-D array with only one value is a float and that's problematic with indexing

    self.newMatrix(('score','M'), None,(1,self.dout))
    self.newMatrix(('score','B'),None,(1,1)) #matrix with one value, a 1-D array with only one value is a float and that's problematic with indexing
#    self.newMatrix(('score','B'),np.zeros((1,1))) #matrix with one value, a 1-D array with only one value is a float and that's problematic with indexing


  def specializeHeads(self):
    print 'Theta, specializing composition parameters for heads'
    cat = 'composition'
    for lhs in self.heads:
      for arity in xrange(1,self.maxArity+1):
        rhs = '('+', '.join(['#X#']*arity)+')'
        self.newMatrix((cat,lhs,rhs,'I','M'), self[(cat,'#X#',rhs,'I','M')])
        self.newMatrix((cat,lhs,rhs,'I','B'), self[(cat,'#X#',rhs,'I','B')])
        for j in xrange(arity):
          self.newMatrix((cat,lhs,rhs,j,'O','M'),self[(cat,'#X#',rhs,j,'O','M')])
          self.newMatrix((cat,lhs,rhs,j,'O','B'),self[(cat,'#X#',rhs,j,'O','B')])

  def specializeRules(self,n=200):
    print 'Theta, specializing composition parameters for rules'
    cat = 'composition'

    for lhs,rhs in self.rules[:n]:
      self.newMatrix((cat,lhs,rhs,'I','M'), self[(cat,lhs,rhs,'I','M')])
      self.newMatrix((cat,lhs,rhs,'I','B'), self[(cat,lhs,rhs,'I','B')])
      arity = len(rhs.split(', '))
      for j in xrange(arity):
        self.newMatrix((cat,lhs,rhs,j,'O','M'),self[(cat,lhs,rhs,j,'O','M')])
        self.newMatrix((cat,lhs,rhs,j,'O','B'),self[(cat,lhs,rhs,j,'O','B')])

  def newMatrix(self, name,M= None, size = (0,0)):
    if name in self:
      return

    if M is not None:
      if sparse.issparse(M): self[name] = M
      else: self[name] = np.copy(M)
    else: self[name] = np.random.random_sample(size)*.2-.1

  def regularize(self, alphaDsize, lambdaL2):
    if lambdaL2==0: return
    for name in self.keys():
      if name[-1] == 'M': self[name] = (1- alphaDsize*lambdaL2)*self[name]
      else: continue

  def update(self, gradient, alpha, historicalGradient = None):
    for name in gradient.keys():
      try:
        grad = gradient[name]
        oldtheta = np.copy(self[name])
        if historicalGradient is not None:
          histgrad = historicalGradient[name]
          histgrad+= np.multiply(grad,grad)
          self[name] -=(alpha/(np.sqrt(histgrad)+1e-6))*grad
        else:
          self[name] -=alpha*grad
      except: print 'updating theta unsuccesful'

  def norm(self):
    names = self.keys()
    return sum([np.linalg.norm(self[name]) for name in names])/len(names)
    #return 0

  def gradient(self):
    return Gradient(self)

  def printDims(self):
    print 'Model dimensionality:'
    print '\tnwords -', self.nwords
    print '\td word -', self.dwords
    print '\td inside -', self.din
    print '\td outside -', self.dout



class Gradient(Theta):
  def __init__(self,theta):
    self.theta = theta

  def __missing__(self, key):
    if key in self.theta.keys():
      mold = self.theta[key]
      self.newMatrix(key, np.zeros_like(mold))
      return self[key]
    else:
      for fakeKey in generalizeKey(key):
        if fakeKey in self.theta.keys():
          self.newMatrix(fakeKey, np.zeros_like(self.theta[fakeKey]))
          return self[fakeKey]
      else:
        print key,'not in gradient, and not able to create it.'
        return None



def generalizeKey(key):
  if key[0] == 'composition':
    lhs = key[1]
    rhs = key[2]
    generalizedHead = '#X#'
    generalizedTail = '('+', '.join(['#X#']*len(rhs[1:-1].split(', ')))+')'
    return[key[:2]+(generalizedTail,)+key[3:],key[:1]+(generalizedHead,generalizedTail,)+key[3:]]
  else: return []