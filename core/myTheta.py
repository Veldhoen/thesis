from __future__ import division
import numpy as np
import sys
from scipy import sparse
from collections import Counter

class Theta(dict):

  def __init__(self, style, dims, embeddings = None, grammar = None, vocabulary = ['UNKNOWN']):
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
        maxArity = max(maxArity,len(RHS.split(',')))
        rulesC[(LHS,RHS)]+=count
    self.maxArity = maxArity
    self.rules = rulesC.most_common()

  def __missing__(self, key):
    if key[0] == 'composition':
      lhs = key[1]
      rhs = key[2]
      if lhs == 'X':  #already looking for the most generic rule
        print 'No theta entry for',lhs,rhs
        sys.exit
      else:
        rhsBits = rhs[1:-1].split(',')
        if all(['X'in x for x in rhsBits]): #already looking for a generic rhs
          fakeKey=key[:1]+('X',)+key[2:]      #change lhs to generic
        else:                              #change rhs to generic, but keep lhs
          fakeKey=key[:2]+('('+','.join(['X']*len(rhsBits))+')',)+key[3:]
        return self[fakeKey]

  def forIORNN(self, embeddings, vocabulary ):
    print 'create composition matrices'
    for arity in xrange(1,self.maxArity+1):
      cat = 'composition'
      lhs = 'X'
      rhs = '('+','.join(['X']*arity)+')'
#      print lhs,rhs
      self.newMatrix((cat,lhs,rhs,'I','M'), None, (self.din,arity*self.din))
      self.newMatrix((cat,lhs,rhs,'I','B'),None,(self.din))
      for j in xrange(arity):
        self.newMatrix((cat,lhs,rhs,j,'O','M'),None,(self.dout,arity*self.din+self.dout))
        self.newMatrix((cat,lhs,rhs,j,'O','B'),None,(self.dout))
    print 'create lookup tables'
    self.lookup={('word',):vocabulary,('root',): ['']}
    self.newMatrix(('word',),embeddings,(self.nwords,self.dwords))
    self.newMatrix(('root',),None,(1,self.dout))


    print 'create score matrices'
    self.newMatrix(('u','M'), None,(self.dout,self.dout))
    self.newMatrix(('u','B'),None,(self.dout)) #matrix with one value, a 1-D array with only one value is a float and that's problematic with indexing

    self.newMatrix(('score','M'), None,(1,self.dout))
    self.newMatrix(('score','B'),None,(1,1)) #matrix with one value, a 1-D array with only one value is a float and that's problematic with indexing


  def specializeHeads(self):
    for lhs in self.heads:
      for arity in xrange(self.maxArity):
        rhs = ','.join([X]*arity)
        self.newMatrix((cat,lhs,rhs,'IM'), self[(cat,lhs,rhs,'IM')])
        self.newMatrix((cat,lhs,rhs,'IB'), self[(cat,lhs,rhs,'IB')])
        for j in xrange(arity):
          self.newMatrix((cat,lhs,rhs,j,'OM'),self[(cat,lhs,rhs,j,'OM')])
          self.newMatrix((cat,lhs,rhs,j,'OB'),self[(cat,lhs,rhs,j,'OB')])

  def specializeRules(self,n=200):
    cat = 'composition'
    for lhs,rhs in self.rules[:n]:
      self.newMatrix((cat,lhs,rhs,'IM'), self[(cat,lhs,rhs,'IM')])
      self.newMatrix((cat,lhs,rhs,'IB'), self[(cat,lhs,rhs,'IB')])
      arity = len(rhs.split(','))
      for j in xrange(arity):
        self.newMatrix((cat,lhs,rhs,j,'OM'),self[(cat,lhs,rhs,j,'OM')])
        self.newMatrix((cat,lhs,rhs,j,'OB'),self[(cat,lhs,rhs,j,'OB')])

  def newMatrix(self, name,M= None, size = (0,0)):
    if name in self: 
      print 'Adding a matrix to theta that is already there. Ignoring.'
      return

    if M is not None: self[name] = M
    elif isinstance(size, (int,long)): self[name] = np.random.rand(size)*.2-.1
    elif len(size) == 2: self[name] = np.random.rand(size[0],size[1])*.02-.01
    else:
      print 'problem in newMatrix', name, M, size
      sys.exit()

  def regularize(self, alphaDsize, lambdaL2):
    if lambdaL2==0: return
    for name in self.keys():
      if name[-1][-1] == 'M': self[name] = (1- alphaDsize*lambdaL2)*self[name]
      else: continue

  def update(self, gradient, alpha, historicalGradient = None):
#    print 'updating theta'
    for name in gradient.keys():
#      print name, 'before:',self[name].shape
      grad = gradient[name]
      if historicalGradient is not None:
        histgrad = historicalGradient[name]
        if sparse.issparse(grad): sq = grad.multiply(grad)
        else: sq = np.multiply(grad,grad)
        subtractFromDense(histgrad, -1*sq)      # add the square of the grad to histgrad
        subtractFromDense(self[name],grad, alpha/(np.sqrt(histgrad)+1e-6))#subtract gradient * alpha/root(histgrad)
      else:
        subtractFromDense(self[name],grad, alpha/np.ones_like(self[name]))

  def norm(self):
    names = self.keys()
    return sum([np.linalg.norm(self[name]) for name in names])/len(names)
    #return 0

  def sparse(self):
    for name in self.keys():
      self[name] = sparse.csc_from_dense(self[name])

  def unSparse(self):
    for name in self.keys():
      self[name] = sparse.dense_from_sparse(self[name])

  def gradient(self):
    return Gradient(self)

  def printDims(self):
    print 'Model dimensionality:'
    print '\tnwords -', self.nwords
    print '\td word -', self.dwords
    print '\td inside -', self.din
    print '\td outside -', self.dout


def subtractFromDense(denseM,decM, factor = None):
#  print 'adding sparse to dense. Sparse:', incM.shape,'Dense:', denseM.shape
  if not sparse.issparse(decM):
    if factor is None: denseM = denseM - decM
    else: denseM = denseM - np.multiply(factor,decM)

  elif sparse.isspmatrix_csc(decM):
    rows, columns = decM.nonzero()
    for i,j in zip(rows, columns):
      val = decM[i,j]
      if factor is not None: f = factor[i,j]
      else: f = 1
#      print i,j
      denseM[i,j] = denseM[i,j] - f*val
  else: print 'subtractFromDense not implemented for this format.'

class Gradient(Theta):
  def __init__(self,theta):
    self.theta = theta

  def __missing__(self, key):
    mold = self.theta[key]

    if key[0] == 'word': self.newMatrix(key,sparse.csc_matrix(mold.shape))
    else:   self.newMatrix(key, np.zeros_like(mold))
