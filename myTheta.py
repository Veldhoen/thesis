#from theano import tensor#, sparse
import numpy as np
import sys
from scipy import sparse

class Theta(dict):

  def __init__(self, style=None, dims=None, embeddings = None, grammar = None):
    if style is None: True
    else:
      if dims is None: print 'no dimensions for initialization of theta'
      elif style == 'IORNN': self.forIORNN(dims,embeddings,grammar)
      elif style == 'RNN': self.forRNN(dims,embeddings)
      else:
        print 'style not supported for theta initialization:', style
        sys.exit()

  def forIORNN(self, dims, embeddings = None, grammar = None):
    dwords = dims['word']
    din = dims['inside']
    dout = dims['outside']
    nwords = dims['nwords']
    self.newMatrix('compositionIM', None, (din,2*din))
  #  self.newMatrix('preterminalM',None,(din,dwords))
  #  self.newMatrix('preterminalB',None,(din))
    self.newMatrix('compositionIM',None,(din,2*din))
    self.newMatrix('compositionIB',None,(din))
    self.newMatrix('compositionLOM',None,(dout,din+dout))
    self.newMatrix('compositionLOB',None,(dout))
    self.newMatrix('compositionROM',None,(dout,din+dout))
    self.newMatrix('compositionROB',None,(dout))
    if grammar:
      rules = grammar[0]
      heads = grammar[1]
      for rule in rules+heads:
        self.newMatrix('composition-'+rule+'-IM',None,(din,2*din))
        self.newMatrix('composition-'+rule+'-IB',None,(din))
        self.newMatrix('composition-'+rule+'-LOM',None,(dout,din+dout))
        self.newMatrix('composition-'+rule+'-LOB',None,(dout))
        self.newMatrix('composition-'+rule+'-ROM',None,(dout,din+dout))
        self.newMatrix('composition-'+rule+'-ROB',None,(dout))
    self.newMatrix('wordIM',embeddings,(nwords,dwords))
    self.newMatrix('wordLOM', None,(dout,din+dout))
    self.newMatrix('wordLOB', None,(dout))
    self.newMatrix('wordROM', None,(dout,din+dout))
    self.newMatrix('wordROB', None,(dout))

    self.newMatrix('uOM', None,(1,dout))
    self.newMatrix('uOB',None,(1,1)) #matrix with one value, a 1-D array with only one value is a float and that's problematic with indexing


  def forRNN(dims):
    dwords = dims['word']
    din = dims['inside']
    dout = dims['outside']
    nwords = dims['nwords']
    self.newMatrix('compositionM', None, (din,2*din))
  #  self.newMatrix('preterminalM',None,(din,dwords))
    self.newMatrix('compositionM',None,(din,2*din))
    self.newMatrix('compositionB',None,(din))
    self.newMatrix('wordM',embeddings,(nwords,dwords))
    self.newMatrix('wordB', None,(dwords))

  def newMatrix(self, name,M= None, size = (0,0)):
    if M is not None: self[name] = M
    elif isinstance(size, (int,long)): self[name] = np.random.rand(size)*.2-.1
    elif len(size) == 2: self[name] = np.random.rand(size[0],size[1])*.2-.1
    else:
      print 'problem in newMatrix', name, M, size
      sys.exit()

  def regularize(self, alphaDsize, lambdaL2):
    for name in self.keys():
      if name[-1] == 'M': self[name] = (1- alphaDsize*lambdaL2)*self[name]
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

  def  norm(self):
    names = self.keys()
    return sum([np.linalg.norm(self[name]) for name in names])/len(names)
    #return 0

  def sparse(self):
    for name in self.keys():
      self[name] = sparse.csc_from_dense(self[name])

  def unSparse(self):
    for name in self.keys():
      self[name] = sparse.dense_from_sparse(self[name])

  def zeros_like(self, sparseWords = True):
    new = Theta()
    for name in self.keys():
      # for the word matrix, create a sparse matrix, as most values will not be updated
      if sparseWords and name == 'wordIM':
        shape = self[name].shape
#        print shape
        new.newMatrix(name,sparse.csc_matrix(shape))
      else:
        new.newMatrix(name, np.zeros_like(self[name]))
    return new

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
