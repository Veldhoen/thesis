from theano import tensor#, sparse
import numpy as np
import sys
from scipy import sparse

class Theta(dict):

  def __init__(self, style=None, dims=None, embeddings = None):
    if style is None: True
    else:
      if dims is None: print 'no dimensions for initialization of theta'
      elif style == 'IORNN': self.forIORNN(dims,embeddings)
      elif style == 'IORNN': self.forIORNN(dims,embeddings)
      else:
        print 'style not supported for theta initialization:', style
        sys.exit()

  def forIORNN(self, dims, embeddings = None):
    dwords = dims['word']
    din = dims['inside']
    dout = dims['outside']
    nwords = dims['nwords']
    self.newMatrix('compositionIM', None, (din,2*din))
  #  self.newMatrix('preterminalM',None,(din,dwords))
  #  self.newMatrix('preterminalB',None,(din))
    self.newMatrix('compositionIM',None,(din,2*din))
    self.newMatrix('compositionIB',None,(din))
    self.newMatrix('compositionOM',None,(din,2*din))
    self.newMatrix('compositionOB',None,(din))
    self.newMatrix('wordIM',embeddings,(nwords,dwords))
    self.newMatrix('wordIB', None,(dwords))
    self.newMatrix('wordOM', None,(2*din,2*din))
    self.newMatrix('wordOB', None,(2*din))
    self.newMatrix('uOM', None,(1,2*din))
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
    elif isinstance(size, (int,long)): self[name] = np.random.rand(size)*.02-.01
    elif len(size) == 2: self[name] = np.random.rand(size[0],size[1])*.02-.01
    else:
      print 'problem in newMatrix', name, M, size
      sys.exit()



  def update(self, gradient, historicalGradient, alpha):
#    print 'updating theta'
    for name in self.keys():
#      print name, 'before:',self[name].shape
      grad = gradient[name]
      histgrad = historicalGradient[name]
      if sparse.issparse(grad):
        addToDense(histgrad, grad.multiply(grad))
        addToDense(self[name],grad, alpha/(histgrad+1e-6))#/ ...
      else:
        histgrad = histgrad + np.square(grad)
        self[name] = self[name] + alpha * grad/(np.sqrt(histgrad)+1e-6)


#      print 'after:',self[name].shape





  def  norm(self):
    #names = self.keys()
    #return sum([np.linalg.norm(self[name]) for name in names])/len(names)
    return 0
    
  def sparse(self):
    for name in self.keys():
      self[name] = sparse.csc_from_dense(self[name])

  def unSparse(self):
    for name in self.keys():
      self[name] = sparse.dense_from_sparse(self[name])

  def zeros_like(self):
    new = Theta()
    for name in self.keys():
      # for the word matrix, create a sparse matrix, as most values will not be updated
      if name == 'wordIM':
        shape = self[name].shape
#        print shape
        new.newMatrix(name,sparse.csc_matrix(shape))
      else:
        new.newMatrix(name, np.zeros_like(self[name]))
    return new

def addToDense(denseM,incM, factor = None):
  if not sparse.issparse(incM):
    denseM = denseM + incM

  elif sparse.isspmatrix_csc(incM):
    for i,j in zip(inc.indices, incM.indptr):
      if factor: denseM[i,j] += incM[i,j] *factor[i,j]
      else: denseM[i,j] += incM[i,j]
  else: print 'addSparseToDense not implemented for this format.'
