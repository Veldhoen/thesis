from __future__ import division
import numpy as np
import sys
import warnings
#from scipy import sparse
from collections import Counter,Iterable
warnings.filterwarnings('error')
class Theta(dict):

  def __init__(self, style, dims, grammar, embeddings = None,  vocabulary = ['UNKNOWN']):
    if dims is None:
      print 'No dimensions for initialization of theta'
      sys.exit()
    self.dims = dims
#    self.dims['nwords'] =len(vocabulary)
    if style not in ['RAE', 'IORNN','RNN']:
      print 'Style not supported for theta initialization:', style
      sys.exit()
    else: self.style = style
    self.grammar2cats(grammar)
    print '\tSet up matrix shapes'
    self.makeMolds(embeddings)
    print '\tCreate lookup tables'
    dict.__setitem__(self,('word',),WordMatrix(vocabulary, default = ('UNKNOWN',None)))
    for i in range(len(vocabulary)):
      if embeddings is None: self[('word',)][vocabulary[i]]=np.random.random_sample(self.dwords)*.2-.1
      else: self[('word',)][vocabulary[i]]=embeddings[i]

  def makeMolds(self,embeddings):
    # set local dimensionality variables
    din=self.dims['inside']
    if self.style == 'IORNN':
      dout=self.dims['outside']

    # create molds
    self.molds = {}
    for arity in xrange(1,self.maxArity+1):
      lhs = '#X#'
      rhs = '('+', '.join(['#X#']*arity)+')'
      cat ='composition'
      self.molds[(cat,lhs,rhs,'I','M')]= (din,arity*din)
      self.molds[(cat,lhs,rhs,'I','B')]= (din)
      if self.style == 'RAE':
        cat = 'reconstruction'
        self.molds[(cat,lhs,rhs,'I','M')]= (arity*din,din)
        self.molds[(cat,lhs,rhs,'I','B')]= (arity*din)
      if self.style == 'IORNN':
        for j in xrange(arity):
          self.molds[(cat,lhs,rhs,j,'O','M')]=(dout,(arity-1)*din+dout)
          self.molds[(cat,lhs,rhs,j,'O','B')]=(dout)

    if self.style == 'RAE':
      self.newMatrix(('reconstructionLeaf','M'),None,(din,din))
      self.newMatrix(('reconstructionLeaf','B'),None,(din))
    if self.style == 'IORNN':
      print '\tCreate score matrices'
      self.newMatrix(('u','M'), None,(dout,din+dout))
      self.newMatrix(('u','B'),None,(dout)) #matrix with one value, a 1-D array with only one value is a float and that's problematic with indexing
      self.newMatrix(('score','M'), None,(1,dout))
      self.newMatrix(('score','B'),None,(1,1)) #matrix with one value, a 1-D array with only one value is a float and that's problematic with indexing

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
    if key in self.molds:
      # if the key is supposed to be in theta, create a matrix for it and return it
      for fakeKey in generalizeKey(key):
        if fakeKey in self:
          self.newMatrix(key, self[fakeKey])
          break
      else:
        self.newMatrix(key, None,self.molds[key])
      return self[key]
    else:
      # else, return the generalized version of it
      for fakeKey in generalizeKey(key):
        if fakeKey in self.molds:
          return self[fakeKey]
      else:
        print key, 'not in theta (missing).'
        return None
        #sys.exit()

  def __setitem__(self, key,val):
    print 'theta.setitem',key
    if key in self.keys(): dict.__setitem__(self, key, val)
    elif key in self.molds: dict.__setitem__(self, key, val)
    else:
      for fakeKey in generalizeKey(key):
        if fakeKey in self.molds: dict.__setitem__(self, fakeKey, val)
        break
      else:
        raise KeyError(str(key)+' not in theta(setting), and not able to create it.')



  def __iadd__(self, other):
    for key in self:
      if isinstance(self[key],np.ndarray):
        if th: self[key] = self[key]+other[key]
        else: self[key] = self[key]+other
      elif isinstance(self[key],dict):
        for word in other[key]:
          if th: self[key][word] = self[key][word]+other[key][word]
          else: self[key][word] = self[key][word]+other
      else:
        print 'Inplace addition of theta failed:', key, 'of type',str(type(self[key]))
        sys.exit()
    return self

  def __add__(self, other):
    if isinstance(other,dict): th=True
    elif isinstance(other,int): th=False

    newT = self.gradient()
    for key in self:
      if isinstance(self[key],np.ndarray):
        if th: newT[key] = self[key]+other[key]
        else: newT[key] = self[key]+other
      elif isinstance(self[key],dict):
        for word in other[key]:
          if th: newT[key][word] = self[key][word]+other[key][word]
          else: newT[key][word] = self[key][word]+other
      else:
        print 'Inplace addition of theta failed:', key, 'of type',str(type(self[key]))
        sys.exit()
    return newT

  def __itruediv__(self,other):
    if isinstance(other,dict): th=True
    elif isinstance(other,int): th=False
    else: print 'unknown type of other in theta.itruediv'

    for key in self:
      if isinstance(self[key],np.ndarray):
        if th: self[key]/=other[key]
        else: self[key]/=other
      elif isinstance(self[key],dict):
        if th:
          for word in other[key]:
            self[key][word]/=other[key][word]
        else:
          for word in self[key]:
            self[key][word]/=other
      else:
        print 'Inplace division of theta failed:', key, 'of type',str(type(self[key]))
        sys.exit()
    return self

  def __idiv__(self,other):
    return self.__itruediv__(other)



  def specializeHeads(self):
    print 'Theta, specializing composition parameters for heads'

    for key,value in self.molds.iteritems():
      if key[0]=='composition' or key[0]=='reconstruction':
        for head in self.heads:
          newKey=key[:1]+(head,)+key[2:]
          self.molds[newKey] = self.molds[key]

  def specializeRules(self,n=200):
    print 'Theta, specializing parameters for rules'

    din = self.dims['inside']
    if 'dout' in self.dims: dout = self.dims['outside']
    for lhs,rhs in self.rules[:n]:
      arity = len(rhs.split(', '))

      cat ='composition'
      self.molds[(cat,lhs,rhs,'I','M')]= (din,arity*din)
      self.molds[(cat,lhs,rhs,'I','B')]= (din)
      if style == 'RAE':
        cat = 'reconstruction'
        self.molds[(cat,lhs,rhs,'I','M')]= (arity*din,din)
        self.molds[(cat,lhs,rhs,'I','B')]= (arity*din)
      if style == 'IORNN':
        for j in xrange(arity):
          self.molds[(cat,lhs,rhs,j,'O','M')]=(dout,(arity-1)*din+dout)
          self.molds[(cat,lhs,rhs,j,'O','B')]=(dout)

  def newMatrix(self, name,M= None, size = (0,0)):
    if name in self:
      return

    if M is not None: dict.__setitem__(self, name, np.copy(M))
#    self[name] = np.copy(M)
    else: dict.__setitem__(self, name, np.random.random_sample(size)*.2-.1)

  def regularize(self, alphaDsize, lambdaL2):
    if lambdaL2==0: return
    for name in self.keys():
      if name[-1] == 'M': self[name] = (1- alphaDsize*lambdaL2)*self[name]
      else: continue

  def update(self, gradient, alpha, historicalGradient = None):
    for key in gradient.keys():
        grad = gradient[key]
        #oldtheta = np.copy(self[name])
        if historicalGradient is not None:
          histgrad = historicalGradient[key]
          if type(self[key]) == np.ndarray:
            histgrad+= np.multiply(grad,grad)
            self[key] -=(alpha/(np.sqrt(histgrad)+1e-6))*grad
          elif type(self[key]) == WordMatrix:
            for word in grad:
              histgrad[word]+= np.multiply(grad[word],grad[word])
              self[key][word] -=(alpha/(np.sqrt(histgrad[word])+1e-6))*grad[word]
        else:
          try: self[key] -=alpha*grad
          except:
            for word in grad: self[key][word] -=alpha*grad[word]

  def norm(self):
    names = [name for name in self.keys() if name[-1] == 'M']
    return sum([np.linalg.norm(self[name]) for name in names])/len(names)
    #return 0

  def gradient(self):
    return Gradient(self)

  def printDims(self):
    print 'Model dimensionality:'
    for key, value in self.dims.iteritems():
      print '\t'+key+' - '+str(value)
#     print '\tnwords -', self.nwords
#     print '\td word -', self.dwords
#     print '\td inside -', self.din
#     print '\td outside -', self.dout

  def __str__(self):
    txt = '<<THETA>>'
    txt+=' words: '+str(len(self[('word',)]))#+str(self[('word',)].keys()[:5])
    return txt

class Gradient(Theta):
#  molds = dict()
  def __init__(self,theta,wordM=None):
    self.molds = dict()
    for key in theta.keys():
      if isinstance(theta[key], np.ndarray):
        self.molds[key] = np.shape(theta[key])
      elif isinstance(theta[key],WordMatrix):
        self.molds[key] = 'wordM'
        voc = theta[key].keys()
#        print len(voc)
        if 'UNKNOWN' in voc: defaultkey = 'UNKNOWN'
        else:
          print 'failed to initialize wordmatrix with a value'
          sys.exit()
        defaultvalue = np.zeros_like(theta[key][defaultkey])
        self[key] = WordMatrix(vocabulary=voc, default = (defaultkey,defaultvalue))
      elif isinstance(theta[key],tuple):
        self.molds[key]=theta[key]
      elif theta[key] == 'wordM':
        self.molds[key]='wordM'
        self[key] = wordM
      else:
        print 'Creating gradient. Cannot instantiate', key, 'of type',str(type(theta[key]))
        sys.exit()


  def __reduce__(self):
    return(self.__class__,(self.molds,self[('word',)]))

  def __missing__(self, key):
    if key in self.molds:
      self.newMatrix(key, np.zeros(self.molds[key]))
      return self[key]
    else:
      for fakeKey in generalizeKey(key):
        if fakeKey in self.molds:
          self.newMatrix(fakeKey, np.zeros(self.molds[fakeKey]))
          return self[fakeKey]
      else:
        print key,'not in gradient(missing), and not able to create it.'
        return None
  def __setitem__(self, key,val):
#    print 'gradient set',key, val
#    if self.default not in self.keys(): True #raise KeyError("Default must be in the vocabulary")
#    dict.__setitem__(self, key, val)
    if key in self.molds: dict.__setitem__(self, key, val)
    else:
      for fakeKey in generalizeKey(key):
        if fakeKey in self.molds: dict.__setitem__(self, fakeKey, val)
        break
      else:
        print key,'not in gradient(setting), and not able to create it.'
#        print 'molds:', [ mold for mold in self.molds.keys() if mold[0]!='composition']
        sys.exit()



class WordMatrix(dict):
#  voc = set()
#  default = ''
  def __init__(self,vocabulary=None, default = ('UNKNOWN',0), dicItems={}):
#    print 'Wordmatrix initialization'
    self.voc = vocabulary
    dkey,dval = default
    if dkey not in self.voc: raise AttributeError("'default' must be in the vocabulary")
    self.default = dkey
    self[self.default] = dval
    self.update(dicItems)
#     for key,value in kwargs.items():
#       if key not in self.voc: self[self.default] += value
#       else: self[key] += value
#    print 'initialized wordmatrix. default:',self.default,self[self.default]

  def __setitem__(self, key,val):
#    if self.default not in self.keys(): True #raise KeyError("Default must be in the vocabulary")
#    dict.__setitem__(self, key, val)
    if key in self.voc: dict.__setitem__(self, key, val)
    else: dict.__setitem__(self, self.default, val)
  def __missing__(self, key):
    if key == self.default: raise KeyError("Default not yet in the vocabulary")#return None
#    raise KeyError("Key not in the vocabulary")
#    return None
#    if self.default not in self.keys(): raise KeyError("Default '%s' must be in the vocabulary. "% str(self.default)+str(self.keys())+str(len(self.voc)))
    if key in self.voc:
      self[key] = np.zeros_like(self[self.default])
      return self[key]
    else:
      return self[self.default]
  def __reduce__(self):
    return(self.__class__,(self.voc,(self.default,self[self.default]),self.items()))
  def update(self,*args,**kwargs):
    if args:
      if len(args) > 1:
        raise TypeError("update expected at most 1 arguments, got %d" % len(args))
      other = dict(args[0])
      for key,val in other.items():
#        print key,val
        if key not in self.voc: self[self.default] += val
        else: self[key] += val
    for key,value in kwargs.items():
      if key not in self.voc: self[self.default] += val
      else: self[key] += val

def generalizeKey(key):
  if key[0] == 'composition' or key[0] == 'reconstruction':
    lhs = key[1]
    rhs = key[2]
    generalizedHead = '#X#'
    generalizedTail = '('+', '.join(['#X#']*len(rhs[1:-1].split(', ')))+')'
    return[key[:2]+(generalizedTail,)+key[3:],key[:1]+(generalizedHead,generalizedTail,)+key[3:]]
  else: return []