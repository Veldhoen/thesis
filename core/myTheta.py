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

  def __iadd__(self, other):
    for key in self:
      if isinstance(self[key],np.ndarray):
        self[key]+=other[key]
      elif isinstance(self[key],dict):
        for word in other[key]:
          self[key][word]+= other[key][word]
      else:
        print 'Inplace addition of theta failed:', key, 'of type',str(type(self[key]))
        sys.exit()
    return self

  def __add__(self, other):
    newT = self.gradient()
    for key in self:
      if isinstance(self[key],np.ndarray):
        newT[key] = self[key]+other[key]
      elif isinstance(self[key],dict):
        for word in other[key]:
          newT[key][word] = self[key][word]+ other[key][word]
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
        self[key]/=other[key]
      elif isinstance(self[key],dict):
        for word in other[key]:
          self[key][word]/= other[key][word]
      else:
        print 'Inplace division of theta failed:', key, 'of type',str(type(self[key]))
        sys.exit()
    return self

  def __idiv__(self,other):
    return self.__itruediv__(other)



  def __iadd__(self, other):
    for key in self:
      if isinstance(self[key],np.ndarray):
        self[key]+=other[key]
      elif isinstance(self[key],dict):
        for word in other[key]:
          self[key][word]+= other[key][word]
      else:
        print 'Inplace addition of theta failed:', key, 'of type',str(type(self[key]))
        sys.exit()
    return self


  def forIORNN(self, embeddings, vocabulary ):
    print '\tCreate composition matrices'
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
    print '\tCreate lookup tables'
    self[('word',)] = WordMatrix(vocabulary, default = ('UNKNOWN',None))
    for i in range(len(vocabulary)):
      if embeddings is None: self[('word',)][vocabulary[i]]=np.random.random_sample(self.dwords)*.2-.1
      else: self[('word',)][vocabulary[i]]=embeddings[i]

#    self.newMatrix(('word',),embeddings,(self.nwords,self.dwords))
    self.newMatrix(('root',),None,(1,self.dout))


    print '\tCreate score matrices'
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

    if M is not None: self[name] = np.copy(M)
    else: self[name] = np.random.random_sample(size)*.2-.1

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
            oldh = np.copy(histgrad)
            histgrad+= np.multiply(grad,grad)
            try: self[key] -=(alpha/(np.sqrt(histgrad)+1e-6))*grad
            except RuntimeWarning: raise NameError("invalid sqrt of histgrad? "+str(histgrad))
          elif type(self[key]) == WordMatrix:
            for word in grad:
              histgrad[word]+= np.multiply(grad[word],grad[word])
              self[key][word] -=(alpha/(np.sqrt(histgrad[word])+1e-6))*grad[word]
          else: raise NameError("Cannot update theta")
        else:
          try: self[key] -=alpha*grad
          except:
            for word in grad: self[key][word] -=alpha*grad[word]
      #except: print 'updating theta unsuccesful'

  def norm(self):
    names = [name for name in self.keys() if name[-1] == 'M']
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

  def __str__(self):
    txt = '<<THETA>>'
    txt+=' words: '+str(len(self[('word',)]))#+str(self[('word',)].keys()[:5])
    return txt

class Gradient(Theta):
  def __init__(self,theta):
    self.molds = dict()
    for key in theta.keys():
      if type(theta[key]) == np.ndarray:
        self.molds[key] = np.shape(theta[key])
      elif type(theta[key]) == WordMatrix:
        voc = theta[key].keys()
#        print len(voc)
        if 'UNKNOWN' in voc: defaultkey = 'UNKNOWN'
        else:
          print 'failed to initialize wordmatrix with a value'
          sys.exit()
        defaultvalue = np.zeros_like(theta[key][defaultkey])
        self[key] = WordMatrix(vocabulary=voc, default = (defaultkey,defaultvalue))
      else:
        print 'Creating gradient. Cannot instantiate', key, 'of type',str(type(theta[key]))
        sys.exit()
#  def __add__(self,other):


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
        print key,'not in gradient, and not able to create it.'
        return None

class WordMatrix(dict):
  voc = set()
  default = ''
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
    else: return self[self.default]
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
  if key[0] == 'composition':
    lhs = key[1]
    rhs = key[2]
    generalizedHead = '#X#'
    generalizedTail = '('+', '.join(['#X#']*len(rhs[1:-1].split(', ')))+')'
    return[key[:2]+(generalizedTail,)+key[3:],key[:1]+(generalizedHead,generalizedTail,)+key[3:]]
  else: return []