from __future__ import division
import numpy as np
import sys
import warnings
from collections import Counter,Iterable
warnings.filterwarnings('error')
class Theta(dict):

  def __init__(self, style, dims, grammar, embeddings = None,  vocabulary = ['UNKNOWN']):
    if dims is None:
      print 'No dimensions for initialization of theta'
      sys.exit()
    self.dims = dims
    self.heads,self.rules = self.grammar2rules(grammar)
    self.style = style
    self.installMatrices()

    if embeddings is None: default = ('UNKNOWN',np.random.random_sample(self.dims['word'])*.2-.1)
    else: default = ('UNKNOWN',embeddings[vocabulary.index('UNKNOWN')])
    self[('word',)] = WordMatrix(vocabulary, default, {})
    for i in range(len(vocabulary)):
      if embeddings is None:
        self[('word',)][vocabulary[i]]=np.random.random_sample(self.dims['word'])*.2-.1
      else: self[('word',)][vocabulary[i]]=embeddings[i]

  def grammar2rules(self, grammar):
    if grammar is None: grammar = Counter(Counter())
    heads = grammar.keys()
    rulesC = Counter()
    for LHS, RHSS in grammar.iteritems():
      for RHS, count in RHSS.iteritems():
        if len(RHS.split(', '))>self.dims['maxArity']: continue
        rulesC[(LHS,RHS)]+=count
    rules = [rule for rule, c in rulesC.most_common()]
    return heads, rules

  def additiveComposition(self):
    d= self.dims['inside']
    for key in self.keys():
      if key[0]=='composition':
        arity = len(key[2].split(', '))
        if key[-1]=='M': self[key]=np.concatenate([np.identity(d)]*arity,1)
        if key[-1]=='B': self[key]=np.zeros_like(self[key])

  def extend4Classify(self,nChildren, nClasses,dComparison = 0):
    if dComparison == 0: dComparison = (nChildren+1)*self.dims['inside']
    self.newMatrix(('comparison','M'),None,(dComparison,nChildren*self.dims['inside']))
    self.newMatrix(('classify','M'),None,(nClasses,dComparison))
    self.newMatrix(('comparison','B'),None,(dComparison,))
    self.newMatrix(('classify','B'),None,(nClasses,))


  def installMatrices(self):
    if self.style == 'classifier':
      self.newMatrix(('comparison','M'),None,(self.dims['comparison'],self.dims['arity']*self.dims['inside']))
      self.newMatrix(('classify','M'),None,(self.dims['nClasses'],self.dims['comparison']))
      self.newMatrix(('comparison','B'),None,(self.dims['comparison'],))
      self.newMatrix(('classify','B'),None,(self.dims['nClasses'],))
    else:
     # set local dimensionality variables
      din=self.dims['inside']
      if self.style == 'IORNN':
        dout=self.dims['outside']

      print '\tCreate composition matrices of all kinds'
      for arity in xrange(1,self.dims['maxArity']+1):
        lhs = '#X#'
        rhs = '('+', '.join(['#X#']*arity)+')'
        cat ='composition'
        self.newMatrix((cat,lhs,rhs,'I','M'),None,(din,arity*din))
        self.newMatrix((cat,lhs,rhs,'I','B'),None,(din,))
        if self.style == 'RAE':
          cat = 'reconstruction'
          self.newMatrix((cat,lhs,rhs,'M'),None,(arity*din,din))
          self.newMatrix((cat,lhs,rhs,'B'),None,(arity*din,))
        if self.style == 'IORNN':
          for j in xrange(arity):
            self.newMatrix((cat,lhs,rhs,j,'O','M'),None,(dout,(arity-1)*din+dout))
            self.newMatrix((cat,lhs,rhs,j,'O','B'),None,(dout,))

      if self.style == 'IORNN':
        print '\tCreate score matrices'
        self.newMatrix(('u','M'),None,(dout,din+dout))
        self.newMatrix(('u','B'),None,(dout,))
        self.newMatrix(('score','M'),None,(1,dout))
        self.newMatrix(('score','B'),None,(1,))
        self.newMatrix(('root',),None,(1,dout))


  def specializeHeads(self):
    print 'Theta, specializing composition parameters for heads'
    cat = 'composition'
    for lhs in self.heads:
      for arity in xrange(1,self.dims['maxArity']+1):
        rhs = '('+', '.join(['#X#']*arity)+')'
        self.newMatrix((cat,lhs,rhs,'I','M'), self[(cat,'#X#',rhs,'I','M')])
        self.newMatrix((cat,lhs,rhs,'I','B'), self[(cat,'#X#',rhs,'I','B')])
        if self.style == 'IORNN':
          for j in xrange(arity):
            self.newMatrix((cat,lhs,rhs,j,'O','M'),self[(cat,'#X#',rhs,j,'O','M')])
            self.newMatrix((cat,lhs,rhs,j,'O','B'),self[(cat,'#X#',rhs,j,'O','B')])
        if self.style == 'RAE':
          cat = 'reconstruction'
          self.newMatrix((cat,lhs,rhs,'M'),self[(cat,'#X#',rhs,'M')])
          self.newMatrix((cat,lhs,rhs,'B'),self[(cat,'#X#',rhs,'B')])
          cat = 'composition'

  def specializeRules(self,n=200):
    print 'Theta, specializing composition parameters for rules'
    cat = 'composition'

    for lhs,rhs in self.rules[:n]:
      self.newMatrix((cat,lhs,rhs,'I','M'), self[(cat,lhs,rhs,'I','M')])
      self.newMatrix((cat,lhs,rhs,'I','B'), self[(cat,lhs,rhs,'I','B')])
      arity = len(rhs.split(', '))
      if self.style == 'IORNN':
        for j in xrange(arity):
          self.newMatrix((cat,lhs,rhs,j,'O','M'),self[(cat,lhs,rhs,j,'O','M')])
          self.newMatrix((cat,lhs,rhs,j,'O','B'),self[(cat,lhs,rhs,j,'O','B')])
      if self.style == 'RAE':
        cat = 'reconstruction'
        self.newMatrix((cat,lhs,rhs,'M'),self[(cat,lhs,rhs,'M')])
        self.newMatrix((cat,lhs,rhs,'B'),self[(cat,lhs,rhs,'B')])
        cat = 'composition'

  def reset(self, cats):
    for cat in cats:
      if isinstance(self[cat],WordMatrix):
        for word in self[cat].keys():
          size = self[cat][word].shape
          self[cat][word] = np.random.random_sample(size)*.2-.1
      else:
        size = self[cat].shape
        self[cat] = np.random.random_sample(size)*.2-.1

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

  def add2Theta(self, gradient, alpha, historicalGradient = None):
    for key in gradient.keys():
      grad = gradient[key]
      if historicalGradient is not None:
        histgrad = historicalGradient[key]


#        historical_grad[name] += np.square(grad[name])
#       theta[name] = theta[name] - alpha*np.divide(grad[name],np.sqrt(historical_grad[name])+1e-6)

        if type(self[key]) == np.ndarray:
          histgrad+= np.square(grad)
          self[key] -= alpha*np.divide(grad,np.sqrt(histgrad)+1e-6)#
        elif type(self[key]) == WordMatrix:
          for word in grad:
            histgrad[word]+= np.square(grad[word])
            self[key][word] -= alpha*np.divide(grad[word],np.sqrt(histgrad[word])+1e-6)
        else: raise NameError("Cannot update theta")
      else:
        try: self[key] -=alpha*grad
        except:
          for word in grad: self[key][word] -=alpha*grad[word]
#      print self[('word',)]['UNKNOWN']

  def norm(self):
    names = [name for name in self.keys() if name[-1] == 'M']
    return sum([np.linalg.norm(self[name]) for name in names])/len(names)

  def gradient(self):
    molds = {}
    for key in self.keys():
      if isinstance(self[key], np.ndarray):
        molds[key]=np.shape(self[key])
    #initialize wordmatrix with zeroes default
    voc = self[('word',)].keys()
    defaultkey = 'UNKNOWN'
    defaultvalue = np.zeros_like(self[('word',)][defaultkey])
    wordM=WordMatrix(vocabulary=voc, default = (defaultkey,defaultvalue))
    return Gradient(molds,wordM)


  def __missing__(self, key):
    for fakeKey in generalizeKey(key):
      if fakeKey in self.keys():
        return self[fakeKey]
        break
    else:
      raise KeyError(str(key)+' not in theta (missing).')

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


  def printDims(self):
    print 'Model dimensionality:'
    for key, value in self.dims.iteritems():
      print '\t'+key+' - '+str(value)

  def __str__(self):
    txt = '<<THETA>>'
    txt+=' words: '+str(len(self[('word',)]))#+str(self[('word',)].keys()[:5])
    return txt

class Gradient(Theta):
  def __init__(self,molds,wordM,kvs={}):
    self.molds = molds
    dict.__setitem__(self,('word',),wordM)
    self.update(kvs)

  def __reduce__(self):
    return(self.__class__,(self.molds,self[('word',)],self.items()))

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
    if key in self.molds: dict.__setitem__(self, key, val)
    else:
      for fakeKey in generalizeKey(key):
        if fakeKey in self.molds: dict.__setitem__(self, fakeKey, val)
        break
      else:
        raise KeyError(str(key)+'not in gradient(setting), and not able to create it.')


class WordMatrix(dict):
  def __init__(self,vocabulary=None, default = ('UNKNOWN',0), dicItems={}):
    self.voc = vocabulary
    dkey,dval = default
    if dkey not in self.voc: raise AttributeError("'default' must be in the vocabulary")
    self.default = dkey
    dict.__setitem__(self, self.default, dval)
    [dicItems.remove((k,v)) for (k,v) in dicItems if k==dkey]
    self.update(dicItems)

  def __setitem__(self, key,val):
    if self.default not in self: raise KeyError("Default not yet in the vocabulary: "+self.default)#return None
    if key in self.voc:
#      if key=='UNKNOWN': print '\tkey is \"UNKNOWN\"!'
      dict.__setitem__(self,key, val)
      #except: sys.exit()
    else:
      dict.__setitem__(self,self.default, val)

  def erase(self):
    for key in self.keys():
      if key == self.default: continue
      else: del self[key]

  def __missing__(self, key):
#    print 'WM missing:', key
    if key == self.default: raise KeyError("Default not yet in the vocabulary: "+self.default)#return None
    if key in self.voc:
      self[key] = np.zeros_like(self[self.default])
      return self[key]
    else:
#      print 'WM missing:', key
      return self[self.default]

  def __reduce__(self):
    return(self.__class__,(self.voc,(self.default,self[self.default]),self.items()))

  def update(self,*args,**kwargs):
    if args:
      if len(args) > 1:
        raise TypeError("update expected at most 1 arguments, got %d" % len(args))
      other = dict(args[0])
      for key,val in other.items():
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
