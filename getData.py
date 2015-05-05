import pickle, os
import nltk, re
import numpy as np
import random
import sys

def getSennaEmbs(source, voc = ['UNK']):
  print '\tObtaining embeddings...'
  source = os.path.join('../originalData',name)
  getVoc = (voc == ['UNK'])
  L = [0]*len(voc)
  with open(source+'/words.lst','r') as words:
    with open(source+'/embeddings.txt','r') as embs:
      for w,e in zip(words,embs):
        word = w.strip().strip('\'')
        emb = [float(e) for e in e.strip().split()]
#        print word, emb
        if getVoc:
          voc.append(word)
          L.append(emb)
        else:
          if w in voc: L[voc.index(word)] = emb
  d = len(emb)
  for i in range(len(L)):
    if L[i]==0: L[i] = np.random.rand(d)*0.02-0.01
  V = np.array(L)
  with open(os.path.join('data',name+'.pik'), 'wb') as f:
    pickle.dump([V,voc], f, -1)

def storeTrees(name):
  source = os.path.join('../originalData',name)
  # make a list of files to open
  if os.path.isdir(source):
    toOpen = [os.path.join(source,f) for f in os.listdir(source)]
    toOpen = [f for f in toOpen if os.path.isfile(f)]
  elif os.path.isfile(source): toOpen = [source]
  else: 
    print 'cannot open source', source
    sys.exit()
  examples = {'TRAIN':[],'TEST':[],'TRIAL':[]}
  vocabulary = set()
  for f in toOpen:
    print f
    with open(f,'r') as f:
      if 'sick' in name: f.next()
      for line in f:
#        print line
        if random.randint(0,4)<1: kind = 'TEST'
        else: kind = 'TRAIN'
        if 'sick' in name:
          bits = line.split('\t')
          ts = [nltk.tree.Tree.fromstring(s) for s in [bits[2],bits[4]]]
          relation = bits[5]
          kind = bits[-1].strip()
        elif 'bowman' in name:
          relation, s1, s2  = line.split('\t')
          ts = [nltk.tree.Tree.fromstring('('+re.sub(r"([^()\s]+)", r"(W \1)", s)+')') for s in [s1,s2]]
        else:
          ts = [nltk.tree.Tree.fromstring(line)]
          relation = None
        [nltk.treetransforms.chomsky_normal_form(t) for t in ts]
        [nltk.treetransforms.collapse_unary(t, collapsePOS = True,collapseRoot = True) for t in ts]
        if kind == 'TRAIN': vocabulary.update([w.lower() for w in t.leaves() for t in ts])
        if relation: examples[kind].append(tuple([ts,relation]))
        else: examples[kind].append(ts)
  vocabulary = list(vocabulary)
  vocabulary.insert(0, 'UNK')
  print len(vocabulary),'words,', len(examples['TRAIN']),'training examples,', len(examples['TEST']),'test examples.'

  with open(os.path.join('data',name+'.pik'), 'wb') as f:
    pickle.dump([examples,vocabulary], f, -1)


#name = 'senna'
#getSennaEmbs(name)
# with open('data/sennaV.pik', 'wb') as f:
#   V, voc =   pickle.load(f)


# names = ['bowman14','bowman15','sick','sickSample','flickr']
# for name in names:
#   print name
#   storeTrees(name)

storeTrees('WSJ')

# with open(os.path.join('data',name+'.pik'),'rb') as f:
#   data = pickle.load(f)

