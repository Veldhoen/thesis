import pickle, os
import nltk, re
import numpy as np
import random
import sys
from collections import defaultdict, Counter

import core.myIORNN as myIORNN


def sennaLeaves(tree,sennaVoc):
  if isinstance(tree, nltk.Tree):
    for i in range(len(tree)):
      tree[i] = sennaLeaves(tree[i],voc)
    return tree
  else:
    return sennaproof(tree,sennaVoc)



def getSennaEmbs(destination):
  print '\tObtaining embeddings...'
  source = os.path.join('../originalData','senna')
  L = []
  voc=[]
  with open(source+'/words.lst','r') as words:
    with open(source+'/embeddings.txt','r') as embs:
      for w,e in zip(words,embs):
        word = w.strip().strip('\'')
        emb = [float(e) for e in e.strip().split()]
        voc.append(word)
        L.append(emb)
  V = np.array(L)
  with open(destination, 'wb') as f:
    pickle.dump([V,voc], f, -1)
  print 'Done.'

def getIORNNs(source,outDir,sennaVoc):
  rules = defaultdict(Counter)
  voc = set()
  if os.path.isdir(source):
    toOpen = [os.path.join(source,f) for f in os.listdir(source)]
    toOpen = [f for f in toOpen if os.path.isfile(f)]
  elif os.path.isfile(source): toOpen = [source]
  else:
    print 'cannot open source', source
    sys.exit()


  for filename in toOpen:
    name = os.path.splitext(os.path.split(filename)[1])[0]
    print 'converting trees from', filename
    nws = []
    with open(filename,'r') as f:
      counter = 0
      for line in f:
        counter+=1
        try:
          tree = nltk.tree.Tree.fromstring(line)
          sennaLeaves(tree,sennaVoc)
#          print tree.leaves()
          nws.append(myIORNN.IORNN(tree,sennaVoc))
          for prod in tree.productions():
            if prod.is_nonlexical():
              rules[str(prod.lhs())][str(prod.rhs())]+=1
          for word, pos in tree.pos():
            voc.add(word)

        except:
          print 'transformation to IORNN failed.'
          print line
        if counter % 50 == 0: print counter
        if counter % 1000 == 0:
          out =os.path.join(outDir,name+'_IORNNS_'+str(counter//1000)+'.pik')
          print 'writing to', out
          with open(out,'wb') as f:
            pickle.dump(nws,f)
            nws = []
  print 'writing rules and vocabulary to file'
  with open(os.path.join(outDir,name+'_RULES.pik'),'wb') as f:
    pickle.dump(rules,f)
  with open(os.path.join(outDir,name+'_VOC.pik'),'wb') as f:
    pickle.dump(voc,f)





def sennaproof(word,sennaVoc):
  word = word.strip('\"').lower()
  if word[0]=='-' and word[-1]=='-':
    word='trace-UNK'
  if word in sennaVoc: return word
  else:
    digit = True
    bits = re.split(',|\.',word)
    for b in bits:
      if not b.isdigit(): digit is False
    if digit: return '0'
    else: return word+'-UNK'


senna = os.path.join('data','sennaEMB'+'.pik')

#getSennaEmbs(senna)

with open(senna, 'rb') as f:
  V, voc =   pickle.load(f)

getIORNNs('../originalData/WSJ','data/newWSJ',voc)
#getIORNNs('../originalData/BNC','data/newBNC',voc)
