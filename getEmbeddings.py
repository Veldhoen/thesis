import numpy as np


def getSennaEmbs(source, voc = ['UNK']):
    print '\tObtaining embeddings...'
    getVoc = (voc == ['UNK'])
    L = [0]*len(voc)
    with open(source+'/words.lst','r') as words:
        with open(source+'/embeddings.txt','r') as embs:
            for w,e in zip(words,embs):
                word = w.strip()
                emb = e.strip().split()
                if getVoc:
                  voc.append(word)
                  L.append(emb)
                else:
                  if w in voc: L[voc.index(word)] = emb
    d = len(emb)
    for i in range(len(L)):
      if L[i]==0: L[i] = np.random.rand(d)*0.02-0.01
    V = np.array(L)

    return V,voc
#getEmbs()