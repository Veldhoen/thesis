import numpy as np


def getSennaEmbs():
    print '\tObtaining embeddings...'
    senna = 'C:/Users/Sara/AI/thesisData/senna'
    d = 50
    v = 130000
    voc = dict()
    L = np.zeros([v+1,d])

    i = 0

    with open(senna+'/hash/words.lst','r') as words:
        with open(senna+'/embeddings/embeddings.txt','r') as embs:

            for w,e in zip(words,embs):
                voc[w.strip()] = i
                L[i] = np.array(e.strip().split())
                i+=1
    L[i] = np.random.rand(len(L[i-1]))
    voc['UNK'] = i
    return L,voc
#getEmbs()