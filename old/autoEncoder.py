import numpy as np
#import nltk
from nltk.tree import Tree

from Node import *



def computeRep(tree, vocabulary, M1, b1):
    if tree.height() < 3:
       return vocabulary[tree[0]]
    else:
       leftRep = computeRep(tree[0], vocabulary, M1, b1)
       rightRep = computeRep(tree[1], vocabulary, M1, b1)
       return M1.dot(np.concatenate([leftRep,rightRep]))+b1

def initializeM1b1(d):
    M = np.random.rand(d, 2*d)  #weights
    b = np.random.rand(d)       #bias
    return M,b

def initializeM2b2(d):
    M = np.random.rand(2*d,d)  #weights
    b = np.random.rand(2*d)       #bias
    return M,b



def initializeVocabulary(words, d):
    vocabulary = dict()
    for word in words:
        embedding = np.random.rand(d)
        vocabulary[word] = embedding
    return vocabulary

def parseSentence(sent, M1, b1, M2, b2, voc):
    representations =  dict()
    for index in range(len(sent)):
        representations[index] = [index,voc[sent[index]]]

    candidates = dict()

    while len(representations) > 1:
       print len(representations)
       for startL, rest in representations.iteritems():
           endL, repL = rest
           startR = endL+1
           if startR in representations.keys():
              endR, repR = representations[startR]
              if startL not in candidates.iterkeys():
                 candidates[startL] = dict()
              if endR in candidates[startL].iterkeys():
                 continue
              else:
                  children = np.concatenate([repL,repR])
                  parent = np.tanh(M1.dot(children)+b1)

                  childrenP = M2.dot(parent)+b2
                  error = .5*np.square(np.linalg.norm(children-childrenP))
                  candidates[startL][endR] = [parent,error]
       bestC = None

       for startC,rest in candidates.iteritems():
           for endC, repEr in rest.iteritems():
               if bestC is None:
                  bestC = [startC,endC,repEr]
               if repEr[1]<bestC[2][1]:
                  bestC = [startC,endC,repEr]
       start = bestC[0]
       end = bestC[1]
       rep = bestC[2][0]
       representations[start] = [end,rep]



    return ""



def main():
    d = 30
    words = ['dog', 'cat', 'chases', 'the']
    M1,b1 = initializeM1b1(d)
    M2,b2 = initializeM2b2(d)
    voc = initializeVocabulary(words,d)

    sentence = ['the','dog','chases', 'the','cat']
    parse = parseSentence(sentence, M1, b1, M2, b2, voc)
    #print M1, b1

if __name__ == "__main__":
   main()