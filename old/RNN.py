import numpy as np
#import nltk
from nltk.tree import Tree

def computeRep(tree, vocabulary, M1, b1):
    if tree.height() < 3:
       return vocabulary[tree[0]]
    else:
       leftRep = computeRep(tree[0], vocabulary, M1, b1)
       rightRep = computeRep(tree[1], vocabulary, M1, b1)
       return np.tanh(M1.dot(np.concatenate([leftRep,rightRep]))+b1)

def initializeM1b1(d):
    M1 = np.random.rand(d, 2*d)  #weights
    b1 = np.random.rand(d)       #bias
    return M1,b1

def initializeVocabulary(words, d):
    vocabulary = dict()
    for word in words:
        embedding = np.random.rand(d)
        vocabulary[word] = embedding
    return vocabulary

def parseSentence(sent, M1, b1, voc):
    return ""



def main():
    d = 30
    words = ['dog', 'cat', 'chases', 'the']
    M1,b1 = initializeM1b1(d)
    voc = initializeVocabulary(words,d)

    tree = Tree.fromstring("(S (NP (Det the) (N dog)) (VP (V chases) (NP (Det the) (N cat))))")
    print tree
    rep = computeRep(tree, voc, M1, b1)
    print rep

    #print M1, b1

if __name__ == "__main__":
   main()