import os, os.path
import re
from nltk import tree
from Node import *
from getEmbeddings import *

def readC(vocabulary):
    print 'Reading corpus...'

#    words = ['dog', 'cat', 'chases', 'the','that', 'mouse']
    global voc
    voc = vocabulary
#    voc = {key: value for (key, value) in zip(words,range(len(words)))}
#    voc['UNK'] = len(words)
#    L,voc = getSennaEmbs()

    trees = set()

    corpusdir = 'C:/Users/Sara/AI/thesisData/vector-entailment-ICLR14-R1'
    for root, _, files in os.walk(corpusdir+'/data-4'):
        for f in files:
            with open(os.path.join(root,f),'r') as f:
                for line in f:
                    bits = line.split('\t')
                    if len(bits) == 3:
                     relation, s1, s2 = bits
                     t1 = stringToNetwork(s1)
                     t2 = stringToNetwork(s2)
                     trees.add(t1)
                     trees.add(t2)
    print 'Done.'
    return trees

def treeToNN(thistree, start, end):
#    print thistree, len(thistree)
    word = thistree.label().strip()

    if len(thistree) > 0:
        wordIndex = None
        children = [treeToNN(child, start, end) for child in thistree]
    else:

#       print word
       if word in voc: wordIndex = voc[word]
       else:           wordIndex = voc['UNK']
       children = []
    parent = Node(children, start, end, wordIndex)
    parent.setWord(word)
    return parent

def stringToNetwork(sent):
    sent = '('+re.sub(r"([^()\s]+)", r"(\1)", sent)+')'
#    sent = '('+re.sub(r"[()\s](\S+)[()\s]", r"(\1)", sent)+')'
#    sent = '('+re.sub(r"\(\s+(\S*)\s+(\S*)\s+\)", r"((\1)(\2))", sent)+')'
#    print sent
    thistree = tree.Tree.fromstring(sent)
    network = treeToNN(thistree, 0,len(thistree.leaves()))
#    print network
    return network




#readC()