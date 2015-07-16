import pickle, sys, os


testData = sys.argv[1]
modelFile = sys.argv[2]

print testData, modelFile

files = [f for f in [os.path.join(testData,f) for f in os.listdir(testData)] if os.path.isfile(f)]
treebanks = [f for f in files if 'IORNNS' in f]



print 'load theta'
with open(modelFile,'rb') as f:
  theta = pickle.load(f)
voc = theta.lookup[('word',)]
#print [key[0] for key in theta.keys() if key[0]!= 'composition']





nars =[]

print 'evaluate networks'
for bank in treebanks:
  print 'reading from:', bank
  with open(bank,'rb') as f:
    nws = pickle.load(f)
    for nw in nws:
      nw.setScoreNodes()
      nar = nw.evaluateNAR(theta,voc)
      print nw, nar
      nars.append(nar)
    #print nars

with open('validationResults.tmp','wb') as f:
   pickle.dump(nars,f)
