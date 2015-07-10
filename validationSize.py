import pickle, sys, os


testData = sys.argv[1]
modelFile = sys.argv[2]

print testData, modelFile

files = [f for f in [os.path.join(testData,f) for f in os.listdir(testData)] if os.path.isfile(f)]
treebanks = [f for f in files if 'IORNNS' in f]

nws = []
print 'obtain networks'
for bank in treebanks:
  with open(bank,'rb') as f:
    nws.extend(pickle.load(f))

print 'load theta'
with open(modelFile,'rb') as f:
  theta = pickle.load(f)
voc = theta.lookup[('word',)]
#print [key[0] for key in theta.keys() if key[0]!= 'composition']


nars =[]
for nw in nws:
  nw.setScoreNodes()
  nar = nw.evaluateNAR(theta,voc)
  print nw, nar
  nars.append(nar)
print nars

with open('validationResults.tmp','wb') as f:
   pickle.dump nars
