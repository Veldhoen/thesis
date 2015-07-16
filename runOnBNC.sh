experiment=$1
data='data/BNC'
treePost='TREES.pik'
vocPost='VOC.pik'
gramPost='RULES.pik'
out=models/WSJ$experiment
nEpochs=5
bSize=10
alpha='0.05'
lambda='0.1'
cores=1

echo "$experiment $$" >> psIds.txt

echo "training on $data for $nEpochs epochs"
echo "initializing word embeddings from $emb"
echo "parameters: batch size = $bSize, alpha = $alpha, lambda = $lambda."
python -u -W once \
  trainIOCopy.py \
  -exp $experiment \
  -s $data \
  -e data/senna.pik \
  -g $data$gramPost \
  -o $out \
  -n $nEpochs \
  -b $bSize \
  -a $alpha \
  -l $lambda \
  -c $cores





