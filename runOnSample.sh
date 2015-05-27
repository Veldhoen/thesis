experiment=$1
data='data/sickSample'
treePost='TREES.pik'
vocPost='VOC.pik'
out="models/sickSample$experiment.pik"
nEpochs=5
bSize=10
alpha='0.2'
lambda='0.0005'
cores=5

echo "training on $data for $nEpochs epochs"
echo "initializing word embeddings from $emb"
echo "parameters: batch size = $bSize, alpha = $alpha, lambda = $lambda."


python -u -Wi::DeprecationWarning \
  trainIOCopy.py \
  -t $data$treePost \
  -v $data$vocPost \
  -o $out \
  -dwrd 5 \
  -n $nEpochs \
  -b $bSize \
  -a $alpha \
  -l $lambda \
  -c $cores
echo "$experiment $!" >> psIds.txt




