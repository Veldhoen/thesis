experiment=$1
data='data/sickSample'
treePost='TREES.pik'
vocPost='VOC.pik'
out="models/sickSample$experiment.pik"
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
  -t $data$treePost \
  -v $data$vocPost \
  -o $out \
  -dwrd 5 \
  -n $nEpochs \
  -b $bSize \
  -a $alpha \
  -l $lambda \
  -c $cores





