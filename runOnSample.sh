experiment=$1
data='data/sickSample'
treePost='TREES.pik'
vocPost='VOC.pik'
gramPost='RULES.pik'
out="models/sickSample$experiment"
nEpochs=5
bSize=10
alpha='0.05'
lambda='0.1'
cores=2

echo "$experiment $$" >> psIds.txt

echo "training on $data for $nEpochs epochs"
echo "initializing word embeddings from $emb"
echo "parameters: batch size = $bSize, alpha = $alpha, lambda = $lambda."

#nohup \
python -u -W once \
  trainIOCopy.py \
  -exp $experiment \
  -s $data \
  -g LHS 5 \
  -o $out \
  -dwrd 5 \
  -n $nEpochs \
  -b $bSize \
  -a $alpha \
  -l $lambda \
  -c $cores
# \
#   >> nohups/$experiment.nohup.out &
# echo "python main $!" >> psIds.txt





