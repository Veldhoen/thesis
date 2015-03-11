def setup(sentence1, sentence2, relation):
    tree1 = RNN(sentence1)
    tree2 = RNN(sentence2)
    network = Comparisonlayer(tree1,tree2)
    
