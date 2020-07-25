import sys
import numpy as np
import KNN
import neural_net
import DecisionTree


def read_data(fname):
    exemplars = []
    image_ids = []
    file = open(fname, 'r')
    for line in file:
        data = tuple([w for w in line.split()])
        exemplars += [list(map(float, data[1:]))]
        image_ids += [data[0]]
    return image_ids, np.array(exemplars)


if __name__ == '__main__':
    if len(sys.argv) != 5:
        raise Exception("Error: expected 4 arguments")

    if sys.argv[1] == 'train':
        # Train your model
        # train train_file.txt model_file.txt [model]
        train_image_ids, train_data = read_data(sys.argv[2])
        if sys.argv[4] == 'nearest':
            # KNN
            KNN.train_data(sys.argv[2], sys.argv[3])
        elif sys.argv[4] == 'tree':
            # Decision tree
            DecisionTree.train(train_data, sys.argv[3])
        elif sys.argv[4] == 'nnet' or sys.argv[4] == 'best':
            # Neural nets is the best case
            neural_net.train(train_data, sys.argv[3])

    elif sys.argv[1] == 'test':
        # Test against your model_file.txt
        # test test_file.txt model_file.txt [model]
        test_image_ids, test_data = read_data(sys.argv[2])
        if sys.argv[4] == 'nearest':
            # KNN
            train_image_ids, train_data = read_data(sys.argv[3])
            KNN.start(train_data, test_data, test_image_ids)
        elif sys.argv[4] == 'tree':
            # Decision tree
            DecisionTree.test(test_data, sys.argv[3], test_image_ids)
        elif sys.argv[4] == 'nnet' or sys.argv[4] == 'best':
            # Neural Nets is the best case
            neural_net.test(sys.argv[3], test_data, test_image_ids)
