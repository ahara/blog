import cPickle
import numpy as np
from elm import ELMClassifier
from sklearn import linear_model


def load_mnist(path='../Data/mnist.pkl'):
    with open(path, 'rb') as f:
        return cPickle.load(f)


def get_datasets(data):
    _train_x, _train_y = data[0][0], np.array(data[0][1]).reshape(len(data[0][1]), 1)
    _val_x, _val_y = data[1][0], np.array(data[1][1]).reshape(len(data[1][1]), 1)
    _test_x, _test_y = data[2][0], np.array(data[2][1]).reshape(len(data[2][1]), 1)

    return _train_x, _train_y, _val_x, _val_y, _test_x, _test_y


if __name__ == '__main__':
    # Load data sets
    train_x, train_y, val_x, val_y, test_x, test_y = get_datasets(load_mnist())
    # Build ELM
    cls = ELMClassifier(n_hidden=7000,
                        alpha=0.93,
                        activation_func='multiquadric',
                        regressor=linear_model.Ridge(),
                        random_state=21398023)
    cls.fit(train_x, train_y)
    # Evaluate model
    print 'Validation error:', cls.score(val_x, val_y)
    print 'Test error:', cls.score(test_x, test_y)
