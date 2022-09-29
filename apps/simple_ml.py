import struct
import gzip
import numpy as np
import idx2numpy


import sys
sys.path.append('python/')
import needle as ndl


def parse_mnist(image_filename, label_filename):
    """ Read an images and labels file in MNIST format.  See this page:
    http://yann.lecun.com/exdb/mnist/ for a description of the file format.

    Args:
        image_filename (str): name of gzipped images file in MNIST format
        label_filename (str): name of gzipped labels file in MNIST format

    Returns:
        Tuple (X,y):
            X (numpy.ndarray[np.float32]): 2D numpy array containing the loaded
                data.  The dimensionality of the data should be
                (num_examples x input_dim) where 'input_dim' is the full
                dimension of the data, e.g., since MNIST images are 28x28, it
                will be 784.  Values should be of type np.float32, and the data
                should be normalized to have a minimum value of 0.0 and a
                maximum value of 1.0.

            y (numpy.ndarray[dypte=np.int8]): 1D numpy array containing the
                labels of the examples.  Values should be of type np.int8 and
                for MNIST will contain the values 0-9.
    """
    ### BEGIN YOUR CODE
    with gzip.open(image_filename) as f:
        X = idx2numpy.convert_from_file(f)
        X = X.reshape(-1, 28 * 28)
        X = X / 255.0
    with gzip.open(label_filename) as f:
        y = idx2numpy.convert_from_file(f)
        y = y.reshape(-1)
    return X.astype('float32'), y.astype('uint8')


def softmax_loss(Z, y_one_hot):
    """ Return softmax loss.  Note that for the purposes of this assignment,
    you don't need to worry about "nicely" scaling the numerical properties
    of the log-sum-exp computation, but can just compute this directly.

    Args:
        Z (ndl.Tensor[np.float32]): 2D Tensor of shape
            (batch_size, num_classes), containing the logit predictions for
            each class.
        y (ndl.Tensor[np.int8]): 2D Tensor of shape (batch_size, num_classes)
            containing a 1 at the index of the true label of each example and
            zeros elsewhere.

    Returns:
        Average softmax loss over the sample. (ndl.Tensor[np.float32])
    """

    # prob = np.exp(Z) / np.sum(np.exp(Z), axis=1, keepdims=True)
    # return -np.log(prob[np.arange(y.shape[0]), y]).sum() / Z.shape[0]
    ### BEGIN YOUR SOLUTION
    rown = Z.shape[0]
    coln = Z.shape[1]
    # prob = ndl.exp(Z) / ndl.summation(ndl.exp(Z), axes=1, keepdims=True)  # B x c
    # return ndl.summation(ndl.exp(Z))
    prob = ndl.exp(Z) / ndl.broadcast_to(ndl.reshape(ndl.summation(ndl.exp(Z), axes=1), (rown, 1)), (rown, coln))
    # return (ndl.summation(prob) + 0 * ndl.summation(y_one_hot))
    res = -ndl.summation(ndl.log(prob) * y_one_hot) / rown
    # print("prob", prob.shape, prob)
    # print("y_one_hot", y_one_hot.shape, y_one_hot)
    # print("x1", prob * y_one_hot)
    # print("x2", ndl.log(prob * y_one_hot))
    # print("x3", ndl.summation(ndl.log(prob * y_one_hot)))

    # print("res", res.shape, res)
    return res



def nn_epoch(B_X, B_y, W1, W2, lr = 0.1, batch=100):
    """ Run a single epoch of SGD for a two-layer neural network defined by the
    weights W1 and W2 (with no bias terms):
        logits = ReLU(X * W1) * W1
    The function should use the step size lr, and the specified batch size (and
    again, without randomizing the order of X).

    Args:
        X (np.ndarray[np.float32]): 2D input array of size
            (num_examples x input_dim).
        y (np.ndarray[np.uint8]): 1D class label array of size (num_examples,)
        W1 (ndl.Tensor[np.float32]): 2D array of first layer weights, of shape
            (input_dim, hidden_dim)
        W2 (ndl.Tensor[np.float32]): 2D array of second layer weights, of shape
            (hidden_dim, num_classes)
        lr (float): step size (learning rate) for SGD
        batch (int): size of SGD mini-batch

    Returns:
        Tuple: (W1, W2)
            W1: ndl.Tensor[np.float32]
            W2: ndl.Tensor[np.float32]
    """

    for index in range(int(np.floor(B_X.shape[0]/batch))):
        X = B_X[index * batch: (index + 1) * batch, :]
        rown = X.shape[0]
        if rown == 0:
            break
        y = B_y[index * batch: (index + 1) * batch]
        # a1 = np.matmul(X, W1)
        # a1_p = np.where(a1 > 0, a1, 0)
        # logit = np.matmul(a1_p, W2)
        a1 = ndl.matmul(ndl.Tensor(X).data, W1)
        a1_p = ndl.relu(a1)
        logit = ndl.matmul(a1_p, W2)

        y_one_hot = np.zeros(logit.shape)
        y_one_hot[np.arange(rown), y] = 1
        y_one_hot = ndl.Tensor(y_one_hot).data
        loss = softmax_loss(logit, y_one_hot)
        loss.backward()
        W1 -= lr * W1.grad.data
        W2 -= lr * W2.grad.data
    return W1, W2


### CODE BELOW IS FOR ILLUSTRATION, YOU DO NOT NEED TO EDIT

def loss_err(h,y):
    """ Helper function to compute both loss and error"""
    y_one_hot = np.zeros((y.shape[0], h.shape[-1]))
    y_one_hot[np.arange(y.size), y] = 1
    y_ = ndl.Tensor(y_one_hot)
    return softmax_loss(h,y_).numpy(), np.mean(h.numpy().argmax(axis=1) != y)
