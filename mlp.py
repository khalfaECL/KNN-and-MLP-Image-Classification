from read_cifar import read_cifar, split_dataset

#the suggessed code below by the repository:
#______________________________________________________________________________
import numpy as np

N = 30  # number of input data
d_in = 3  # input dimension
d_h = 3  # number of neurons in the hidden layer
d_out = 2  # output dimension (number of neurons of the output layer)

# Random initialization of the network weights and biaises
w1 = 2 * np.random.rand(d_in, d_h) - 1  # first layer weights
b1 = np.zeros((1, d_h))  # first layer biaises
w2 = 2 * np.random.rand(d_h, d_out) - 1  # second layer weights
b2 = np.zeros((1, d_out))  # second layer biaises

data = np.random.rand(N, d_in)  # create a random data
targets = np.random.rand(N, d_out)  # create a random targets

# Forward pass
a0 = data # the data are the input of the first layer
z1 = np.matmul(a0, w1) + b1  # input of the hidden layer
a1 = 1 / (1 + np.exp(-z1))  # output of the hidden layer (sigmoid activation function)
z2 = np.matmul(a1, w2) + b2  # input of the output layer
a2 = 1 / (1 + np.exp(-z2))  # output of the output layer (sigmoid activation function)
predictions = a2  # the predicted values are the outputs of the output layer

# Compute loss (MSE)
loss = np.mean(np.square(predictions - targets))
print(loss)
#__________________________________________________________________________________________
#here I defined the sigmoid function to be used later in the MLP training.
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_deriv(a):
    return a * (1 - a)


#10
#I implemented one full forward and backward pass so the network updates 
# its weights using the MSE gradient. I followed the derivative expressions 
# step by step to make sure each matrix multiplication matches the shapes from the forward pass.
def learn_once_mse(w1, b1, w2, b2, data, targets, learning_rate):
    z1 = data @ w1.T + b1
    a1 = sigmoid(z1)

    z2 = a1 @ w2.T + b2
    a2 = sigmoid(z2)

    loss = np.mean((a2 - targets) ** 2)

    dC_da2 = (2 / targets.shape[0]) * (a2 - targets)
    dC_dz2 = dC_da2 * sigmoid_deriv(a2)
    dC_dw2 = dC_dz2.T @ a1
    dC_db2 = np.sum(dC_dz2, axis=0, keepdims=True)

    dC_da1 = dC_dz2 @ w2
    dC_dz1 = dC_da1 * sigmoid_deriv(a1)
    dC_dw1 = dC_dz1.T @ data
    dC_db1 = np.sum(dC_dz1, axis=0, keepdims=True)

    w2 = w2 - learning_rate * dC_dw2
    b2 = b2 - learning_rate * dC_db2
    w1 = w1 - learning_rate * dC_dw1
    b1 = b1 - learning_rate * dC_db1

    return w1, b1, w2, b2, loss


#I converted the integer labels into one-hot vectors because 
# cross-entropy requires targets to be probability distributions. 
# This encoding lets the model compare its softmax output directly to the correct class.
def one_hot(labels):
    n_classes = np.max(labels) + 1
    one_hot_matrix = np.eye(n_classes)[labels]
    return one_hot_matrix

#I defined the softmax function as well to convert raw output scores into probabilities.
def softmax(x):
    e = np.exp(x - np.max(x, axis=1, keepdims=True))
    return e / np.sum(e, axis=1, keepdims=True)


def learn_once_cross_entropy(w1, b1, w2, b2, data, labels_train, learning_rate, sigmoid, sigmoid_deriv):
    a0 = data
    z1 = a0 @ w1 + b1
    a1 = sigmoid(z1)

    z2 = a1 @ w2 + b2
    a2 = softmax(z2)

    onehot = np.zeros((labels_train.shape[0], a2.shape[1]))
    onehot[np.arange(labels_train.shape[0]), labels_train] = 1

    loss = -np.mean(np.sum(onehot * np.log(a2 + 1e-12), axis=1))

    dC_dz2 = a2 - onehot
    dC_dw2 = a1.T @ dC_dz2
    dC_db2 = np.sum(dC_dz2, axis=0, keepdims=True)

    dC_da1 = dC_dz2 @ w2.T
    dC_dz1 = dC_da1 * sigmoid_deriv(a1)
    dC_dw1 = a0.T @ dC_dz1
    dC_db1 = np.sum(dC_dz1, axis=0, keepdims=True)

    w1 = w1 - learning_rate * dC_dw1
    b1 = b1 - learning_rate * dC_db1
    w2 = w2 - learning_rate * dC_dw2
    b2 = b2 - learning_rate * dC_db2

    return w1, b1, w2, b2, loss


def train_mlp(w1, b1, w2, b2, data_train, labels_train, learning_rate, num_epoch, learn_once_cross_entropy, sigmoid, sigmoid_deriv):
    train_accuracies = []
    for _ in range(num_epoch):
        w1, b1, w2, b2, _ = learn_once_cross_entropy(w1, b1, w2, b2, data_train, labels_train, learning_rate, sigmoid, sigmoid_deriv)

        z1 = data_train @ w1 + b1
        a1 = sigmoid(z1)
        z2 = a1 @ w2 + b2
        a2 = np.exp(z2 - np.max(z2, axis=1, keepdims=True))
        a2 = a2 / np.sum(a2, axis=1, keepdims=True)

        preds = np.argmax(a2, axis=1)
        acc = np.mean(preds == labels_train)
        train_accuracies.append(acc)

    return w1, b1, w2, b2, train_accuracies

#I ran a forward pass on the test set and computed the accuracy 
# by comparing predicted labels to ground truth. 
# This gives a clean measure of generalization after training.
def test_mlp(w1, b1, w2, b2, data_test, labels_test, sigmoid):
    z1 = data_test @ w1 + b1
    a1 = sigmoid(z1)
    z2 = a1 @ w2 + b2
    e = np.exp(z2 - np.max(z2, axis=1, keepdims=True))
    a2 = e / np.sum(e, axis=1, keepdims=True)
    preds = np.argmax(a2, axis=1)
    test_accuracy = np.mean(preds == labels_test)
    return test_accuracy



#I initialized the network, trained it for the required number of epochs, and evaluated it on 
# the test set so the function returns both the learning curve and the final performance. 
# This wraps the full workflow into a single call.
def run_mlp_training(data_train, labels_train, data_test, labels_test, d_h, learning_rate, num_epoch,
                     learn_once_cross_entropy, test_mlp, sigmoid, sigmoid_deriv):
    d_in = data_train.shape[1]
    d_out = np.max(labels_train) + 1

    w1 = 2 * np.random.rand(d_in, d_h) - 1
    b1 = np.zeros((1, d_h))
    w2 = 2 * np.random.rand(d_h, d_out) - 1
    b2 = np.zeros((1, d_out))

    train_accuracies = []

    for _ in range(num_epoch):
        w1, b1, w2, b2, _ = learn_once_cross_entropy(
            w1, b1, w2, b2, data_train, labels_train, learning_rate, sigmoid, sigmoid_deriv
        )

        z1 = data_train @ w1 + b1
        a1 = sigmoid(z1)
        z2 = a1 @ w2 + b2
        e = np.exp(z2 - np.max(z2, axis=1, keepdims=True))
        a2 = e / np.sum(e, axis=1, keepdims=True)
        preds = np.argmax(a2, axis=1)
        acc = np.mean(preds == labels_train)
        train_accuracies.append(acc)

    test_accuracy = test_mlp(w1, b1, w2, b2, data_test, labels_test, sigmoid)

    return w1, b1, w2, b2, train_accuracies, test_accuracy



import matplotlib.pyplot as plt
import os


data, labels = read_cifar("data//cifar-10-batches-py")
data_train, labels_train, data_test, labels_test = split_dataset(data, labels, split=0.9)
#I used the same trick as in k-NN to reduce the dataset size for faster training during development.
data_train, labels_train = data_train[:3000], labels_train[:3000]
data_test, labels_test   = data_test[:800],  labels_test[:800]

w1, b1, w2, b2, train_acc, test_acc = run_mlp_training(
    data_train, labels_train,
    data_test, labels_test,
    d_h=64,
    learning_rate=0.1,
    num_epoch=100,
    learn_once_cross_entropy=learn_once_cross_entropy,
    test_mlp=test_mlp,
    sigmoid=sigmoid,
    sigmoid_deriv=sigmoid_deriv
)

os.makedirs("results", exist_ok=True)
plt.plot(train_acc)
plt.xlabel("Epoch")
plt.ylabel("Training Accuracy")
plt.title("MLP Training Accuracy")
plt.grid(True)
plt.savefig("results/mlp.png")
plt.close()
