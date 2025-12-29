
#K-nearest neighbors (k-NN) classifier implementation and evaluation on CIFAR-10 dataset.
import numpy as np
import os
import matplotlib.pyplot as plt
from read_cifar import read_cifar, split_dataset

#1
#I applied the given hint (a-b)² = a² + b² - 2ab to compute distances 
# directly to matrices, which let me compute all pairwise distances in one vectorized step without loops.
#using the formula: ‖x−y‖² = ‖x‖² + ‖y‖² − 2x⋅y (@ for matrix multiplication, T for transpose)
def distance_matrix(X, Y):
    X_sq = np.sum(X ** 2, axis=1, keepdims=True)
    Y_sq = np.sum(Y ** 2, axis=1, keepdims=True).T
    XY = X @ Y.T
    dists = np.sqrt(X_sq + Y_sq - 2 * XY)
    return dists
#2
#I sorted the distances for each test sample to find the indices of the k nearest neighbors.    
def knn_predict(dists, labels_train, k):
    idx = np.argsort(dists, axis=0)[:k] 
    nearest_labels = labels_train[idx]
    preds = np.array([np.bincount(col).argmax() for col in nearest_labels.T])
    #bincount counts how many times each label appears in a list of labels.
    return preds
#3   
#Evaluating accuracy on the test set.
def evaluate_knn(data_train,labels_train,data_test,labels_test,k):
    dists = distance_matrix(data_train, data_test)
    preds = knn_predict(dists, labels_train, k)
    acc = np.mean(preds == labels_test)
    return acc   


data, labels = read_cifar("data//cifar-10-batches-py")
data_train, labels_train, data_test, labels_test = split_dataset(data, labels, split=0.9)
# I reduced the dataset size here because computing a full distance matrix on the entire CIFAR
# set is too slow, and subsampling still gives a representative accuracy curve.

data_train, labels_train = data_train[:3000], labels_train[:3000]
data_test, labels_test   = data_test[:800],  labels_test[:800]


accuracies = []
ks = range(1, 21)
for k in ks:
    acc = evaluate_knn(data_train, labels_train, data_test, labels_test, k)
    accuracies.append(acc)

os.makedirs("results", exist_ok=True)
plt.plot(ks, accuracies, marker='o')
plt.xlabel("k")
plt.ylabel("Accuracy")
plt.title("k-NN Accuracy vs k (split=0.9)")
plt.grid(True)
plt.savefig("results/knn.png")
plt.close()
