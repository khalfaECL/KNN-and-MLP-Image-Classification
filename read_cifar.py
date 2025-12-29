import pickle
import numpy as np
import os

#I loaded a single CIFAR batch and extracted the raw arrays because each batch 
# already contains 10 000 samples flattened into 3072 values. I converted everything 
# to the right NumPy types so the data is usable later without extra fixes.
def read_cifar_batch(batch_path: str):
    with open(batch_path, 'rb') as fo:
        batch = pickle.load(fo, encoding='bytes')
    data = batch[b'data']  
    labels = np.array(batch[b'labels'])
    data = data.astype(np.float32)
    labels = labels.astype(np.int64)
    return data, labels

#I combined all six CIFAR batches because the dataset is originally 
# split across multiple files, and I wanted one unified matrix of 60 000 samples.
# I simply loaded each batch and stacked them to rebuild the full dataset cleanly.
def read_cifar(dir_path: str):
    all_data = []
    all_labels = []
    for i in range(1, 6):
        batch_path = os.path.join(dir_path, f"data_batch_{i}")
        data, labels = read_cifar_batch(batch_path)
        all_data.append(data)
        all_labels.append(labels)
    test_path = os.path.join(dir_path, "test_batch")
    test_data, test_labels = read_cifar_batch(test_path)
    all_data.append(test_data)
    all_labels.append(test_labels)
    data = np.concatenate(all_data, axis=0).astype(np.float32)
    labels = np.concatenate(all_labels, axis=0).astype(np.int64)

    return data, labels

import numpy as np
#I shuffled all indices and then split them because I needed a random,
# unbiased separation between training and test sets. 
# This way, each call gives a different partition while keeping data 
# and labels perfectly aligned.
def split_dataset(data, labels, split):
    assert data.shape[0] == labels.shape[0]
    assert 0.0 < split < 1.0
    N = data.shape[0]
    indices = np.arange(N)
    np.random.shuffle(indices)
    split_idx = int(N * split)
    train_idx = indices[:split_idx]
    test_idx = indices[split_idx:]
    data_train = data[train_idx]
    labels_train = labels[train_idx]
    data_test = data[test_idx]
    labels_test = labels[test_idx]
    return data_train, labels_train, data_test, labels_test
