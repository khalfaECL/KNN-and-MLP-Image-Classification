Image Classification

## Author  
Student at École Centrale de Lyon – Deep Learning & Artificial Intelligence  

## Overview  
This work consists of designing and implementing a complete image classification pipeline based on two different approaches: a traditional K-Nearest Neighbors (KNN) method and a Multilayer Perceptron (MLP) neural network. The study is conducted using the CIFAR-10 dataset, which provides a diverse set of labeled images used to train and evaluate both models. The first part focuses on the KNN algorithm, where I implemented an efficient computation of the distance matrix and evaluated its performance as a baseline classifier. The second part extends the work to neural networks, where I developed a one-hidden-layer MLP trained with the cross-entropy loss function and a softmax output layer to handle multi-class predictions.  

Throughout the project, I structured the workflow to include data preparation, model training, accuracy evaluation, and visualization of results. I progressively improved the models by adjusting parameters such as the number of neighbors, learning rate, and number of epochs, in order to observe their influence on the final accuracy. Both methods were trained and tested on separate subsets of the CIFAR-10 dataset to ensure a fair performance comparison.  

The results of the experiments highlight the differences between the two approaches in terms of accuracy and computational behavior. The KNN model, while simple, provides a good reference for understanding basic classification principles, whereas the MLP demonstrates the advantage of learning representations through gradient-based optimization. The evolution of training accuracy over epochs and the final test performance were plotted and saved as `knn.png` and `mlp.png` in the `results` directory.  


## Artificial Neural Network
### 2 – Expression of ∂C/∂A²

$C = \frac{1}{N_{\text{out}}} \sum_i (\hat{y}_i - y_i)^2$ with $\hat{y}_i = A^{(2)}_i$.

Differentiating with respect to each output activation:

$\frac{\partial C}{\partial A^{(2)}_i} = \frac{2}{N_{\text{out}}} (A^{(2)}_i - Y_i)$

In vector form:

$\frac{\partial C}{\partial A^{(2)}} = \frac{2}{N_{\text{out}}} (A^{(2)} - Y)$

### 3 – Expression of ∂C/∂Z²

$\frac{\partial C}{\partial Z^{(2)}} 
= \frac{\partial C}{\partial A^{(2)}} \; \sigma'(Z^{(2)})$

Using $A^{(2)} = \sigma(Z^{(2)})$, this becomes:

$\frac{\partial C}{\partial Z^{(2)}}
= \frac{\partial C}{\partial A^{(2)}} \; \odot \; \left(A^{(2)}(1 - A^{(2)})\right)$

### 4 – Expression of ∂C/∂W²

$\frac{\partial Z^{(2)}_i}{\partial W^{(2)}_{ij}} = A^{(1)}_j$

Using the chain rule:

$\frac{\partial C}{\partial W^{(2)}_{ij}}
= \frac{\partial C}{\partial Z^{(2)}_i} \, A^{(1)}_j$

In matrix form, the gradient is:

$\frac{\partial C}{\partial W^{(2)}}
= \frac{\partial C}{\partial Z^{(2)}} \; (A^{(1)})^{T}$


##" 5 – Expression of ∂C/∂B²

$\frac{\partial Z^{(2)}_i}{\partial B^{(2)}_i} = 1$

by the chain rule:

$\frac{\partial C}{\partial B^{(2)}_i}
= \frac{\partial C}{\partial Z^{(2)}_i}$

In vector form:

$\frac{\partial C}{\partial B^{(2)}}
= \frac{\partial C}{\partial Z^{(2)}}$

### 6 – Expression of ∂C/∂A¹

$\frac{\partial Z^{(2)}_i}{\partial A^{(1)}_j} = W^{(2)}_{ij}$

Using the chain rule:

$\frac{\partial C}{\partial A^{(1)}_j}
= \sum_i \frac{\partial C}{\partial Z^{(2)}_i} \, \frac{\partial Z^{(2)}_i}{\partial A^{(1)}_j}
= \sum_i \frac{\partial C}{\partial Z^{(2)}_i} \, W^{(2)}_{ij}$

In matrix form, this gives:

$\frac{\partial C}{\partial A^{(1)}}
= (W^{(2)})^{T} \, \frac{\partial C}{\partial Z^{(2)}}$


### 7 – Expression of ∂C/∂Z¹

By the chain rule:

$\frac{\partial C}{\partial Z^{(1)}} 
= \frac{\partial C}{\partial A^{(1)}} \odot \sigma'(Z^{(1)})$

Using $A^{(1)} = \sigma(Z^{(1)})$, this becomes:

$\frac{\partial C}{\partial Z^{(1)}}
= \frac{\partial C}{\partial A^{(1)}} \odot \left(A^{(1)} (1 - A^{(1)})\right)$

### 8 – Expression of ∂C/∂W¹

$\frac{\partial Z^{(1)}_i}{\partial W^{(1)}_{ij}} = X_j$

Applying the chain rule:

$\frac{\partial C}{\partial W^{(1)}_{ij}}
= \frac{\partial C}{\partial Z^{(1)}_i} \, X_j$

In matrix form, this gives:

$\frac{\partial C}{\partial W^{(1)}}
= \frac{\partial C}{\partial Z^{(1)}} \; X^{T}$


### 9 – Expression of ∂C/∂B¹

$\frac{\partial Z^{(1)}_i}{\partial B^{(1)}_i} = 1$

and the bias of neuron $i$ does not influence $Z^{(1)}_j$ for $j \neq i$.

By the chain rule:

$\frac{\partial C}{\partial B^{(1)}_i}
= \frac{\partial C}{\partial Z^{(1)}_i}$

In vector form:

$\frac{\partial C}{\partial B^{(1)}}
= \frac{\partial C}{\partial Z^{(1)}}$

In summary, this project allowed me to apply core concepts of supervised learning and neural network training in practice, from algorithmic implementation to experimental evaluation. It provided a clear understanding of how classical and modern methods approach the same classification problem, and how model architecture and learning parameters can significantly impact the outcome.  

## Web App Guide
### Prerequisites
- Python 3.9+
- CIFAR-10 data located at `data/cifar-10-batches-py`

### Install
```bash
pip install -r requirements.txt
```

### Train Models (KNN + MLP)
```bash
python app.py --train --train-only
```

Useful training options:
- `--knn-auto-k` auto-tunes K using a validation split.
- `--knn-weighted` uses distance-weighted voting.
- `--knn-pca 128` uses PCA to reduce noise.
- `--knn-extra-dir test_images --knn-extra-label cat` adds extra images for KNN training.

Example:
```bash
python app.py --train --train-only --knn-auto-k --knn-weighted --knn-pca 128
```

### Run the Web App
```bash
python app.py
```
Open `http://127.0.0.1:5000` in your browser.

### Uploads and Gallery
- Use the upload panel to preview an image.
- Optional: add a label before saving.
- Click "Save to Gallery" to store it on the server.
- Use "Delete" to remove saved images.

### Notes
- The web UI uses MathJax (CDN) to render formulas.
- CIFAR-10 models may not generalize to high-resolution real photos. For better results, add more labeled images or use a stronger model.
