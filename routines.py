import os
import numpy as np
import matplotlib.pyplot as plt



def get_fashion_mnist():
    if not os.path.isfile('train-images-idx3-ubyte'):
        os.system('wget http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/train-images-idx3-ubyte.gz')
        os.system('wget http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/train-labels-idx1-ubyte.gz')
        os.system('wget http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/t10k-images-idx3-ubyte.gz')
        os.system('wget http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/t10k-labels-idx1-ubyte.gz')
        os.system('gunzip *.gz')



    with open('train-images-idx3-ubyte', 'rb') as f:
        X = np.frombuffer(f.read(), dtype=np.uint8, offset=16).copy()
        X = X.reshape((60000, 28*28))

    with open('train-labels-idx1-ubyte', 'rb') as f:
        y = np.frombuffer(f.read(), dtype=np.uint8, offset=8)
        
    with open('t10k-images-idx3-ubyte', 'rb') as f:
        X_test = np.frombuffer(f.read(), dtype=np.uint8, offset=16).copy()
        X_test = X_test.reshape((10000, 28*28))

    with open('t10k-labels-idx1-ubyte', 'rb') as f:
        y_test = np.frombuffer(f.read(), dtype=np.uint8, offset=8)
        
    return X, y, X_test, y_test

def visualize_mnist(X, y, field_size=10):
    FIELD_SIZE = field_size

    ind = np.random.randint(0, X.shape[0], FIELD_SIZE*FIELD_SIZE)
    X_vis = X[ind, :].reshape(FIELD_SIZE, FIELD_SIZE, 28, 28) 
    y_vis = y[ind].reshape(FIELD_SIZE, FIELD_SIZE)

    plt.figure(figsize=(8, 8))
    for i in range(FIELD_SIZE):
        for j in range(FIELD_SIZE):
            plt.subplot(FIELD_SIZE, FIELD_SIZE, i*FIELD_SIZE+j+1)
            plt.imshow(X_vis[i, j, :], cmap=plt.cm.Greys)
            plt.gca().xaxis.set_ticks([])
            plt.gca().yaxis.set_ticks([])
            plt.title(y_vis[i, j])

    plt.tight_layout()
    plt.show()
    
    
class Visualizer:
    def __init__(self, size, title, xlabel, ylabel, max_epochs, min_val=None):
        self.fig = plt.figure(figsize=size)
        self.ax = plt.axes()

        self.ax.set_xlabel(xlabel)
        self.ax.set_ylabel(ylabel)
        self.ax.set_title(title)
        self.ax.set_xlim(1,max_epochs)
        self.ax.grid()
        self.min_val = min_val
        
    def update_plot(self, fold_losses, labels):
        if self.min_val is not None:
            self.ax.set_ylim(0, fold_losses.max()+fold_losses.max()*0.1)
        else:
            self.ax.set_ylim(fold_losses.max()*0.8, fold_losses.max()+fold_losses.max()*0.1)
        x = np.arange(1, fold_losses.shape[0]+1)

        if self.ax.lines:
            for i, line in enumerate(self.ax.lines):
                line.set_xdata(x)
                line.set_ydata(fold_losses[:, i])
        else:
            for i, label in enumerate(labels):
                self.ax.plot(x, fold_losses[:, i], label=label)
        self.fig.canvas.draw()
        self.ax.legend()