import numpy as np
from sklearn.datasets import fetch_openml

def load_mnist():
    #load
    mnist = fetch_openml("mnist_784", version=1, as_frame=False, parser="pandas") 

    #cast to float32 for speed and normalize X values to 0-1
    X = mnist.data.astype(np.float32) / 255 
    y = mnist.target.astype(np.float32)

    np.save("X_data.npy", X)
    np.save("Y_data.npy", y)
    
load_mnist()
