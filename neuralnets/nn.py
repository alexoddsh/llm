import tkinter as tk
from tkinter import ttk
from tkinter import font

import numpy as np
import numpy.typing as npt

import matplotlib
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

matplotlib.use("TkAgg")
rng = np.random.default_rng()

class Network:
    def __init__(self, 
            layers: list[int], 
            weights: npt.NDArray[np.float32] | None = None, 
            biases: npt.NDArray[np.float32] | None = None, 
            batch_size: int = 16, 
            lr: float = 0.001
        ):
        
        self.sizes = layers # [layer1, layer2, layer3 ...]
        self.batch_size = batch_size
        self.layers = layers
        self.lr = lr

        self.epochs_trained = 0
        self.batches_trained = 0
        self.current_loss = 0
        
        if weights is None:
            self.weights = [rng.normal(loc=0.0, scale=(1/(in_*out_)), size=(out_, in_)) for in_, out_ in zip(self.layers[:-1], self.layers[1:])]
        else:
            self.weights = weights

        if biases is None:
            self.biases = [np.full((out_, ), 1) for out_ in self.layers[1:]]
        else:
            self.biases = biases

    def sigmoid(self, z):
        return 1 / (1+np.exp(-z))

    def prime_sigmoid(self, z):
        s = self.sigmoid(z)
        return s * (1-s)

    def forward(self, A):
        #A (Batch, in)
        
        activations = [A] #cache A^L-1
        
        for W, B in zip(self.weights, self.biases):
            Z = A @ W.T + B
            A = self.sigmoid(Z)
            
            activations.append(A)
        
        return activations

    def backward(self, A_cache, y):
        gradients_weights = []
        gradients_biases = []
        
        dZ = 2*(A_cache[-1] - y) * A_cache[-1]*(1-A_cache[-1]) #dC/dA * dA/dZ equiv dC/dZ (Batch, out)
        
        for i in reversed(range(len(self.sizes)-1)): 
            dW = (dZ.T @ A_cache[i]) / self.batch_size #Avg dC/dW across all batches (out, Batch) X (Batch, in)
            dB = dZ.mean(axis=0, keepdims=True) #Avg dC/dB across all batches
            
            gradients_weights.append(dW)
            gradients_biases.append(dB)

            if i>0:
                dA_prev = dZ @ self.weights[i] 
                dZ = dA_prev * A_cache[i] * (1-A_cache[i])

        return gradients_weights, gradients_biases

    def train_model(self, A, y):
        A_cache = self.forward(A)
        gradients_weights, gradients_biases = self.backward(A_cache, y)
            
        weights = [w - self.lr * g for w, g in zip(self.weights, reversed(gradients_weights))]
        biases = [b - self.lr * b for w, b in zip(self.biases, reversed(gradients_biases))]

        return weights, biases
    
    def test_loss(self, A, y):
        loss = np.sum((A - y)) / 10000 
        
        mask = (y == 1)
        max_cols = np.argmax(A, axis=1)
        max_mask = (np.arange(A.shape[1]) == max_cols[:, None])
        bool_mask = mask & max_mask
        acc = np.mean(bool_mask)

        return loss, acc
    
        
class Visualizer:
        def __init__(self, network):
            self.network = network
            self.training_ongoing = False
            
            self.loss_data = []
            self.accuracy_data = []
            self.batch_tick = []

            #styles
            style = ttk.Style()
            header_font = font.Font(family="Monaco", size=16, weight="bold")
            style.configure("Header.TLabel", font=header_font)

            #layout
            root = tk.Tk()
            root.geometry("1000x600")

            main_container = ttk.Frame(root, padding=10)
            main_container.pack(fill=tk.BOTH, expand=True)
            ttk.Label(main_container, text="Neural Network Training", style="Header.TLabel")

            header_bar = ttk.Frame(main_container, padding=5)
            header_bar.pack(side=tk.TOP, fill=tk.X)

            ttk.Label(header_bar, text=f"Epoch: {network.epochs_trained}").pack(side=tk.RIGHT, padx=2)
            ttk.Label(header_bar, text=f"Batch: {network.batches_trained}").pack(side=tk.RIGHT, padx=2)
            self.start_btn = ttk.Button(header_bar, text="Start", command=self.start_training).pack(side=tk.RIGHT, padx=2)
            self.stop_btn = ttk.Button(header_bar, text="Stop", command=self.stop_training).pack(side=tk.RIGHT, padx=2)
            ttk.Button(header_bar, text="Quit", command=root.destroy).pack(side=tk.RIGHT, padx=2)

            content_area = ttk.Frame(main_container, padding=10)
            content_area.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

            self.fig = Figure(figsize=(10, 8), layout="tight")
            self.ax = self.fig.subplots(2, 2)

            self.ax[0, 0].set_title("Total Loss (C)")
            self.ax[0, 0].set_xlabel("Batch Number")
            self.ax[0, 0].set_ylabel("Loss Value")
            self.loss_line,  = self.ax[0, 0].plot([], [])

            self.ax[0, 1].set_title("Accuracy Val Data (%)")
            self.ax[0, 1].set_xlabel("Batch Number")
            self.ax[0, 1].set_ylabel("Accuracy (%)")
            self.accuracy_line,  = self.ax[0, 1].plot([], [])

            self.ax[1, 0].set_title("Weight Development")
            self.ax[1, 0].set_xlabel("")

            self.ax[1, 1].set_title("Bias Development")
            self.ax[1, 1].set_xlabel("")

            self.canvas = FigureCanvasTkAgg(self.fig, master=content_area)
            self.canvas_widget = self.canvas.get_tk_widget()
            self.canvas_widget.pack(side=tk.TOP, fill=tk.BOTH, expand=True, padx=5, pady=5)
            self.canvas.draw()

            self.update_charts()

            root.mainloop()
            print("visuals created")

        def update_charts(self):
            self.loss_line.set_data(self.batch_tick, self.loss_data)
            self.accuracy_line.set_data(self.batch_tick, self.accuracy_data)

            self.ax[0, 0].relim()
            self.ax[0, 1].relim()
            self.ax[1, 0].relim()
            self.ax[1, 1].relim()

            self.ax[0, 0].autoscale_view()
            self.ax[0, 1].autoscale_view()
            self.ax[1, 0].autoscale_view()
            self.ax[1, 1].autoscale_view()

            self.canvas.draw()

        def stop_training(self):
            self.training_ongoing = False
            self.stop_btn.config(state=tk.DISABLED)
            self.start_btn.config(state=tk.ENABLED)
        
        def start_training(self):
            self.training_stopped = True
            self.start_btn.config(state=tk.DISABLED)
            self.stop_btn.config(state=tk.ENABLED)

def main():
    X = np.load("X_data.npy")
    Y = np.load("Y_data.npy")
    X_train, X_val = X[:60000], X[60001:70000]
    Y_train, Y_val = Y[:60000], Y[60001:70000]
    
    labels_train = Y_train.astype(int)
    labels_val = Y_val.astype(int)
    size_train = len(labels_train)
    size_val = len(labels_train)
    
    y_train = np.zeros((size_train, 10))
    y_train[np.arange(size_train), labels_train] = 1.0
    y_val = np.zeros((size_val, 10))
    y_val = [np.arange(size_val), labels_val]
    
    #params
    lr = 0.001
    batch_size = 16
    layers = [784, 128, 16, 10]
    
    #instances
    model = Network(layers=layers, batch_size=batch_size, lr=lr)
    visuals = Visualizer(network=model)
    
    while visuals.training_ongoing:
        indices = np.random.permutation(len(X_train))
        x_train, y_train = X_train[indices], y_train[indices]

        y_batches = np.array_split(y_train, 60000/model.batch_size, axis=0) 
        a_batches = np.array_split(x_train, 60000/model.batch_size, axis=0)
    
        for A, y in zip(a_batches, y_batches):
            print("a")
            w, b = model.train_model(A, y)
            c, a = model.test_loss(X_val, y_val)

            model.weights = w
            model.biases = b
            model.batches_trained +=1
            model.current_loss = c

            visuals.loss_data.append(model.current_loss)
            visuals.accuracy_data.append(a)
            visuals.batch_tick.append(model.batches_trained)
                
            if not visuals.training_ongoing:
                np.save("model_weights", model.weights)
                np.save("model_biases", model.biases)
                break
        
        model.epochs_trained +=1
        visuals.update_charts()

        if not visuals.training_ongoing:
            break
    
if __name__ == "__main__":
    main()
    
