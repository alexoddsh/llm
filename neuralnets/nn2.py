from typing import Callable, TypeAlias
import tkinter as tk
from tkinter import ttk
from tkinter import font

import numpy as np
import numpy.typing as npt

import matplotlib
import matplotlib.ticker as ticker
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk

import threading
import queue

matplotlib.use("TkAgg")
rng = np.random.default_rng()

#to do
#1. optimizer Adam
#2. He initialize?
#3. GUI
#4. performance
#.5 droput
# batchnorm


#5. CNN
#6. LLM

NetworkData: TypeAlias = tuple[
    npt.NDArray[np.float32], npt.NDArray[np.float32], 
    npt.NDArray[np.float32], npt.NDArray[np.float32]
]

ReturnData: TypeAlias = tuple[
    npt.NDArray[np.float32],
    npt.NDArray[np.float32]
]

ProgressPayload: TypeAlias = tuple[int, int, float, float]

class Network:
    def __init__(self, 
            X: npt.NDArray[np.float32],
            Y: npt.NDArray[np.float32],
            layers: list[int], 
            weights: npt.NDArray[np.float32] | None = None, 
            biases: npt.NDArray[np.float32] | None = None, 
            batch_size: int = 16, 
            lr: float = 0.001
        ):

        #Global Lvl Data
        self.X = X
        self.Y = Y
        self.X_train, self.y_train, self.X_val, self.y_val = self.prepare_data()

        self.sizes = layers # [layer1, layer2, layer3 ...]
        self.batch_size = batch_size
        self.layers = layers
        self.lr = lr

        self.epochs_trained = 0
        self.batches_trained = 0
        
        if weights is None:
            self.weights = [rng.normal(loc=0.0, scale=np.sqrt(2/(in_)), size=(out_, in_)) 
            for in_, out_ in zip(self.layers[:-1], self.layers[1:])] #Xavier 
        else:
            self.weights = weights

        if biases is None:
            self.biases = [np.full((out_, ), 0.01) for out_ in self.layers[1:]]
        else:
            self.biases = biases
    
    def run(self, queue_data: Callable[ProgressPayload, None], train: threading.Event) -> None:
        while train.is_set():
            y_batches, a_batches = self.epoch_batch_data()
            tick = 0
            
            for A, y in zip(a_batches, y_batches):
                self.train_model(A, y)
                                        
                if tick % 500 == 0:
                    self.batches_trained += self.batch_size*500
                    c, a = self.test_loss(self.X_val, self.y_val)
                    queue_data((self.epochs_trained, self.batches_trained, c, a))
                    
                if not train.is_set():
                    np.save("model_weights", self.weights)
                    np.save("model_biases", self.biases)
                    break

                tick += 1
                    
            if not train.is_set():
                break
            
            self.epochs_trained += 1 
    
    def prepare_data(self) -> NetworkData:
        #Prepares data only once
        X_train, X_val = self.X[:60000], self.X[60001:70000]
        Y_train, Y_val = self.Y[:60000], self.Y[60001:70000]
        
        labels_train = Y_train.astype(int)
        labels_val = Y_val.astype(int)

        size_train = len(labels_train)
        size_val = len(labels_val)
        y_train = np.zeros((size_train, 10))
        y_train[np.arange(size_train), labels_train] = 1.0
        
        y_val = np.zeros((size_val, 10))
        y_val[np.arange(size_val), labels_val] = 1.0

        return X_train, y_train, X_val, y_val
        
    def epoch_batch_data(self) -> ReturnData:
        #called each epoch, randomises data order to prevent overfit
        indices = np.random.permutation(len(self.X_train))
        x_train, y_train = self.X_train[indices], self.y_train[indices]
        y_batches = np.array_split(y_train, 60000/self.batch_size, axis=0) 
        a_batches = np.array_split(x_train, 60000/self.batch_size, axis=0)

        return y_batches, a_batches

    def forward(self, A: npt.NDArray[np.float32]) -> ReturnData:
        #A (Batch, in)
        
        activations = [A] #Cache every layer
        sums = []
        
        for i, (W, B) in enumerate(zip(self.weights, self.biases)):
            Z = A @ W.T + B 
            if i == len(self.weights) - 1:
                A = self.sigmoid(Z)
            else:
                A = self.leaky_relu(Z)
            
            #Shape [(Batch, out)]
            activations.append(A)
            sums.append(Z)
        
        return activations, sums

    def backward(self, A_cache: npt.NDArray[np.float32], Z_cache: npt.NDArray[np.float32], y: npt.NDArray[np.float32]) -> ReturnData:
        gradients_weights = []
        gradients_biases = []
        
        dZ = 2*(A_cache[-1] - y) * self.prime_sigmoid(Z_cache[-1]) #dC/dA * dA/dZ equiv dC/dZ (Batch, out)
        _ = Z_cache.pop()
        
        for i in reversed(range(len(self.weights))): 
            dW = (dZ.T @ A_cache[i]) / self.batch_size #Avg dC/dW across all batches (out, Batch) X (Batch, in)
            dB = dZ.mean(axis=0) #Avg dC/dB across all batches
            
            gradients_weights.append(dW)
            gradients_biases.append(dB)

            if i>0:
                dA_prev = dZ @ self.weights[i] 
                dZ = dA_prev * self.prime_leaky_relu(Z_cache.pop())

        return gradients_weights, gradients_biases

    def train_model(self, A: npt.NDArray, y: npt.NDArray) -> None:
        A_cache, Z_cache = self.forward(A)
        gradients_weights, gradients_biases = self.backward(A_cache, Z_cache, y)
            
        weights = [w - self.lr * g for w, g in zip(self.weights, reversed(gradients_weights))]
        biases = [b - self.lr * g for b, g in zip(self.biases, reversed(gradients_biases))]

        self.weights = weights
        self.biases = biases
    
    def test_loss(self, X: npt.NDArray, y: npt.NDArray) -> tuple[float, float]:
        activations, _ = self.forward(X)
        A = activations[-1]
        
        loss = np.mean((A - y)**2) 
        
        mask = (y == 1)
        max_cols = np.argmax(A, axis=1)
        max_mask = (np.arange(A.shape[1]) == max_cols[:, None])
        bool_mask = mask & max_mask
        acc = np.mean(bool_mask) * 10

        return loss, acc    
    
    def sigmoid(self, z):
        return 1 / (1+np.exp(-z))

    def prime_sigmoid(self, z):
        s = self.sigmoid(z)
        return s * (1-s)

    def leaky_relu(self, z):
        return np.maximum(z*0.01, z) #if Z < 0 return z*0.01
    
    def prime_leaky_relu(self, z):
        return np.where(z > 0, 1, 0.01)
        
class Visualizer:
        def __init__(self, root, network):
            self.network = network
            self.training_on = threading.Event()
            self.queue = queue.Queue()
            
            self.batches_history = [0]
            self.loss_history = [0]
            self.accuracy_history = [0]

            #styles
            style = ttk.Style()
            header_font = font.Font(family="Monaco", size=16, weight="bold")
            style.configure("Header.TLabel", font=header_font)

            #layout
            self.root = root
            self.root.geometry("1000x600")

            main_container = ttk.Frame(root, padding=10)
            main_container.pack(fill=tk.BOTH, expand=True)

            header_bar = ttk.Frame(main_container, padding=5)
            header_bar.pack(side=tk.TOP, fill=tk.X)

            ttk.Label(header_bar, text="Neural Network Training", style="Header.TLabel").pack(side=tk.LEFT)
            self.epoch_label = ttk.Label(header_bar, text="Epoch: 0")
            self.epoch_label.pack(side=tk.RIGHT, padx=2)
            
            self.batch_label = ttk.Label(header_bar, text="Batch: 0")
            self.batch_label.pack(side=tk.RIGHT, padx=2)
            
            self.start_btn = ttk.Button(header_bar, text="Start", command=self.start_training)
            self.start_btn.pack(side=tk.RIGHT, padx=2)
            
            self.stop_btn = ttk.Button(header_bar, text="Stop", command=self.stop_training)
            self.stop_btn.pack(side=tk.RIGHT, padx=2)
            ttk.Button(header_bar, text="Quit", command=self.root.destroy).pack(side=tk.RIGHT, padx=2)

            content_area = ttk.Frame(main_container, padding=10)
            content_area.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

            #charts
            self.fig = Figure(figsize=(10, 8), layout="tight")
            self.ax = self.fig.subplots(2, 2)

            self.ax[0, 0].set_title("Total Loss (C)")
            self.ax[0, 0].set_xlabel("Batch Number")
            self.ax[0, 0].set_ylabel("Loss Value")
            self.ax[0, 0].xaxis.set_major_formatter(ticker.StrMethodFormatter("{x:,.0}"))
            self.loss_line,  = self.ax[0, 0].plot([], [], color="red")

            self.ax[0, 1].set_title("Accuracy Val Data (%)")
            self.ax[0, 1].set_xlabel("Batch Number")
            self.ax[0, 1].set_ylabel("Accuracy (%)")
            self.ax[0, 1].xaxis.set_major_formatter(ticker.StrMethodFormatter("{x:,.0}"))
            self.ax[0, 1].yaxis.set_major_formatter(ticker.StrMethodFormatter("{x:.2%}"))
            self.accuracy_line,  = self.ax[0, 1].plot([], [], color="green")

            self.ax[1, 0].set_title("Weight Development")
            self.ax[1, 0].set_xlabel("")

            self.ax[1, 1].set_title("Bias Development")
            self.ax[1, 1].set_xlabel("")

            self.canvas = FigureCanvasTkAgg(self.fig, master=content_area)
            self.canvas_widget = self.canvas.get_tk_widget()
            toolbar = NavigationToolbar2Tk(self.canvas, content_area)
            toolbar.update()
            
            self.canvas_widget.pack(side=tk.TOP, fill=tk.BOTH, expand=True, padx=5, pady=5)
            self.canvas.draw()

            self.listen_to_queue()
        
        def listen_to_queue(self) -> None:
            try:
                while True:
                    epoch, batch, loss, accuracy = self.queue.get_nowait()
                    self.update_charts(epoch, batch, loss, accuracy)
            except queue.Empty:
                pass
    
            self.root.after(100, self.listen_to_queue)

        def update_charts(self, epoch: int, batch: int, loss: float, accuracy: float) -> None:
            self.batches_history.append(batch)
            self.loss_history.append(loss)
            self.accuracy_history.append(accuracy)

            self.epoch_label.config(text=f"Epoch: {epoch}")
            self.batch_label.config(text=f"Batch: {batch}")
            
            self.loss_line.set_data(self.batches_history, self.loss_history)
            self.accuracy_line.set_data(self.batches_history, self.accuracy_history)

            self.ax[0, 0].relim()
            self.ax[0, 1].relim()
            self.ax[1, 0].relim()
            self.ax[1, 1].relim()

            self.ax[0, 0].autoscale_view()
            self.ax[0, 1].autoscale_view()
            self.ax[1, 0].autoscale_view()
            self.ax[1, 1].autoscale_view()

            self.canvas.draw_idle()

        def stop_training(self):
            self.stop_btn.config(state=tk.DISABLED)
            self.start_btn.config(state=tk.NORMAL)
            self.training_on.clear()
        
        def start_training(self):
            self.start_btn.config(state=tk.DISABLED)
            self.stop_btn.config(state=tk.NORMAL)
            self.training_on.set()

            t = threading.Thread(
                target=self.network.run,
                args=(self.queue.put, self.training_on)
            )
            t.start()
    
if __name__ == "__main__":
    root = tk.Tk()
    
    X = np.load("X_data.npy")
    Y = np.load("Y_data.npy")
    
    lr = 0.001
    batch_size = 16
    layers = [784, 128, 16, 10]
    
    #instances
    model = Network(X=X, Y=Y, layers=layers, batch_size=batch_size, lr=lr)
    visuals = Visualizer(root=root, network=model)
    
    root.mainloop()
    
