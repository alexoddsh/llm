import queue
import threading
import tkinter as tk
from collections.abc import Callable
from tkinter import font, ttk

import matplotlib
import numpy as np
import numpy.typing as npt
from matplotlib import ticker
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure

type NetworkArray = npt.NDArray[np.float32]

type NetworkData = tuple[NetworkArray, NetworkArray, NetworkArray, NetworkArray]

type ReturnData = tuple[NetworkArray, NetworkArray]

type BatchData = list[NetworkArray]

type ProgressPayload = tuple[int, int, float, float]


class Network:
    def __init__(
        self,
        X: npt.NDArray[np.float32],
        Y: npt.NDArray[np.float32],
        layers: list[int],
        weights: BatchData | None = None,
        m_weights: BatchData | None = None,
        v_weights: BatchData | None = None,
        biases: BatchData | None = None,
        m_biases: BatchData | None = None,
        v_biases: BatchData | None = None,
        batch_size: int = 16,
        lr: float = 0.001,
        beta1: float = 0.9,
        beta2: float = 0.999,
        name: str = "MyNetwork",
    ):

        # Global Lvl Data
        rng = np.random.default_rng()
        self.name = name

        self.X = X
        self.Y = Y
        self.X_train, self.y_train, self.X_val, self.y_val = self.prepare_data()

        self.sizes = layers  # [layer1, layer2, layer3 ...]
        self.batch_size = batch_size
        self.layers = layers
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2

        self.epochs_trained = 0
        self.batches_trained = 0
        self.current_loss = 0
        self.acc = 0

        if weights is None:
            self.weights = [
                rng.normal(loc=0.0, scale=np.sqrt(2 / (in_)), size=(out_, in_)).astype(np.float32)
                for in_, out_ in zip(self.layers[:-1], self.layers[1:])
            ]  # He Init
            self.mw = [np.zeros(w.shape).astype(np.float32) for w in self.weights]
            self.vw = [np.zeros(w.shape).astype(np.float32) for w in self.weights]
        else:
            self.weights = weights
            self.mw = m_weights
            self.vw = v_weights

        if biases is None:
            self.biases = [np.full((out_,), 0.01).astype(np.float32) for out_ in self.layers[1:]]
            self.mb = [np.zeros_like(b).astype(np.float32) for b in self.biases]
            self.vb = [np.zeros_like(b).astype(np.float32) for b in self.biases]
        else:
            self.biases = biases
            self.mb = m_biases
            self.vb = v_biases

    def __str__(self):
        return f"""
            Name: {self.name}
            Epoch: {self.epochs_trained}
            Batch: {self.batches_trained}
            Loss: {self.current_loss:,}
            Acc: {self.acc:,.2%}
        """

    def run(self, queue_data: Callable[ProgressPayload, None], train: threading.Event) -> None:
        while train.is_set():
            y_batches, a_batches = self.epoch_batch_data()

            for A, y in zip(a_batches, y_batches):
                self.batches_trained += 1
                activations = self.train_model(A, y)

                if self.batches_trained % 1000 == 0:
                    self.test_loss(self.X_val, self.y_val)
                    queue_data((self.epochs_trained, self.batches_trained, self.current_loss, self.acc, self.weights, activations))
                    # queue_data((self.name, self.epochs_trained, self.batches_trained, self.current_loss, self.acc))

                if not train.is_set():
                    np.save("model_weights", self.weights)
                    np.save("model_biases", self.biases)
                    break

            if not train.is_set():
                break

            self.epochs_trained += 1

    def prepare_data(self) -> NetworkData:
        # Prepares data only once
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
        # called each epoch, randomises data order to prevent overfit
        indices = np.random.permutation(len(self.X_train))
        x_train, y_train = self.X_train[indices], self.y_train[indices]
        y_batches = np.array_split(y_train, 60000 / self.batch_size, axis=0)
        a_batches = np.array_split(x_train, 60000 / self.batch_size, axis=0)

        return y_batches, a_batches

    def forward(self, A: NetworkArray, train_mode: bool = True) -> ReturnData:
        # A (Batch, in)

        activations = [A]  # Cache every layer
        sums = []

        for i, (W, B) in enumerate(zip(self.weights, self.biases)):
            Z = A @ W.T + B  # Batch, out
            if i == len(self.weights) - 1:
                A = self.softmax(Z)
            else:
                if train_mode:
                    A = self.dropout(self.leaky_relu(Z))
                else:
                    A = self.leaky_relu(Z)

            # Shape [(Batch, out)]
            activations.append(A)
            sums.append(Z)

        return activations, sums

    def backward(self, A_cache: BatchData, Z_cache: BatchData, y: NetworkArray) -> ReturnData:
        gradients_weights = []
        gradients_biases = []

        dZ = A_cache[-1] - y  # dC/dA * dA/dZ equiv dC/dZ (Batch, out) softmax
        _ = Z_cache.pop()

        for i in reversed(range(len(self.weights))):
            dW = (dZ.T @ A_cache[i]) / self.batch_size  # Avg dC/dW across all batches (out, Batch) X (Batch, in)
            dB = dZ.mean(axis=0)  # Avg dC/dB across all batches

            gradients_weights.append(dW)
            gradients_biases.append(dB)

            if i > 0:
                dA_prev = dZ @ self.weights[i]
                dZ = dA_prev * self.prime_leaky_relu(Z_cache.pop())

        return gradients_weights, gradients_biases

    def train_model(self, A: NetworkArray, y: NetworkArray) -> BatchData:
        A_cache, Z_cache = self.forward(A)
        gradients_weights, gradients_biases = self.backward(A_cache, Z_cache, y)

        weights = [
            w - (self.lr / (np.sqrt(v / (1 - self.beta2**self.batches_trained)) + 1e-8)) * (m / (1 - self.beta1**self.batches_trained))
            for w, m, v in zip(self.weights, self.mw, self.vw)
        ]

        biases = [
            b - (self.lr / (np.sqrt(v / (1 - self.beta2**self.batches_trained)) + 1e-8)) * (m / (1 - self.beta1**self.batches_trained))
            for b, m, v in zip(self.biases, self.mb, self.vb)
        ]

        self.weights = weights
        self.biases = biases

        self.mw = [self.beta1 * m + (1 - self.beta1) * g for m, g in zip(self.mw, reversed(gradients_weights))]
        self.vw = [self.beta2 * v + (1 - self.beta2) * g**2 for v, g in zip(self.vw, reversed(gradients_weights))]

        self.mb = [self.beta1 * m + (1 - self.beta1) * g for m, g in zip(self.mb, reversed(gradients_biases))]
        self.vb = [self.beta2 * v + (1 - self.beta2) * g**2 for v, g in zip(self.vb, reversed(gradients_biases))]

        return A_cache

    def test_loss(self, X: NetworkArray, y: NetworkArray) -> None:
        activations, _ = self.forward(X, train_mode=False)
        A = activations[-1]

        mask = y == 1
        self.current_loss = -np.mean(np.log(np.clip(A[mask], 1e-7, 1.0)))  # Cross Entropy Loss

        max_cols = np.argmax(A, axis=1)
        max_mask = np.arange(A.shape[1]) == max_cols[:, None]
        bool_mask = mask & max_mask
        self.acc = np.mean(bool_mask) * 10

    def softmax(self, z):
        e = np.exp(z - np.max(z, axis=1, keepdims=True))  # subtracting the largest numerical stability from overflow
        return e / e.sum(axis=1, keepdims=True)

    def leaky_relu(self, z):
        return np.maximum(z * 0.01, z)  # if Z < 0 return z*0.01

    def prime_leaky_relu(self, z):
        return np.where(z > 0, 1, 0.01)

    def dropout(self, A, p=0.2):
        mask = (np.random.rand(*A.shape) > p).astype(np.float32) / (1 - p)
        return A * mask


class Visualizer:
    def __init__(self, root, network: Network):
        self.network = network
        self.training_on = threading.Event()
        self.queue = queue.Queue()

        self.batches_history = [0]
        self.loss_history = [0.0]
        self.accuracy_history = [0.0]
        self.dead_neurons_history = [0]

        # styles
        style = ttk.Style()
        header_font = font.Font(family="Monaco", size=16, weight="bold")
        style.configure("Header.TLabel", font=header_font)

        # layout
        self.root = root
        self.root.geometry("1000x600")

        main_container = ttk.Frame(root, padding=10)
        main_container.pack(fill=tk.BOTH, expand=True)

        header_bar = ttk.Frame(main_container, padding=5)
        header_bar.pack(side=tk.TOP, fill=tk.X)

        self.acc_label = ttk.Label(header_bar, text="--%", style="Header.TLabel")
        self.acc_label.pack(side=tk.LEFT)
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

        # charts
        self.fig = Figure(figsize=(10, 8), layout="tight")
        self.ax = self.fig.subplots(2, 2)

        self.ax[0, 0].set_title("Total Loss (C)")
        self.ax[0, 0].set_xlabel("Batch Number")
        self.ax[0, 0].set_ylabel("Loss Value")
        self.ax[0, 0].xaxis.set_major_formatter(ticker.StrMethodFormatter("{x:,.0}"))
        (self.loss_line,) = self.ax[0, 0].plot([], [], color="pink")

        self.ax[0, 1].set_title("Accuracy Val Data (%)")
        self.ax[0, 1].set_xlabel("Batch Number")
        self.ax[0, 1].set_ylabel("Accuracy (%)")
        self.ax[0, 1].xaxis.set_major_formatter(ticker.StrMethodFormatter("{x:,.0}"))
        self.ax[0, 1].yaxis.set_major_formatter(ticker.StrMethodFormatter("{x:.2%}"))
        (self.accuracy_line,) = self.ax[0, 1].plot([], [], color="green")

        self.ax[1, 0].set_title("Weight Development")
        self.ax[1, 0].set_ylabel("Avg Val")
        cats = [str(layer) for layer in self.network.layers[:-1]]
        self.wbar = self.ax[1, 0].bar(cats, [0] * (len(self.network.layers) - 1), color="skyblue")

        self.ax[1, 1].set_title("Activations and Gradients")
        self.ax[1, 1].set_xlabel("Batch Number")
        (self.dead_line,) = self.ax[1, 1].plot([], [], color="red")

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
                epoch, batch, loss, accuracy, weights, activations = self.queue.get_nowait()
                self.update_charts(epoch, batch, loss, accuracy, weights, activations)
        except queue.Empty:
            pass

        self.root.after(100, self.listen_to_queue)

    def update_charts(self, epoch: int, batch: int, loss: float, accuracy: float, weights: BatchData, activations: BatchData) -> None:
        self.batches_history.append(batch)
        self.loss_history.append(loss)
        self.accuracy_history.append(accuracy)

        self.epoch_label.config(text=f"Epoch: {epoch}")
        self.batch_label.config(text=f"Batch: {batch}")
        self.acc_label.config(text=f"Acc: {accuracy:.2%}")

        self.loss_line.set_data(self.batches_history, self.loss_history)
        self.accuracy_line.set_data(self.batches_history, self.accuracy_history)

        for bar, w in zip(self.wbar, weights):
            bar.set_height(np.mean(w))

        dead_neurons = 0
        for a in activations:
            dead_neurons += np.count_nonzero(np.all(a == 0, axis=0))
        self.dead_neurons_history.append(dead_neurons)
        self.dead_line.set_data(self.batches_history, self.dead_neurons_history)

        self.ax[0, 0].relim()
        self.ax[0, 1].relim()
        self.ax[1, 0].relim()
        self.ax[1, 1].relim()

        self.ax[0, 0].autoscale_view()
        self.ax[0, 1].autoscale_view()
        self.ax[1, 0].autoscale_view()
        self.ax[1, 1].autoscale_view()

        print(self.network)
        self.canvas.draw_idle()

    def stop_training(self):
        self.stop_btn.config(state=tk.DISABLED)
        self.start_btn.config(state=tk.NORMAL)
        self.training_on.clear()

    def start_training(self):
        self.start_btn.config(state=tk.DISABLED)
        self.stop_btn.config(state=tk.NORMAL)
        self.training_on.set()

        t = threading.Thread(target=self.network.run, args=(self.queue.put, self.training_on))
        t.start()


if __name__ == "__main__":
    root = tk.Tk()
    matplotlib.use("TkAgg")

    X = np.load("X_data.npy")
    Y = np.load("Y_data.npy")

    lr = 0.001
    batch_size = 16
    layers = [784, 256, 128, 10]

    # instances
    model = Network(X=X, Y=Y, layers=layers, batch_size=batch_size, lr=lr)
    visuals = Visualizer(root=root, network=model)

    root.mainloop()
