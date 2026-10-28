import textwrap
from dataclasses import dataclass

import numpy as np
from numpy import float32
from numpy.typing import NDArray

NPArr = NDArray[float32]

class DataError(Exception):
    """Malformed or insufficient data"""

class NeuralNetwork:
    def __init__(
        self,
        data: TrainingData,
        layers: list[tuple[int, int]] | None = None,
        weights: list[NPArr] | None = None,
        biases: list[NPArr] | None = None,
        learning_rate: float = 0.01,
    ):
        self.data = data
        self.lr = learning_rate

        self.activations = {}
        self.logits = {}

        self.history = {
            "losses": [],
            "accuracies": [],
        }

        self.batches_trained = 0
        self.epochs_trained = 0

        self.two_dimensional_y = None

        if weights:
            self.weights = weights
        elif layers:
            self.weights = [
                np.random.normal(loc=0, scale=self.he_init(inn), size=(inn, out))
                for (inn, out) in layers
            ]
        else:
            raise DataError("No weights or layers provided to instantiate them")

        if biases:
            self.biases = biases
        elif layers:
            self.biases = [
                np.full((inn,), 0).astype(float32) for (inn, _) in layers[1:]
            ]
            self.biases.append(np.full((layers[-1][-1],), 0).astype(float32))
        else:
            raise DataError("No biases or layers provided to instantiate them")

        if self.data.x_train.shape[1] == self.weights[0].shape[1]:
            raise DataError("Training data shape does not match shape of weights")

        if len(self.data.x_train) != len(self.data.y_train):
            raise DataError("Input and target training data are not same length")

        self.epoch_size = len(self.data.x_train)
        self.log_steps = [k*50 for k in range(1, int(self.epoch_size / self.data.batch_size))]

    def train_network(self) -> None:
        self.curr_batch_xdata, self.curr_batch_ydata = self.data.batch_data(self.batches_trained)
        self.forward(validation=False)
        gw, gb = self.backward()
        self.update_params(gw, gb)
        self.log_state()

        self.batches_trained += 1
        if self.batches_trained in self.log_steps:
            print(self.log_batch(gw))

        print(self.batches_trained)

        if self.batches_trained * self.data.batch_size >= self.epoch_size:
            self.batches_trained = 0
            self.epochs_trained += 1
            print(self.log_epoch())

    def forward(self, validation: bool) -> None:
        if validation:
            self.activations[0] = self.data.x_test
        else:
            self.activations[0] = self.curr_batch_xdata

        for i, (w, b) in enumerate(zip
                                       (self.weights, self.biases), 1):
            z = self.activations[i - 1] @ w + b
            self.logits[i] = z
            if i == len(self.weights):
                self.activations[i] = self.softmax(z)
            else:
                self.activations[i] = self.relu(z)

            # (60_000, 784) * (784, 256)
            # (60_000, 256) + (256, )

    def backward(self) -> tuple[list[NPArr], ...]:
        gradients_weights = []
        gradients_biases = []

        dl_dz = self.loss_gradient()  # (batch, out)

        for i in reversed(range(len(self.weights))):
            # (out, in) = (out, batch) * (batch, in), avg dL/dW of all batches
            dl_dw = (dl_dz.T @ self.activations[i]) / self.data.batch_size

            # (out, )
            dl_db = dl_dz.mean(axis=0)

            gradients_weights.append(dl_dw)
            gradients_biases.append(dl_db)

            if i > 0:
                dl_da = dl_dz @ self.weights[i].T  # (batch, in)
                dl_dz = dl_da * self.prime_relu(list(self.logits.values())[i - 1])

        return gradients_weights, gradients_biases

    def loss_gradient(self) -> NPArr:
        p = list(self.activations.values())[-1]
        self.two_dimensional_y = self.reshape_y(self.curr_batch_ydata)
        return p - self.two_dimensional_y  # (p - y)

    def update_params(self, gw: list[NPArr], gb: list[NPArr]) -> None:
        for i in range(len(self.weights)):
            self.weights[len(self.weights) - 1 - i] -= self.lr * gw[i].T
            self.biases[len(self.weights) - 1 - i] -= self.lr * gb[i].T

    @staticmethod
    def he_init(n: int) -> float:
        return np.sqrt(2.0 / n)

    @staticmethod
    def softmax(z: NPArr) -> NPArr:
        exp = np.exp(z - np.max(z, axis=1, keepdims=True))  # (batch, out)
        return (exp / np.sum(exp, axis=1, keepdims=True)).astype(float32)

    @staticmethod
    def relu(z: NPArr) -> NPArr:
        return np.maximum(0, z)

    @staticmethod
    def prime_relu(z: NPArr) -> NPArr:
        return np.where(z > 0, 1, 0.01)

    @staticmethod
    def reshape_y(y: NPArr) -> NPArr:
        n = len(y)
        arr = np.zeros((n, 10), dtype=int)
        arr[np.arange(n), y.astype(int)] = 1
        return arr
    
    def calculate_acc(self, y: NPArr):
        p = list(self.activations.values())[-1]
        max_mask = np.argmax(p, axis=1)
        corr_mask = np.argmax(y, axis=1)
        acc = np.sum(max_mask == corr_mask) / len(y)
        return acc
    
    def calculate_loss(self, y: NPArr):
        p = list(self.activations.values())[-1]
        p_clipped = np.clip(p, 1e-12, 1)
        loss = -np.sum(y * np.log(p_clipped)) / self.data.batch_size
        return  loss
    
    def log_state(self) -> None:    
        self.history["accuracies"].append(self.calculate_acc(self.two_dimensional_y))
        self.history["losses"].append(self.calculate_loss(self.two_dimensional_y))

    def validate_network(self) -> tuple[float, float]:
        self.forward(validation=True)

        y = self.reshape_y(self.data.y_test)
        acc = self.calculate_acc(y)
        loss = self.calculate_loss(y)
        return loss, acc

    def log_epoch(self) -> str:
        losses = self.history["losses"]
        accuracies = self.history["accuracies"]
        epoch_batches = int(self.epoch_size / self.data.batch_size)

        loss, vacc = self.validate_network()

        return textwrap.dedent(f"""
        Training Accuracy (Epoch Avg): {sum(accuracies[-epoch_batches:]) / epoch_batches}
        Validation Accuracy (Validation Avg): {vacc}

        Training Loss (Epoch Avg): {sum(losses[-epoch_batches:]) / epoch_batches}
        Validation Loss (Validation Avg): {loss}
        """)

    def log_batch(self, gw: list[NPArr]) -> str:
        gw = list(reversed(gw))

        return textwrap.dedent(f"""
        Step: {self.batches_trained} 
        Loss: {self.calculate_loss(self.two_dimensional_y):3f}
        
        Gradients |W| (per layer): {[f"Layer {i+1}: {np.linalg.norm(grad):4f}" for i, grad in enumerate(gw)]}
        Update Ratio |GW|/|W| (per layer): {[f"Layer {i+1}: {np.linalg.norm(self.lr * gw[i].T) / np.linalg.norm(self.weights[i]):4f}"  
                                      for i in range(len(self.weights))]}
                                      
        Activations (mean, per layer): {[f"Layer {i}: {np.mean(act):4f}" for i, act in enumerate(self.activations.values())]}
        Activations (std, per layer): {[f"Layer {i}: {np.std(act):4f}" for i, act in enumerate(self.activations.values())]}
        
        Dead Units (fraction of batch): {sum([np.sum(act==0) for act in self.activations.values()]) 
                                         / sum([act.shape[0] * act.shape[1] for act in self.activations.values()]):3f} 
        """)

    def __str__(self) -> str:
        return textwrap.dedent(f"""
        Weights: {[w.shape for w in self.weights]}
        Biases: {[b.shape for b in self.biases]}
        Batch Data: {[self.curr_batch_xdata.shape, self.curr_batch_ydata.shape]}
        Logits: {[l.shape for l in self.logits.values()]}
        Activations: {[a.shape for a in self.activations.values()]}

        Main Data: {[d.shape for d in vars(self.data).values() if isinstance(d, np.ndarray)]}
        """)

    def debug_dtypes(self) -> str:
        return textwrap.dedent(f"""
        Weights: {[w.dtype for w in self.weights]}
        Biases: {[b.dtype for b in self.biases]}
        Batch Data: {[self.curr_batch_xdata.dtype, self.curr_batch_ydata.dtype]}
        Logits: {[l.dtype for l in self.logits.values()]}
        Activations: {[a.dtype for a in self.activations.values()]}
        """)


@dataclass
class TrainingData:
    x_train: NPArr
    y_train: NPArr
    x_test: NPArr
    y_test: NPArr
    
    batch_size: int

    @classmethod
    def load_data(cls, x_path: str, y_path: str, batch_size: int) -> TrainingData:
        try:
            x = np.load(x_path, mmap_mode="r")
            y = np.load(y_path, mmap_mode="r")
        except OSError, ValueError:
            raise DataError("X/Y Data cannot be found or contains object arrays")
        else:
            xt, yt, x, y = cls.split_data(x, y)
        return cls(xt, yt, x, y, batch_size)

    @staticmethod
    def split_data(x: NPArr, y: NPArr) -> tuple[NPArr, ...]:
        x_train = x[:-200, :]
        y_train = y[:-200]
        x_test = x[-200:, :]
        y_test = y[-200: ]
        return x_train, y_train, x_test, y_test
    
    def batch_data(self, batches_trained) -> tuple[NPArr, ...]:
        x = self.x_train[batches_trained*self.batch_size: (1+batches_trained)*self.batch_size, :]
        y = self.y_train[batches_trained*self.batch_size: (1+batches_trained)*self.batch_size]
        return self.normalize_data(x), self.normalize_data(y)

    @staticmethod
    def normalize_data(x: NPArr) -> NPArr:
        return ((x - np.mean(x))/np.std(x)).astype(np.float32)

def main():
    layers = [(180_000, 1_000), (1_000, 1_000), (1_000, 1_000), (1_000, 1_000), (1_000, 10)]
    data = TrainingData.load_data("images.npy", "output.npy", batch_size=32)
    nn = NeuralNetwork(data, layers, learning_rate=0.05)

    while True:
        try:
            nn.train_network()
        except KeyboardInterrupt:
            ipt = input("Save weights? (y/n): ")
            if ipt == "y":
                np.savez("weights.npz", **{f"w_{i}": w for i, w in enumerate(nn.weights)}) # type: ignore
            break

if __name__ == "__main__":
    main()
