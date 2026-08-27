import multiprocessing
import queue
import numpy as np
from nn4 import Network

def train_network(X, Y, config, result_queue, training_on):
    model = Network(X, Y, layers=config["layers"], lr=config["lr"], name=config["name"])
    model.run(result_queue.put, training_on)

def main():
    multiprocessing.set_start_method('fork')
    X = np.load("X_data.npy")
    Y = np.load("Y_data.npy")
    
    configs = [
        {'layers': [784, 256, 128, 10], 'lr': 0.001, "name": 1},
        {'layers': [784, 512, 256, 10], 'lr': 0.001, "name": 2},
        {'layers': [784, 256, 128, 10], 'lr': 0.0001, "name": 3},
    ]

    training_on = multiprocessing.Event()
    result_queue = multiprocessing.Queue()

    training_on.set()

    processes = [multiprocessing.Process(target=train_network, args=(X, Y, c, result_queue, training_on)) for c in configs]

    for p in processes:
        p.start()
    
    while training_on.is_set():
        try:
            name, epoch, batch, loss, accuracy = result_queue.get_nowait()
            print(f"Name: {name}")
            print(f"Epoch: {epoch}")
            print(f"Batch: {batch}")
            print(f"Loss: {loss:,}")
            print(f"Acc: {accuracy:,.2%}")
        except queue.Empty:
            pass
    
if __name__ == "__main__":
    main()

