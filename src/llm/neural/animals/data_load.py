import random
from pathlib import Path

import modal
import numpy as np
from numpy.lib.format import open_memmap
from numpy.typing import NDArray
from PIL import Image

PATH = Path("Desktop/kagglehub/datasets/animals")

class Loader:
    def __init__(self):
        try:
            with open("images.txt", "r") as file:
                self.images = [Path(img.strip()) for img in file]
        except FileNotFoundError:
            self.images = [p for p in PATH.rglob("*")
                           if p.is_file() and p.suffix == ".jpg" or p.suffix == ".jpeg"]
        random.shuffle(self.images)

        n, m = len(self.images), 300*200*3
        self.images_array = open_memmap("images.npy", mode="w+", dtype=np.float32, shape=(n, m))
        self.output_array = open_memmap("output.npy", mode="w+", dtype=np.float32, shape=(n, ))
        self.chunked_insert(n)

    def chunked_insert(self, n: int) -> None:
        for start in range(0, n, 100):
            stop = min(start + 100, n)
            self.images_array[start: stop] = np.stack([self.parse_img(img) for img in self.images[start:stop]], axis=0)
            self.output_array[start:stop] = [
                self.categorize_img(img) for img in self.images[start:stop]
            ]

    @staticmethod
    def parse_img(img: Path) -> NDArray[np.uint8]:
        image = Image.open(str(img))
        image = image.convert("RGB").resize((300, 200))

        return np.ravel(np.asarray(image)).astype(np.uint8)

    @staticmethod
    def categorize_img(img: Path) -> int:
        match img.parent.name:
            case "cane":
                return 0
            case "cavallo":
                return 1
            case "elefante":
                return 2
            case "farfalla":
                return 3
            case "gallina":
                return 4
            case "gatto":
                return 5
            case "mucca":
                return 6
            case "pecora":
                return 7
            case "ragno":
                return 8
            case "scoiattolo":
                return 9
            case _:
                raise ValueError(f"Unclassifiable image in dataset: {img}")

def main():
    loader = Loader()

main()