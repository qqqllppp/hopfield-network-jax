import gzip
import array
import struct
import jax.numpy as jnp
from os import path


def get_mnist_data(path_):
    def parse_labels(filename):
        with gzip.open(filename, "rb") as fh:
            _ = struct.unpack(">II", fh.read(8))
            return jnp.array(array.array("B", fh.read()), dtype=jnp.uint8)

    def parse_images(filename):
        with gzip.open(filename, "rb") as fh:
            _, num_data, rows, cols = struct.unpack(">IIII", fh.read(16))
            return jnp.array(array.array("B", fh.read()), dtype=jnp.uint8).reshape(
                num_data, rows, cols
            )

    for filename in [
        "train-images-idx3-ubyte.gz",
        "train-labels-idx1-ubyte.gz",
        "t10k-images-idx3-ubyte.gz",
        "t10k-labels-idx1-ubyte.gz",
    ]:

        train_images = parse_images(path.join(path_, "train-images-idx3-ubyte.gz"))
        train_labels = parse_labels(path.join(path_, "train-labels-idx1-ubyte.gz"))
        test_images = parse_images(path.join(path_, "t10k-images-idx3-ubyte.gz"))
        test_labels = parse_labels(path.join(path_, "t10k-labels-idx1-ubyte.gz"))

    return (
        train_images,
        train_labels,
        test_images,
        test_labels,
    )


def reshape(data):
    dim = int(jnp.sqrt(len(data)))
    data = jnp.reshape(data, (dim, dim))
    return data
