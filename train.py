# import required libraries

import struct
import numpy as np
import os
import pickle
from model import SimpleNN
import matplotlib.pyplot as plt

# loading the data
def load_mnist_images(filename):
    with open(filename, 'rb') as f:
        magic, num_images = struct.unpack(">II", f.read(8))
        rows, cols = struct.unpack(">II", f.read(8))
        images = np.frombuffer(f.read(), dtype=np.uint8).reshape(num_images, rows * cols)
        return images.astype(np.float32) / 255.0

# loading labels
def load_mnist_labels(filename):
    with open(filename, 'rb') as f:
        magic, num_labels = struct.unpack(">II", f.read(8))
        labels = np.frombuffer(f.read(), dtype=np.uint8)
        return labels

# main dataset path
main_path = "./MNIST/raw"

train_images = load_mnist_images(os.path.join(main_path, "train-images-idx3-ubyte"))
train_labels = load_mnist_labels(os.path.join(main_path, "train-labels-idx1-ubyte"))
test_images  = load_mnist_images(os.path.join(main_path, "t10k-images-idx3-ubyte"))
test_labels  = load_mnist_labels(os.path.join(main_path, "t10k-labels-idx1-ubyte"))

# model training with parameters
model = SimpleNN(784, 128, 10)
epochs = 10
batch_size = 64

# calculating accuracy
def accuracy(model, X, y):
    return np.mean(model.forward(X).argmax(axis=1) == y)

for epoch in range(epochs):
    perm = np.random.permutation(len(train_images))
    X, y = train_images[perm], train_labels[perm]

    for i in range(0, len(X), batch_size):
        xb = X[i:i+batch_size]
        yb = y[i:i+batch_size]

        model.forward(xb)
        model.backward(xb, yb, lr=0.1)

    acc = accuracy(model, test_images, test_labels)
    print(f"epoch {epoch+1}/{epochs} with accuracy: {acc*100:.3f}%")

# saving model after training
with open("model.pkl", "wb") as f:
    pickle.dump(model, f)

print("model is successfully saved as model.pkl!")

# final model accuracy
final_acc = accuracy(model, test_images, test_labels)
print(f"test accuracy: {final_acc * 100:.3f}%")

# visualizations
def show_predictions(model, X, y, num=5):
    preds = model.forward(X[:num]).argmax(axis=1)

    plt.figure(figsize=(10, 3))
    for i in range(num):
        plt.subplot(1, num, i+1)
        plt.imshow(X[i].reshape(28, 28), cmap='gray')
        plt.title(f"T:{y[i]} P:{preds[i]}")
        plt.axis('off')
    plt.show()

# call it
show_predictions(model, test_images, test_labels)