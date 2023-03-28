import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
df = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data', header=None)


class Perceptron:

    def __init__(self, test_set_values, train_set_values, test_set_classes, train_set_classes,
                 n_epochs: int, learning_rate: float):
        self.n_epochs = n_epochs
        self.test_set_values = test_set_values
        self.train_set_values = train_set_values
        self.learning_rate = learning_rate
        self.test_set_classes = test_set_classes
        self.train_set_classes = train_set_classes
        self.bias = 0.0
        self.weights = None
        self.err = []

    def train(self):
        """

        """
        self.bias = 0.0
        self.weights = np.zeros(self.train_set_values.shape[1])
        self.err = []

        for i in range(self.n_epochs):
            error = 0
            for x, y in zip(self.train_set_values, self.train_set_classes):
                update = self.learning_rate * (y - self.thresholding_function(x))
                self.bias += update
                self.weights += update * x
                error += int(update != 0.0)
            self.err.append(error)

    def activation_function(self, x: np.array) -> float:
        """
        compute the output of the neuron
        :param x: input features
        :return: the output of the neuron
        """
        return np.dot(x, self.weights) + self.bias

    def thresholding_function(self, x: np.array):
        """
        convert the output of the neuron to a binary output
        :param x: input features
        :return: 1 if the output for the sample is positive (or zero),
        -1 otherwise
        """
        return np.where(self.activation_function(x) >= 0, 1, -1)
    @staticmethod
    def accuracy(predicted, actual):
        accuracy = 0
        for i in range(predicted.shape[0]):
            if predicted[i] == actual[i][0]:
                accuracy += 1

        return accuracy/float(len(actual)) * 100.0

# Data preprocessing
df.columns = ['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm', 'Species']
X_data = df.iloc[:150, [0, 1, 2, 3]].values
y_data = df.iloc[:150, [4]].values
y_data = np.where(y_data == 'Iris-versicolor', 1, -1)

X_train, X_test, y_train, y_test = train_test_split(X_data, y_data, test_size=0.25, random_state=0)

perceptron = Perceptron(X_test, X_train, y_test, y_train, n_epochs=100, learning_rate=0.01)
perceptron.train()
predictions = perceptron.thresholding_function(X_test)
print("accuracy ", perceptron.accuracy(predictions, y_test), "%")

plt.plot(range(1, len(perceptron.err) + 1), perceptron.err, marker='o')
plt.xlabel('Epoch')
plt.ylabel('Errors')
plt.show()

