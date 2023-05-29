import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
df = pd.read_csv('iris.data', header=None)


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

import csv
# Data preprocessing
df.columns = ['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm', 'Species']
predict_list = []
species_list = ['Iris-versicolor', 'Iris-setosa', 'Iris-virginica']
x = np.arange(0.05, 0.95 + 0.05, 0.05)
with open('data.csv', 'a+', newline='') as csvfile:
    acc_score_species = []
    for species in species_list:
        print(species)
        acc_score = []
        for size in x:
            print(size)
            for i in range(0, 10):
                X_data = df.iloc[:150, [0, 3]].values
                y_data = df.iloc[:150, [4]].values
                y_data = np.where(y_data == species, 1, -1)
                X_train, X_test, y_train, y_test = train_test_split(X_data, y_data, test_size=size, random_state=0)
                neural_network = Perceptron(X_test, X_train, y_test, y_train, n_epochs=100, learning_rate=0.1)
                neural_network.train()
                predictions = neural_network.thresholding_function(X_test)
                predict_list.append(neural_network.accuracy(predicted=predictions, actual=y_test))
                del(X_data, y_data, X_train, X_test, y_train, y_test, neural_network, predictions)
            acc_sum = 0
            for i in range(0, len(predict_list)):
                acc_sum += predict_list[i]
            mean_acc = float(acc_sum / len(predict_list))
            acc_score.append(mean_acc)
            print("Mean accuracy score: ", mean_acc, "%")
        acc_score_species.append(acc_score)
        del acc_score
    write = csv.writer(csvfile)
    write.writerows(acc_score_species)

for i in range(0, len(acc_score_species)):
    plt.plot(x, acc_score_species[i], label=species_list[i])
plt.title("Rosenblatt Accuracy - data vector len() = 2")
plt.xlabel("Train - Test split")
plt.ylabel("Accuracy score [%]")
plt.grid()
plt.legend(species_list)
plt.show()

