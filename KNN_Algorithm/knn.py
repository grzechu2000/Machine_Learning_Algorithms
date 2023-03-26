from csv import reader
from math import sqrt


abs_path_iris = "C:\Users\Grzesiek\Documents\GitHub\Machine_Learning_Algorithms\KNN_Algorithm\iris_test.csv"
abs_path_test = "C:\Users\Grzesiek\Documents\GitHub\Machine_Learning_Algorithms\KNN_Algorithm\iris.csv"

class KNearestNeighbors:
    """ A class for the k nearest neighbors algorithm """

    def __init__(self, test_set_csv: str = abs_path_test, dataset_csv: str = abs_path_iris, n_neighbors: int = 5):
        self.n_neighbors = n_neighbors
        self.test_set = self.data_conversion(self.load_csv(test_set_csv))
        self.dataset = self.data_conversion(self.load_csv(dataset_csv))

    @staticmethod
    def load_csv(csv_file: str) -> list:
        """ This method takes in a csv file with data, and converts it into a python list.


        :param csv_file: name of the csv file with data
        :return: dataset: formatted list of type list[list[str]]
        """
        dataset = list()
        with open(csv_file, 'r') as file:
            csv_reader = reader(file)
            for row in csv_reader:
                if not row:
                    continue
                dataset.append(row)
            return dataset

    @staticmethod
    def data_conversion(dataset: list) -> list:
        """ This method takes a formatted dataset from the load_csv method, and converts data of type
        string to data of type float and/or int.


        :param dataset: a formatted list
        :return: a formatted list
        """
        for row in dataset:
            for i in range(len(row)-1):
                row[i] = float(row[i])
            if row[-1] == 'Iris-setosa':
                row[-1] = 0
            elif row[-1] == 'Iris-versicolor':
                row[-1] = 1
            elif row[-1] == 'Iris-virginica':
                row[-1] = 2
            else:
                row[-1] = float(row[-1])
        return dataset

    @staticmethod
    def calculate_distance(test_row: list, train_row: list) -> float:
        """ This method calculates the euclidean distance between data points

        :param test_row: test_data point of type list
        :param train_row: train_data point of type list
        :return: euclidean distance between data points in the n-th dimension
        """
        euclidean_distance = 0.0
        for i in range(len(train_row)-1):
            euclidean_distance += (test_row[i] - train_row[i])**2
        return sqrt(euclidean_distance)

    def get_nearest_neighbor(self, test_row: list) -> list:
        """ This method calculates the nearest neighboring points from train_data to
        the test_data point

        :param test_row: test_data point of type list
        :return: list of n nearest neighbors to the test_data point
        """
        nearest_neighbor = []
        for train_row in self.dataset:
            distance = self.calculate_distance(test_row, train_row)
            nearest_neighbor.append((train_row, distance))
            nearest_neighbor.sort(key=lambda neighbor: neighbor[1])
        neighbor_list = []
        for i in range(self.n_neighbors):
            neighbor_list.append(nearest_neighbor[i][0])
        return neighbor_list

    def predict_classification(self, test_row: list) -> int:
        """ This method predicts the class of the submitted test_data point according
        to it's nearest neighbors

        :param test_row: test_data point of type list
        :return: prediction in int format in relation to different data classifications
        EXAMPLE:
        - "Iris-Setosa" -> prediction = 0
        - "Iris-Versicolor" -> prediction = 1
        - "Iris-Virginica" -> prediction = 2
        """
        neighbors = self.get_nearest_neighbor(test_row)
        species = [data_row[-1] for data_row in neighbors]
        prediction = max(set(species), key=species.count)
        return prediction

    def k_nearest_neighbors(self):
        """ This method classifies test_data points according to the training_data

        :return: list of predicted classifications for submitted test_data points
        """
        test_result = []
        for test_row in self.test_set:
            test_result.append(self.predict_classification(test_row))
        for i in range(len(test_result)):
            if test_result[i] == 0:
                test_result[i] = "Iris-setosa"
            elif test_result[i] == 1:
                test_result[i] = "Iris-versicolor"
            elif test_result[i] == 2:
                test_result[i] = "Iris-virginica"
        return test_result


neural_network = KNearestNeighbors()
print(neural_network.k_nearest_neighbors())


# class Foo:
#
#     def __init__(self, obj):
#         self.obj = obj
#
#
# my_tup = ("UserBase", "CustBase")
#
#
# def return_userbase(my_tuple: tuple)->str:
#     with Foo(my_tup[0]):
#         yield my_tup[0]
#
#
# def return_custbase(my_tuple: tuple)->str:
#     with Foo(my_tup[1]):
#         yield my_tup[0]
#
#
# print(return_custbase(my_tup))
# print(return_userbase(my_tup))


