from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn import datasets
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import csv

most_popular_fonts = ['HELVETICA', 'GARAMOND',
                      'BODONI', 'TIMES', 'VERDANA', 'ROCKWELL', 'FRANKLIN']

# DATASET IMPORT AND TRANSFORMATION

# Read dataset from csv file


def convert_data(location):
    results = []
    with open(location) as csvfile:
        print("Reading Data from CSV...")
        print("...")
        print("...")
        print("...")
        reader = csv.reader(csvfile)
        print("Converting data...")
        print("...")
        print("...")
        print("...")
        for row in reader:
            results.append(row)
        print("Data Converted.")
        print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
    return results


# extract X from dataset and format into numpy array
def extract_X(array):
    res = []
    print("Extracting X...")
    print("...")
    print("...")
    print("...")
    for row in range(1, len(array)):
        new_row = []
        for col in array[row][3:]:
            new_row.append(float(col))
        res.append(new_row)
    formatted = np.array(res)
    print("X extracted.")
    print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
    return formatted

# extract m_labels from dataset and format into numpy array


def extract_y(array):
    targets = []
    print("Extracting y...")
    print("...")
    print("...")
    print("...")
    for x in range(1, len(array)):
        label = array[x][2]
        targets.append(float(label))
    formatted = np.array(targets)
    formatted = targets
    print("Y extracted.")
    print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
    return formatted


# KNN ALGORITHM AND METHODS
def most_common(lst):
    return max(set(lst), key=lst.count)


def euclidean(point, data):
    return np.sqrt(np.sum((point - data)**2, axis=1))


class KNeighborsClassifier:
    def __init__(self, k=21, dist_metric=euclidean):
        self.k = k
        self.dist_metric = dist_metric

    def fit(self, X_train, y_train):
        self.X_train = X_train
        self.y_train = y_train

    def predict(self, X_test):
        neighbors = []
        for x in X_test:
            distances = self.dist_metric(x, self.X_train)
            y_sorted = [y for _, y in sorted(zip(distances, self.y_train))]
            neighbors.append(y_sorted[:self.k])
        return list(map(most_common, neighbors))

    def evaluate(self, X_test, y_test):
        y_pred = self.predict(X_test)
        # Both y_pred and y_test must be numpy arrays to make use of the boolean array to test equality between them
        y_pred = np.array(y_pred)
        y_test = np.array(y_test)
        accuracy = sum(y_pred == y_test) / len(y_test)
        return accuracy


# Get dataset
dataset_array = convert_data("../ARIAL.csv")
# dataset_array = convert_data("../fonts.csv")

# iris = datasets.load_iris()
# X = iris['data']
# y = iris['target']


# TOY DATASETS for testing
# iris = datasets.load_iris()
digits = datasets.load_digits()
# print(digits.data.shape)
# X = iris['data']
# y = iris['target']
X = digits['data']
y = digits['target']


# Extract X and y arrays
# X = extract_X(dataset_array)
# y = extract_y(dataset_array)

# Split into train and test data
print("Splitting train and test data...")
print("...")
print("...")
print("...")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.01)
print("test and train data split.")
print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")

# Scale the train and test data using SrandardScaler
ss = StandardScaler().fit(X_train)
X_train, X_test = ss.transform(X_train), ss.transform(X_test)


# RUN KNN ALOGRITHM WITH DATASET AND PLOT RESULTS

accuracies = []
ks = range(1, 11)
for k in ks:
    knn = KNeighborsClassifier(k=k)
    knn.fit(X_train, y_train)
    accuracy = knn.evaluate(X_test, y_test)
    accuracies.append(accuracy)


print("Accuracies")
for ind, a in enumerate(accuracies):
    print(f"Case {ind + 1}: ", f"{round(a * 100, 2)}%")

# fig, ax = plt.subplots()
# ax.plot(ks, accuracies)
# ax.set(xlabel="k",
#        ylabel="Accuracy",
#        title="Performance of knn")
# plt.show()
