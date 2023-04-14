from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn import datasets
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv('./fonts/ARIAL.csv')
print(data)

font_list = ['GEORGIA', 'PALATINO', 'FRANKLIN', 'STYLUS', 'NINA', 'GOUDY', 'BRITANNIC', 'CURLZ', 'ROMANTIC', 'CONSTANTIA', 'SIMPLEX', 'BAUHAUS', 'GUNPLAY', 'GABRIOLA', 'ERAS', 'CONSOLAS', 'RAGE', 'ENGLISH', 'NUMERICS', 'BUXTON', 'NIAGARA', 'MONOTYPE', 'YI BAITI', 'LEELAWADEE', 'JUICE', 'SITKA', 'BAITI', 'CORBEL', 'BASKERVILLE', 'MINGLIU', 'BLACKADDER', 'EDWARDIAN', 'VLADIMIR', 'FORTE', 'GOTHICE', 'MONOSPAC821', 'TREBUCHET', 'SNAP', 'SCRIPT', 'PHAGSPA', 'SKETCHFLOW', 'CALIFORNIAN', 'SYLFAEN', 'SEGOE', 'GADUGI', 'BANKGOTHIC', 'STENCIL', 'COUNTRYBLUEPRINT', 'HANDPRINT', 'COMIC', 'TIMES', 'CALISTO', 'PLAYBILL', 'BERNARD', 'JOKERMAN', 'MODERN', 'TAI', 'CAARD', 'CENTURY', 'CENTAUR', 'PANROMAN', 'RICHARD', 'HAETTENSCHWEILER', 'CASTELLAR', 'ITALIC', 'EBRIMA', 'MONEY', 'MYANMAR', 'ENGRAVERS', 'PROXY', 'PALACE', 'QUICKTYPE', 'TECHNIC', 'VINER', 'RAVIE', 'CREDITCARD',
             'SUPERFRENCH', 'PRISTINA', 'GIGI', 'CANDARA', 'IMPACT', 'CHILLER', 'VIVALDI', 'ARIAL', 'CAMBRIA', 'COMPLEX', 'MAGNETO', 'COOPER', 'KUNSTLER', 'BERLIN', 'ROCKWELL', 'SWIS721', 'FELIX TITLING', 'TXT', 'GLOUCESTER', 'NIRMALA', 'IMPRINT', 'HIGH TOWER', 'AGENCY', 'BRADLEY', 'PERPETUA', 'EUROROMAN', 'ISOC', 'SHOWCARD', 'PAPYRUS', 'REFERENCE', 'COMMERCIALSCRIPT', 'SCRIPTB', 'BELL', 'CITYBLUEPRINT', 'TAHOMA', 'PMINGLIU-EXTB', 'COURIER', 'BOOK', 'HIMALAYA', 'KRISTEN', 'GILL', 'BROADWAY', 'ONYX', 'VINETA', 'BOOKMAN', 'MISTRAL', 'TEMPUS', 'HARRINGTON', 'VIN', 'GARAMOND', 'JAVANESE', 'WIDE', 'BITSTREAMVERA', 'OCRB', 'INFORMAL', 'ROMAN', 'MATURA', 'BRUSH', 'BODONI', 'MONOTXT', 'SANSSERIF', 'ELEPHANT', 'SERIF', 'MAIANDRA', 'CALIBRI', 'OCRA', 'FREESTYLE', 'FRENCH', 'TW', 'FOOTLIGHT', 'E13B', 'LUCIDA', 'MV_BOLI', 'COPPERPLATE', 'DUTCH801', 'VERDANA', 'HARLOW']

print(font_list)

# def most_common(lst):
#     return max(set(lst), key=lst.count)


# def euclidean(point, data):
#     return np.sqrt(np.sum((point - data)**2, axis=1))


# class KNeighborsClassifier:
#     def __init__(self, k=21, dist_metric=euclidean):
#         self.k = k
#         self.dist_metric = dist_metric

#     def fit(self, X_train, y_train):
#         self.X_train = X_train
#         self.y_train = y_train

#     def predict(self, X_test):
#         neighbors = []
#         for x in X_test:
#             distances = self.dist_metric(x, self.X_train)
#             y_sorted = [y for _, y in sorted(zip(distances, self.y_train))]
#             neighbors.append(y_sorted[:self.k])
#         return list(map(most_common, neighbors))

#     def evaluate(self, X_test, y_test):
#         y_pred = self.predict(X_test)
#         accuracy = sum(y_pred == y_test) / len(y_test)
#         return accuracy


# iris = datasets.load_iris()
# X = iris['data']
# y = iris['target']


# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)


# ss = StandardScaler().fit(X_train)
# X_train, X_test = ss.transform(X_train), ss.transform(X_test)


# accuracies = []
# ks = range(1, 30)
# for k in ks:
#     knn = KNeighborsClassifier(k=k)
#     knn.fit(X_train, y_train)
#     accuracy = knn.evaluate(X_test, y_test)
#     accuracies.append(accuracy)


# fig, ax = plt.subplots()
# ax.plot(ks, accuracies)
# ax.set(xlabel="k",
#        ylabel="Accuracy",
#        title="Performance of knn")
# plt.show()