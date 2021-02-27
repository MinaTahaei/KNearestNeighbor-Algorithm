import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score

url = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"

columnNames = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'Class']

dataset = pd.read_csv(url, names=columnNames)

X = dataset.iloc[:, :-1].values  # first four columns
y = dataset.iloc[:, 4].values  # Labels

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20)  # Split the data

# Scale the Data
scaler = StandardScaler()
scaler.fit(X_train)

X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

classifier = KNeighborsClassifier(n_neighbors=5)
classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)
print(classification_report(y_test, y_pred))

# accuracy: (tp + tn) / (p + n)
accuracy = accuracy_score(y_test, y_pred)
print('Accuracy: %f' % accuracy)
# precision tp / (tp + fp)
# precision = precision_score(y_test, y_pred)
# print('Precision: %f' % precision)
# recall: tp / (tp + fn)
# recall = recall_score(y_test, y_pred)
# print('Recall: %f' % recall)
# f1: 2 tp / (2 tp + fp + fn)
# f1 = f1_score(y_test, y_pred)
# print('F1 score: %f' % f1)
