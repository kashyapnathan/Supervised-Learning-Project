import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
import time

# Load the dataset
data = load_breast_cancer()
X = data.data
y = data.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

training_times = {}

# Decision Trees with GINI index and pruning
dt_clf = DecisionTreeClassifier(criterion="gini", max_depth=5)
start_time = time.time()
dt_clf.fit(X_train, y_train)
training_times["Decision Trees"] = time.time() - start_time
dt_score = dt_clf.score(X_test, y_test)
print(f"Decision Tree Accuracy: {dt_score:.2f}")

# Neural Network with varied architectures
mlp_clf = MLPClassifier(hidden_layer_sizes=(100, 50), activation="relu", max_iter=1000)
start_time = time.time()
mlp_clf.fit(X_train, y_train)
training_times["Neural Networks"] = time.time() - start_time
mlp_score = mlp_clf.score(X_test, y_test)
print(f"Neural Network Accuracy: {mlp_score:.2f}")

# Boosting with aggressive pruning
ada_clf = AdaBoostClassifier(base_estimator=DecisionTreeClassifier(max_depth=3), n_estimators=50)
start_time = time.time()
ada_clf.fit(X_train, y_train)
training_times["Boosting"] = time.time() - start_time
ada_score = ada_clf.score(X_test, y_test)
print(f"Boosting Accuracy: {ada_score:.2f}")

# Support Vector Machines with polynomial kernel
svc_poly_clf = SVC(kernel="poly", degree=3, probability=True)
start_time = time.time()
svc_poly_clf.fit(X_train, y_train)
training_times["SVM (Poly Kernel)"] = time.time() - start_time
svc_poly_score = svc_poly_clf.score(X_test, y_test)
print(f"SVM (Poly Kernel) Accuracy: {svc_poly_score:.2f}")

# Support Vector Machines with linear kernel
svc_lin_clf = SVC(kernel='linear', probability=True, random_state=42, C=0.1)
start_time = time.time()
svc_lin_clf.fit(X_train, y_train)
training_times["SVM (Linear Kernel)"] = time.time() - start_time
svc_lin_score = svc_lin_clf.score(X_test, y_test)
print(f"SVM (Linear Kernel) Accuracy: {svc_lin_score:.2f}")

# k-Nearest Neighbors with varied k
knn_clf = KNeighborsClassifier(n_neighbors=5)
start_time = time.time()
knn_clf.fit(X_train, y_train)
training_times["k-Nearest Neighbors"] = time.time() - start_time
knn_score = knn_clf.score(X_test, y_test)
print(f"kNN (k=5) Accuracy: {knn_score:.2f}")

print(training_times)
