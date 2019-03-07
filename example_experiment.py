"""
Prosty eksperyment klasyfikacji.
"""
from sklearn import datasets
from sklearn import preprocessing
from sklearn import model_selection
from sklearn import neighbors, naive_bayes, tree, svm, neural_network
from sklearn import base
from sklearn import metrics
import method
import numpy as np

# Zbiory danych
datasets = [datasets.load_wine(), datasets.load_iris()]

# Metody klasyfikacji
clfs = {
    "kNN": neighbors.KNeighborsClassifier(),
    "GNB": naive_bayes.GaussianNB(),
    "DTC": tree.DecisionTreeClassifier(),
    "SVC": svm.SVC(gamma='scale'),
    "MLP": neural_network.MLPClassifier(),
    "AEC": method.DumbGuessClassifier(),
}

k=5
# Tabela wyników
scores = np.zeros((len(datasets), len(clfs), k))

# Iteracja zbiorów
for did, dataset in enumerate(datasets):
    X = dataset.data
    y = dataset.target

    print(X.shape)

    # Normalizacja standardowa
    scaler = preprocessing.StandardScaler()
    scaler.fit(X)
    scaled_X = scaler.transform(X)

    # Walidacja krzyżowa
    skf = model_selection.StratifiedKFold(n_splits=k)
    for fold, (train, test) in enumerate(skf.split(scaled_X, y)):
        X_train, X_test = scaled_X[train], scaled_X[test]
        y_train, y_test = y[train], y[test]

        # Iteracja klasyfikatorów
        for cid, clfn in enumerate(clfs):
            clf = base.clone(clfs[clfn])

            # Dopasowanie modelu
            clf.fit(X_train, y_train)

            # Predykcja
            y_pred = clf.predict(X_test)

            # Obliczenie jakości
            score = metrics.accuracy_score(y_test, y_pred)

            scores[did, cid, fold] = score

# Prezentacja wyników
print(np.mean(scores, axis=2))
