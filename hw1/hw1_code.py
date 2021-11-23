from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn import tree
import matplotlib.pyplot as plt

import pandas as pd

RANDOM_SEED = 1


def load_data():
    # fake label: 1, real label: 0
    X_fake = read_txt("hw1_data/data/clean_fake.txt")
    y_fake = [1] * len(X_fake)

    X_real = read_txt("hw1_data/data/clean_real.txt")
    y_real = [0] * len(X_real)

    # combine the data
    X = X_fake + X_real
    y = y_fake + y_real

    # extract bag of words features from the input
    vectorizer = CountVectorizer()
    X_features = vectorizer.fit_transform(X)

    # split in train (0.7) and test_val (0.3)
    X_train, X_test_and_val, y_train, y_test_and_val = train_test_split(X_features, y, test_size=0.3,
                                                                        random_state=RANDOM_SEED)

    # split in test (0.15) and val (0.15)
    X_test, X_val, y_test, y_val = train_test_split(X_test_and_val, y_test_and_val, test_size=0.5,
                                                    random_state=RANDOM_SEED)
    return X_train, X_val, X_test, y_train, y_val, y_test


def select_tree_model(X_train, X_val, y_train, y_val):
    # choose 5 *sensible* candidate depth values
    depths = [2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192]

    dt_ents = [DecisionTreeClassifier(criterion='entropy', random_state=0, max_depth=d) for d in depths]
    dt_ginis = [DecisionTreeClassifier(criterion='gini', random_state=0, max_depth=d) for d in depths]
    dts = dt_ents + dt_ginis
    
    # fit all classifiers to train data
    dt_vals = []
    for dt in dts:
        dt.fit(X_train, y_train)
        y_pred = dt.predict(X_val)
        val_acc = sum(y_pred == y_val) / len(y_val)
        dt_vals.append(val_acc)
        print(f"Validation accuracy with depth {dt.max_depth} and criterion {dt.criterion} = {val_acc}")

    # return the best performing model
    best_val_acc = max(dt_vals)
    best_dt_model = dts[dt_vals.index(best_val_acc)]

    print(f"Best DT model has depth {best_dt_model.max_depth} and criterion {best_dt_model.criterion} "
          f"with validation accuracy = {best_val_acc}")
    return best_dt_model


def test_and_plot(best_dt_model, X_test, y_test):
    # test the model
    y_test_pred = best_dt_model.predict(X_test)
    test_acc = sum(y_test_pred == y_test) / len(y_test)
    print(f"Best DT model has test accuracy: {test_acc}")

    # visualize the model
    plt.figure()  # set plot size (denoted in inches)
    tree.plot_tree(best_dt_model, max_depth=2)
    plt.show()


def compute_information_gain(X_train, y_train):
    # hint: https://stackoverflow.com/questions/46752650/information-gain-calculation-with-scikit-learn
    pass


def select_and_test_knn_model(X_train, X_val, X_test, y_train, y_val, y_test):
    ks = range(1, 21)
    knns = [KNeighborsClassifier(n_neighbors=k) for k in ks]

    # fit all classifiers to train data
    knn_val_accs, knn_train_accs = [], []

    for knn in knns:
        knn.fit(X_train, y_train)
        y_pred_val = knn.predict(X_val)
        y_pred_train = knn.predict(X_train)
        val_acc = sum(y_pred_val == y_val) / len(y_val)
        train_acc = sum(y_pred_train == y_train) / len(y_train)
        knn_val_accs.append(val_acc)
        knn_train_accs.append(train_acc)
        print(f"With K = {knn.n_neighbors}, val error: {1 - val_acc} and train error: {1 - train_acc}")

    # return the best performing model
    best_val_acc = max(knn_val_accs)
    train_acc = knn_train_accs[knn_val_accs.index(best_val_acc)]
    best_knn_model = knns[knn_val_accs.index(best_val_acc)]

    print(f"Best KNN model has N = {best_knn_model.n_neighbors} "
          f"with val: {best_val_acc} and train: {train_acc}")

    # test the model
    y_test_pred = best_knn_model.predict(X_test)
    test_acc = sum(y_test_pred == y_test) / len(y_test)
    print(f"Best KNN model has test accuracy: {test_acc}")
    return best_knn_model


def read_txt(filename):
    with open(filename) as file:
        lines = file.readlines()
        lines = [line.rstrip() for line in lines]
    return lines


X_train, X_val, X_test, y_train, y_val, y_test = load_data()
best_dt_model = select_tree_model(X_train, X_val, y_train, y_val)
test_and_plot(best_dt_model, X_test, y_test)
best_knn_model = select_and_test_knn_model(X_train, X_val, X_test, y_train, y_val, y_test)

import pdb
pdb.set_trace()
