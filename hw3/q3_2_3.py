import q3_0
import numpy as np
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier


def fit_dec_tree(train_data, train_labels, test_data, test_labels, max_d):
    # decision tree classifier
    dec_tree_clf = DecisionTreeClassifier(max_depth=max_d)
    # fit the classifier to data
    dec_tree_clf.fit(train_data, train_labels)
    # print the scores. For max depth of 10, the raw score should be around 84.7%.
    print("Decision Tree Score: ", dec_tree_clf.score(test_data, test_labels))


def fit_adaboosted_dec_tree(train_data, train_labels, test_data, test_labels, max_d, num_est):
    # declare a base estimator
    base_clf = DecisionTreeClassifier(max_depth=max_d)
    # declare AdaBoost Classifier
    clf = AdaBoostClassifier(base_clf, n_estimators=num_est)
    # fit adaboosted classifier
    clf.fit(train_data, train_labels)
    # print the scores for this classifier
    print("Adaboosted Decision Tree Score: ", clf.score(test_data, test_labels))
    
    # an example of how to access and predict individual samples (if you need it)    
    # reshaped_sample = np.array(test_data[2]).reshape(1, len(test_data[0]))
    # print(clf.predict(reshaped_sample))
    # print(test_labels[2])


if __name__ == "__main__":
    # preprocess and import data:
    train_data, train_labels, test_data, test_labels = q3_0.data.load_all_data_from_zip("a3digits.zip", "data")
    # I chose 8 for depth of the tree because at that depth we get Adaboost reaching 96% accuracy while decision tree
    # gets 82%.
    fit_dec_tree(train_data, train_labels, test_data, test_labels, 8)
    # I am using 50 estimators for Adaboost
    fit_adaboosted_dec_tree(train_data, train_labels, test_data, test_labels, 8, 50)

