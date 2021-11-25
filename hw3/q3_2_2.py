import q3_0
import numpy as np
from sklearn import svm, metrics
from sklearn.model_selection import GridSearchCV


def grid_search_fit_and_test(train_data, train_labels, test_data, test_labels, number_cores):
    # create a matrix with range for gamma parameter and flatten it
    gammas = np.outer(np.logspace(-4, 0, 4), np.array([1, 4])).flatten()
    # create a matrix with range for C parameter and flatten it
    Cs = np.outer(np.logspace(-2, 2, 5), np.array([1, 4])).flatten()
    
    # declare parameters that need to be passed into grid search
    parameters = {'kernel':['rbf'], 'C': Cs, "gamma": gammas}
    
    # declare classifier
    clf = svm.SVC()
    
    # declare a grid search classifier and specify number of cores (n_jobs)
    grid_classifier = GridSearchCV(clf, parameters, n_jobs=number_cores)
    # fit the classifier with grid search
    grid_classifier.fit(train_data, train_labels)
    
    # return best classifier and best parameters
    best_classifier = grid_classifier.best_estimator_
    best_params = grid_classifier.best_params_
    
    # predictions
    pred = best_classifier.predict(test_data)
    
    # print report
    print(
        f"Classification report for classifier with grid search {clf}:\n"
        f"{metrics.classification_report(test_labels, pred)}\n"
    )


def fit_and_test(train_data, train_labels, test_data, test_labels):
    # declare classifier and parameters - this already uses the best parameters found with grid search
    clf = svm.SVC(kernel="rbf", gamma=0.1856635533445111, C=4.0)
    
    # fit classifier
    clf.fit(train_data, train_labels)
    
    # predict values
    pred = clf.predict(test_data)
    
    # print report
    print(
        f"Classification report for classifier {clf}:\n"
        f"{metrics.classification_report(test_labels, pred)}\n"
    )


if __name__ == "__main__":
    train_data, train_labels, test_data, test_labels = q3_0.data.load_all_data_from_zip("a3digits.zip", "data")
    # grid_search_fit_and_test(train_data, train_labels, test_data, test_labels, 1)
    fit_and_test(train_data, train_labels, test_data, test_labels)


# ----------------------------------------------- FOR YASH ----------------------------------------------------
"""
Here are some resources I found that might help you with your part. I briefly saw confusion matrix
in the svm example from sklearn:
https://scikit-learn.org/stable/auto_examples/classification/plot_digits_classification.html
I think "metrics" package has a lot of different classification metrics, some of which match (at least in names) with 
what we are asked to implement/find in q3_3.

I also found this repo that helped:
https://github.com/ksopyla/svm_mnist_digit_classification/blob/master/svm_mnist_grid_search.py
they are using different dataset, but it is similar. The author prints out confusion matrices.

I did grid search and these turned out to be best params (98% accuracy):
{'C': 4.0, 'gamma': 0.1856635533445111, 'kernel': 'rbf'}
"""
