'''
Question 3.1 Skeleton Code

Here you should implement and evaluate the k-NN classifier.
'''
from sklearn.model_selection import KFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import label_binarize
from tqdm import tqdm

import data
import numpy as np
from collections import Counter, defaultdict
# Import pyplot - plt.imshow is useful!
import matplotlib.pyplot as plt

from q3_3 import summarize_and_save_model_report, visualize_roc_curve


class KNearestNeighbor(object):
    '''
    K Nearest Neighbor classifier
    '''

    def __init__(self, train_data, train_labels):
        self.train_data = train_data
        self.train_norm = (self.train_data**2).sum(axis=1).reshape(-1,1)
        self.train_labels = train_labels

    def l2_distance(self, test_point):
        '''
        Compute L2 distance between test point and each training point
        
        Input: test_point is a 1d numpy array
        Output: dist is a numpy array containing the distances between the test point and each training point
        '''
        # Process test point shape
        test_point = np.squeeze(test_point)
        if test_point.ndim == 1:
            test_point = test_point.reshape(1, -1)
        assert test_point.shape[1] == self.train_data.shape[1]

        # Compute squared distance
        test_norm = (test_point**2).sum(axis=1).reshape(1,-1)
        dist = self.train_norm + test_norm - 2*self.train_data.dot(test_point.transpose())
        return np.squeeze(dist)

    def query_knn(self, test_point, k):
        '''
        Query a single test point using the k-NN algorithm

        You should return the digit label provided by the algorithm
        '''
        # calculate distances from each train sample
        distances = self.l2_distance(test_point)
        # build tuple of distances and labels
        dist_label_tuples = list(zip(distances, self.train_labels))
        # sort the tuples based on lowest to highest distances
        dist_label_tuples = sorted(dist_label_tuples, key=lambda x: x[0])
        # take top-k samples
        closest_k_points = dist_label_tuples[:k]
        # categorize then in a dictionary based on  their class and distance
        class_distance_dict = defaultdict(list)
        for dist, label in closest_k_points:
            class_distance_dict[label].append(dist)
        # find classes with maximum frequency
        max_freq = max([len(class_distance_dict[_class]) for _class in class_distance_dict])
        max_freq_classes = [_class for _class in class_distance_dict if len(class_distance_dict[_class]) == max_freq]
        # tie-breaking
        if len(max_freq_classes) > 1:
            # find minimum distance from maximum frequency classes
            max_freq_min_dist = [(_class, min(class_distance_dict[_class])) for _class in max_freq_classes]
            # sort by minimum distances and pick first class
            digit = sorted(max_freq_min_dist, key=lambda x: x[1])[0][0]
        else:
            digit = max_freq_classes[0]
        return digit


def cross_validation(train_data, train_labels, k_range=np.arange(1,16)):
    '''
    Perform 10-fold cross validation to find the best value for k

    Note: Previously this function took knn as an argument instead of train_data,train_labels.
    The intention was for students to take the training data from the knn object - this should be clearer
    from the new function signature.
    '''
    # perform cross-validation over various k values and return best
    mean_accs_k = []
    for k in k_range:
        folds = KFold(n_splits=10, shuffle=True)
        fold_splits = list(folds.split(train_data, train_labels))
        fold_accs = []
        # loop over folds
        for train_inds, val_inds in fold_splits:
            fold_train_data, fold_train_labels = train_data[train_inds], train_labels[train_inds]
            fold_val_data, fold_val_labels = train_data[val_inds], train_labels[val_inds]
            # fit KNN on train fold
            fold_knn = KNearestNeighbor(fold_train_data, fold_train_labels)
            # evaluate on val fold
            val_acc = classification_accuracy(fold_knn, k, fold_val_data, fold_val_labels)
            fold_accs.append(val_acc)
        # append mean accuracy
        mean_acc = sum(fold_accs) / len(fold_accs)
        mean_accs_k.append((mean_acc, k))
    return mean_accs_k

def classification_accuracy(knn, k, eval_data, eval_labels, return_preds=False):
    '''
    Evaluate the classification accuracy of knn on the given 'eval_data'
    using the labels
    '''
    preds = [knn.query_knn(point, k) for point in tqdm(eval_data)]
    accs = [1.0 if pred == label else 0 for pred, label in zip(preds, eval_labels)]
    acc = sum(accs) / len(accs)
    if return_preds:
        return acc, preds
    return acc


def main():
    """
    Part-1:
        K=1 KNN Accuracy, Train: 1.0; Test:0.96875
        K=15 KNN Accuracy, Train: 0.9637142857142857; Test:0.961

    Part-2:
        Blah.

    Part-3:


    """
    train_data, train_labels, test_data, test_labels = data.load_all_data('data')
    knn = KNearestNeighbor(train_data, train_labels)

    # part 1
    k1_train_acc = classification_accuracy(knn, 1, train_data, train_labels)
    k1_test_acc = classification_accuracy(knn, 1, test_data, test_labels)
    k15_train_acc = classification_accuracy(knn, 15, train_data, train_labels)
    k15_test_acc = classification_accuracy(knn, 15, test_data, test_labels)
    print(f"K=1 KNN Accuracy, Train: {k1_train_acc}; Test:{k1_test_acc}")
    print(f"K=15 KNN Accuracy, Train: {k15_train_acc}; Test:{k15_test_acc}")

    # part 2 (see in query_knn function) -- we use the class of closest sample to break ties

    # part 3
    mean_accs_k = cross_validation(train_data, train_labels)
    for acc, k in mean_accs_k:
        print(f"10-Fold with K={k} Mean Fold Accuracy={acc}")

    best_k = sorted(mean_accs_k, key=lambda x:x[0], reverse=True)[0][1]
    best_k_train_acc = classification_accuracy(knn, best_k, train_data, train_labels)
    best_k_test_acc, test_preds = classification_accuracy(knn, best_k, test_data, test_labels, return_preds=True)
    print(f"With best K={best_k}, Train Accuracy: {best_k_train_acc}, Test Accuracy: {best_k_test_acc}")

    # save model summary for comparison
    summarize_and_save_model_report(np.array(test_preds), np.array(test_labels), "knn")

    # using sklearn-knn ONLY for computing ROC curve!
    clf = KNeighborsClassifier(n_neighbors=best_k).fit(train_data, train_labels)
    probs = clf.predict_proba(test_data)
    binarized_labels = label_binarize(test_labels, classes=range(10))
    visualize_roc_curve(probs, binarized_labels, name="knn")


if __name__ == '__main__':
    main()
