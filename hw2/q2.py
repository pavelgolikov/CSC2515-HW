from sklearn import datasets
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np


def load_data():
    boston = datasets.load_boston()
    X = boston.data
    y = boston.target
    features = boston.feature_names
    return X, y, features


def visualize(X, y, features):
    """
    Visualize the features of the dataset against the solutions.
    :param X: Data set Matrix.
    :param y: Solutions.
    :param features: Names of the features of the Data Set.
    :return: None
    """
    plt.figure(figsize=(20, 5))
    feature_count = X.shape[1]
    
    # i: index
    for i in range(feature_count):
        plt.subplot(3, 5, i + 1)
        #TODO: Plot feature i against y
        plt.scatter(X[:, i - 1], y)
        plt.ylabel("Dollars US (1K)")
        plt.xlabel(features[i])
        plt.savefig("Features.png")
        # plt.title(features[i])
    
    plt.tight_layout()
    plt.show()


def fit_regression(X, Y):
    """
    Fits dataset X to targets Y using direct solution of linear regression.
    :param X: Training data-set.
    :param Y: Training targets.
    :return: Vector of trained weights.
    """
    #TODO: implement linear regression
    # Remember to use np.linalg.solve instead of inverting!
    
    # Adding bias to X:
    X = np.c_[np.ones(X.shape[0]), X]
    
    # compute the square matrix required for linalg.solve:
    transpose = np.transpose(X)
    m1 = np.matmul(transpose, X)
    m2 = np.matmul(transpose, Y)
    
    # solve the resulting linear system with linalg.solve
    return np.linalg.solve(m1, m2)


def tabulate_weights(w, features):
    """
    Produces a table with computed weights and their names
    :param w: vector of weights
    :param features: feature names
    :return: None
    """
    # omitting bias term in w
    weights_minus_bias = [["{:.2f}".format(x) for x in w[1:]]]
    features = features.tolist()
    fig, ax = plt.subplots()
    ax.set_axis_off()
    table = ax.table(
        weights_minus_bias,
        rowLabels=["Weights"],
        colLabels=['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT'],
        cellLoc='center',
        loc='center'
        )
    table.auto_set_font_size(False)
    table.set_fontsize(7)
    plt.tight_layout()
    # plt.show()
    
def compute_MAE(X_test, y_test, w):
    """
    Computes Mean Absolute Error of the given test set and targets with given weights.
    :param X_test: Test data set.
    :param y_test: Test targets.
    :param w: Vector of weights
    :return: Mean Absolute Error of the predictions.
    """
    # adding bias column to X_test:
    X_test = np.c_[np.ones(X_test.shape[0]), X_test]
    # calculate result
    result = np.dot(X_test, w)
    # calculate rest of MSE
    abs_residual = np.abs(result - y_test)
    return sum(abs_residual) / (y_test.shape[0])

    
    
def calculate_MSE(X_test, y_test, w):
    """
    Computes Mean Squared Error of the given test set and targets with given weights.
    :param X_test: Test data set.
    :param y_test: Test targets.
    :param w: Vector of weights
    :return: Mean Squared Error of the predictions.
    """
    # adding bias column to X_test:
    X_test = np.c_[np.ones(X_test.shape[0]), X_test]
    # calculate result
    result = np.dot(X_test, w)
    # calculate rest of MSE
    residual = result - y_test
    return sum(residual * residual) / (y_test.shape[0])


def main():
    # a) Load the data
    X, y, features = load_data()
    print("Features: {}".format(features))
    
    visualize(X, y, features)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)
    
    w = fit_regression(X_train, y_train)
    
    tabulate_weights(w, features)
    
    mse = calculate_MSE(X_test, y_test, w)
    print("Mean Squared Error obtained is: ", mse)
    
    squared_mse = np.sqrt(mse);
    print("Root Mean Squared Error obtained is: ", squared_mse)
    
    mae = compute_MAE(X_test, y_test, w)
    print("Mean Absolute Error obtained is: ", mae)
    

if __name__ == "__main__":
    main()

