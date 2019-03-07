from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.utils.multiclass import check_classification_targets
from sklearn.utils.multiclass import unique_labels
from sklearn.metrics.pairwise import manhattan_distances
import numpy as np


class DumbGuessClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, favourite_number=3):
        self.favourite_number = favourite_number

    def fit(self, X, y):
        # Check that X and y have correct shape
        X, y = check_X_y(X, y)
        check_classification_targets(y)

        # Store the classes seen during fit
        self.classes_ = unique_labels(y)

        self.X_ = X
        self.y_ = y

        # Return the classifier
        return self

    def predict(self, X):
        # Check is fit had been called
        check_is_fitted(self, ["X_", "y_"])

        # Input validation
        X = check_array(X)

        closest = np.argmin(manhattan_distances(X, self.X_), axis=1)
        return self.y_[closest]
