from sklearn.utils.estimator_checks import check_estimator
from method import DumbGuessClassifier

def test_exposer():
    return check_estimator(DumbGuessClassifier)
