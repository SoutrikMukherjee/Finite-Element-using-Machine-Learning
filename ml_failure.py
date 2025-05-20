import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

def generate_synthetic_data(n_samples=200):
    if not isinstance(n_samples, int) or n_samples <= 0:
        raise ValueError("n_samples must be a positive integer.")
    np.random.seed(42)
    strain = np.random.uniform(0, 0.01, n_samples)
    stress = np.random.uniform(0, 400, n_samples)
    material = np.random.randint(0, 2, n_samples)
    label = ((stress > 250) & (strain > 0.005)).astype(int)
    X = np.vstack([strain, stress, material]).T
    return X, label

def train_failure_predictor(X, y):
    if not isinstance(X, np.ndarray) or not isinstance(y, np.ndarray):
        raise ValueError("X and y must be numpy arrays.")
    if X.ndim != 2 or y.ndim != 1:
        raise ValueError("X must be a 2D array and y must be a 1D array.")
    if X.shape[0] != y.shape[0]:
        raise ValueError("X and y must have the same number of samples.")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)
    clf = RandomForestClassifier(n_estimators=50, random_state=1)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    print("ML Failure Prediction Report:")
    print(classification_report(y_test, y_pred))
    return clf

def predict_failure(clf, new_data):
    if not isinstance(new_data, np.ndarray):
        raise ValueError("new_data must be a numpy array.")
    if new_data.ndim != 2:
        raise ValueError("new_data must be a 2D array.")
    return clf.predict(new_data)
