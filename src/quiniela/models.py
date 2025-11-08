import pickle

# Import all necessary libraries
import numpy as np
import pandas as pd

# For machine learning (you will probably need to add more)
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV, cross_validate
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import classification_report, mean_squared_error, accuracy_score, f1_score, confusion_matrix, ConfusionMatrixDisplay

# Models
from sklearn.linear_model import (
    LogisticRegression, PassiveAggressiveClassifier, RidgeClassifier
)
from sklearn.ensemble import (
    RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier,
    HistGradientBoostingClassifier, BaggingClassifier
)
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier

def get_result(row):
    if row['home_goals'] > row['away_goals']:
        return '1'
    elif row['home_goals'] < row['away_goals']:
        return '2'
    else:
        return 'X'  


class QuinielaModel:
    def train(self, train_data):
        # Do something here to train the model
        matches = train_data.copy()
        valid_matches = matches[matches['score'].notna() & matches['score'].str.contains(':')].copy()
        valid_matches[['home_goals', 'away_goals']] = (valid_matches['score'].str.split(':', expand=True).astype(int))
        valid_matches['result'] = valid_matches.apply(get_result, axis=1)
        
        # Split the data into training and testing sets
        train_set, test_set = train_test_split(valid_matches, test_size=0.2, random_state=42)

        # Train the model
        y_train = train_set["quality"]
        X_train = train_set.drop("quality", axis=1)

        y_test = test_set["quality"]
        X_test = test_set.drop("quality", axis=1)

        # Define the model
        log_reg_liblinear = LogisticRegression(solver='liblinear', max_iter=10000)

        # Cross Validation
        print("Cross Validation results:")
        cv_results_log_liblinear = cross_validate(log_reg_liblinear, X_train, y_train, cv=4, scoring=["accuracy", "f1_weighted"])
        print(cv_results_log_liblinear, "\n")

        # Define hyperparameter grid
        param_grid_liblinear = {
            'C': [0.01, 0.1, 1, 10],
            'penalty': ['l1', 'l2']
        }

        # Create GridSearch with cross-validation of 4 partitions
        grid_search_logReg_liblinear = GridSearchCV(
            estimator=log_reg_liblinear,
            param_grid=param_grid_liblinear,
            cv=4,
            scoring='accuracy',
            n_jobs=-1
        )

        grid_search_logReg_liblinear.fit(X_train, y_train)

    def predict(self, predict_data):
        # Do something here to predict
        return ["X" for _ in range(len(predict_data))]

    @classmethod
    def load(cls, filename):
        """Load model from file"""
        with open(filename, "rb") as f:
            model = pickle.load(f)
            assert type(model) is cls
        return model

    def save(self, filename):
        """Save a model in a file"""
        with open(filename, "wb") as f:
            pickle.dump(self, f)
