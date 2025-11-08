import pickle

# Import all necessary libraries
import numpy as np
import pandas as pd

# For machine learning (you will probably need to add more)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import classification_report, mean_squared_error

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
        valid_matches
        print(" Training completed.")

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
