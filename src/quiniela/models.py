import pickle

# Import all necessary libraries
import pandas as pd
import matplotlib.pyplot as plt

# For machine learning (you will probably need to add more)
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay

# Model
from sklearn.ensemble import RandomForestClassifier

def get_result(row):
    if row['home_goals'] > row['away_goals']:
        return '1'
    elif row['home_goals'] < row['away_goals']:
        return '2'
    else:
        return 'X'  
    
def generate_date(row):
    date = row['date'].split("/")
    month = date[0]
    day = date[1]
    year = date[2]
    season = row['season'].split("-")[0]
    year = season[0:2] + year
    return f"{month}/{day}/{year}"



class QuinielaModel:

    def train(self, train_data):

        # Adapt data to do the training, all data has to be type int or float except our prediction column
        matches = train_data.copy()
        valid_matches = matches[matches['score'].notna() & matches['score'].str.contains(':')].copy()
        valid_matches[['home_goals', 'away_goals']] = (valid_matches['score'].str.split(':', expand=True).astype(int))
        valid_matches['result'] = valid_matches.apply(get_result, axis=1)

        valid_matches["date"] = valid_matches.apply(generate_date, axis=1)
        valid_matches[["month", "day", "year"]] = valid_matches["date"].str.split("/", expand=True).astype(int)

        valid_matches = valid_matches.drop(columns=["season", "date", "score", "time", "home_goals", "away_goals"])  
        self.teams = pd.unique(valid_matches[["home_team", "away_team"]].values.ravel()) # Save the teams that have been used to fit the data

        encoder = OneHotEncoder(handle_unknown="ignore", categories=[self.teams, self.teams])
        encoded = encoder.fit_transform(valid_matches[["home_team", "away_team"]])
        encoded_df = pd.DataFrame(encoded.toarray(), columns=encoder.get_feature_names_out(["home_team", "away_team"]))

        valid_matches = valid_matches.drop(columns=["home_team", "away_team"])

        df_final = pd.concat([encoded_df, valid_matches], axis=1)

        # Define the data to train
        y_train = df_final["result"]
        X_train = df_final.drop("result", axis=1)

        # Define the model
        random_forest = RandomForestClassifier(random_state=42)

        # Train the model
        random_forest.fit(X_train, y_train)
        self.trained_model = random_forest

    def predict(self, predict_data):
        
        # Adapt data to do the prediction, all data has to be type int or float except our prediction column
        valid_matches = predict_data.copy()
        valid_matches[['home_goals', 'away_goals']] = (valid_matches['score'].str.split(':', expand=True).astype(int))
        valid_matches['result'] = valid_matches.apply(get_result, axis=1)

        valid_matches["date"] = valid_matches.apply(generate_date, axis=1)
        valid_matches[["month", "day", "year"]] = valid_matches["date"].str.split("/", expand=True).astype(int)
        
        valid_matches.info()
        valid_matches.describe()

        valid_matches = valid_matches.drop(columns=["season", "date", "score", "time", "home_goals", "away_goals"])  

        encoder = OneHotEncoder(handle_unknown="ignore", categories=[self.teams, self.teams])
        encoded = encoder.fit_transform(valid_matches[["home_team", "away_team"]])
        encoded_df = pd.DataFrame(encoded.toarray(), columns=encoder.get_feature_names_out(["home_team", "away_team"]))
        
        valid_matches = valid_matches.drop(columns=["home_team", "away_team"])
        
        df_final = pd.concat([encoded_df, valid_matches], axis=1)

        # Define the data to predict
        y_test = df_final["result"]
        X_test = df_final.drop("result", axis=1)

        # Do the prediction
        y_pred = self.trained_model.predict(X_test)
        
        # Create the Confusion Matrix
        feature_names = sorted(y_test.unique())
        conf_mat = confusion_matrix(y_test, y_pred, labels=feature_names)
        disp = ConfusionMatrixDisplay(confusion_matrix=conf_mat, display_labels=feature_names)
        disp.plot()
        plt.title("Random Forest Classifier")
        plt.savefig("confusion_matrix.png", dpi=300)
        
        print(f"Accuracy on test Random Forest Classifier: {accuracy_score(y_test, y_pred):.4f}")

        return y_pred
        

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
