import pickle

# Import all necessary libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

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
        return 1
    elif row['home_goals'] < row['away_goals']:
        return 2
    else:
        return 0  
    
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
        # Do something here to train the model
        matches = train_data.copy()
        valid_matches = matches[matches['score'].notna() & matches['score'].str.contains(':')].copy()
        valid_matches[['home_goals', 'away_goals']] = (valid_matches['score'].str.split(':', expand=True).astype(int))
        valid_matches['result'] = valid_matches.apply(get_result, axis=1)

        # valid_matches["season"] = valid_matches["season"].str.split("-").str[0].astype(int)
        valid_matches["date"] = valid_matches.apply(generate_date, axis=1)
        valid_matches[["month", "day", "year"]] = valid_matches["date"].str.split("/", expand=True).astype(int)
        
        # valid_matches.info()
        # valid_matches.describe()
        # valid_matches.isna().sum()
        # valid_matches["time"].value_counts()

        valid_matches = valid_matches.drop(columns=["season", "date", "score", "time", "home_goals", "away_goals"])  
        self.teams = pd.unique(valid_matches[["home_team", "away_team"]].values.ravel()) # Save the teams that have been used to fit the data

        encoder = OneHotEncoder(handle_unknown="ignore", categories=[self.teams, self.teams])
        encoded = encoder.fit_transform(valid_matches[["home_team", "away_team"]])
        encoded_df = pd.DataFrame(encoded.toarray(), columns=encoder.get_feature_names_out(["home_team", "away_team"]))

        valid_matches = valid_matches.drop(columns=["home_team", "away_team"])

        df_final = pd.concat([encoded_df, valid_matches], axis=1)
        
        """
        correlation_matrix = df_final.corr()
        corr_with_target = correlation_matrix["result"].sort_values(ascending=False)

        plt.figure(figsize=(10, 8))
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f') 
        plt.title('Correlation Matrix')
        plt.savefig("correlation_matrix.png", dpi=300)  # Guardar en PNG
        plt.show()

        plt.figure(figsize=(12,6))
        corr_with_target.plot(kind="bar")
        plt.ylabel("Correlación con result_num")
        plt.title("Correlación de variables con el resultado")
        plt.tight_layout()
        plt.savefig("correlation_with_result.png", dpi=300)
        plt.show()
        

        df_final.info()
        df_final.describe()
        df_final.isna().sum()
        print(df_final.head())
        """

        # Train the model
        y_train = df_final["result"]
        X_train = df_final.drop("result", axis=1)

        # Define the model
        gradient_boosting = GradientBoostingClassifier()

        # Define the model
        log_reg_liblinear = LogisticRegression(solver='liblinear', max_iter=10000)

        # Define the model
        random_forest = RandomForestClassifier(random_state=42)

        """

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
        self.trained_model = grid_search_logReg_liblinear.best_estimator_
        """

        """
        # Define hyperparameter grid for RandomSearch
        param_dist_gradient = {
            'n_estimators': [100, 200, 300],
            'learning_rate': [0.01, 0.05, 0.1, 0.2],
            'max_depth': [3, 4, 5, 6],
            'subsample': [0.6, 0.8, 1.0],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 3, 5]
        }

        random_search_gradient = RandomizedSearchCV(
            gradient_boosting,
            param_distributions=param_dist_gradient,
            n_iter=5,  # número of random combinations to test
            cv=4,       # cross-validation 4-fold
            scoring='accuracy',  # 'accuracy' or 'f1_macro' or 'f1_weighted'
            n_jobs=-1,
            random_state=42,
            verbose=1
        )

        random_search_gradient.fit(X_train, y_train)
        
        self.trained_model = random_search_gradient.best_estimator_
        """

        # gradient_boosting.fit(X_train, y_train)
        # self.trained_model = gradient_boosting

        # log_reg_liblinear.fit(X_train, y_train)
        # self.trained_model = log_reg_liblinear

        random_forest.fit(X_train, y_train)
        self.trained_model = random_forest
        

        print(df_final)

    def predict(self, predict_data):
        # Do something here to predict
        matches = predict_data.copy()
        print("MATCHES")
        print(matches)
        valid_matches = matches[matches['score'].notna() & matches['score'].str.contains(':')].copy()
        valid_matches[['home_goals', 'away_goals']] = (valid_matches['score'].str.split(':', expand=True).astype(int))
        valid_matches['result'] = valid_matches.apply(get_result, axis=1)

        # valid_matches["season"] = valid_matches["season"].str.split("-").str[0].astype(int)
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

        y_test = df_final["result"]
        X_test = df_final.drop("result", axis=1)

        y_pred = self.trained_model.predict(X_test)
        
        feature_names = sorted(y_test.unique())
        conf_mat_liblinear = confusion_matrix(y_test, y_pred, labels=feature_names)
        disp_liblinear = ConfusionMatrixDisplay(confusion_matrix=conf_mat_liblinear, display_labels=feature_names)
        disp_liblinear.plot()
        plt.title("Logistic Regression with liblinear solver")
        plt.savefig("confusion_matrix.png", dpi=300)
        
        print(df_final)
        print(f"Accuracy on test Gradient Boosting Classifier: {accuracy_score(y_test, y_pred):.4f}")

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
