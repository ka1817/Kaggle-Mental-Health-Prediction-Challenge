# src/model_building.py
import os
import joblib
import pandas as pd
import mlflow
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from data_preprocessing import data_preprocessing, get_preprocessor

mlflow.set_tracking_uri("http://127.0.0.1:5000")
mlflow.set_experiment("Depression_Prediction_Models1")

def train_and_log_models():
    df_train, df_test = data_preprocessing()

    # Split features and target
    X_train = df_train.drop('Depression', axis=1)
    y_train = df_train['Depression']
    X_test = df_test.copy()

    # Get preprocessor
    preprocessor = get_preprocessor()

    # Define models with parameter grids for tuning
    models = {
        'lr': {
            'model': LogisticRegression(max_iter=1000),
            'params': {
                'classifier__C': [0.1, 1, 10],
                'classifier__penalty': ['l2'],
                'classifier__solver': ['lbfgs']
            }
        },
        'dt': {
            'model': DecisionTreeClassifier(random_state=42),
            'params': {
                'classifier__max_depth': [3, 5, 10],
                'classifier__min_samples_split': [2, 5, 10]
            }
        },
        'svc': {
            'model': SVC(probability=True),
            'params': {
                'classifier__C': [0.1, 1, 10],
                'classifier__kernel': ['linear', 'rbf']
            }
        }
    }

    model_filename_map = {
        'lr': 'LogisticRegression',
        'dt': 'DecisionTreeClassifier',
        'svc': 'SVC'
    }

    model_dir = os.path.abspath(os.path.join(os.getcwd(), "..", "models"))
    os.makedirs(model_dir, exist_ok=True)

    # Train and log each model with MLflow
    for name, config in models.items():
        with mlflow.start_run(run_name=name):
            pipeline = Pipeline([
                ('preprocessor', preprocessor),
                ('classifier', config['model'])
            ])

            grid = GridSearchCV(pipeline, config['params'], cv=3, n_jobs=-1)
            grid.fit(X_train, y_train)

            y_pred = grid.predict(X_test)
            print(f"\n{name.upper()} Predictions (first 10):\n{y_pred[:10]}")

            mlflow.log_params(grid.best_params_)

            # Save best model locally without MLflow metadata
            model_name = model_filename_map[name]
            model_path = os.path.join(model_dir, f"{model_name}.pkl")
            joblib.dump(grid.best_estimator_, model_path)

if __name__ == "__main__":
    train_and_log_models()