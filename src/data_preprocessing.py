import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder, StandardScaler
from data_ingestion import data_ingestion

def data_preprocessing():
    df_train, df_test = data_ingestion()
    df_train.drop('index', axis=1, inplace=True)
    df_test.drop('index', axis=1, inplace=True)
    return df_train, df_test

def get_preprocessor():
    ordinal_features = ['Dietary Habits', 'Sleep Duration']
    ordinal_categories = [
        ['Unhealthy', 'Moderate', 'Healthy'],
        ['Less than 5 hours', '5-6 hours', '6-7 hours', '7-8 hours', 'More than 8 hours']
    ]

    nominal_features = ['Gender', 'Have you ever had suicidal thoughts ?', 'Family History of Mental Illness']
    numeric_features = ['Age', 'Academic Pressure', 'Study Satisfaction', 'Study Hours', 'Financial Stress']

    preprocessor = ColumnTransformer(
        transformers=[
            ('ord', OrdinalEncoder(categories=ordinal_categories), ordinal_features),
            ('nom', OneHotEncoder(drop='first'), nominal_features),
            ('num', StandardScaler(), numeric_features)
        ]
    )

    return preprocessor

if __name__ == "__main__":
    data_preprocessing()
