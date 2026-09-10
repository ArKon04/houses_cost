from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

from pathlib import Path

import joblib

import pandas as pd


def save_preprocessor(preprocessor, path):

    path = Path(path)

    path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    joblib.dump(
        preprocessor,
        path
    )


def load_preprocessor(path):

    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(
            f"Preprocessor not found: {path}"
        )

    return joblib.load(path)


def build_preprocessor(numeric_features, categorical_features):

    numeric_pipline = Pipeline(
        steps=[
            (
            "imputer",
            SimpleImputer(strategy= "median")
        ),
        ("scaler", StandardScaler())
    ]
    )   

    categorical_pipline = Pipeline(
        steps=[
            (
                'imputer',
                SimpleImputer(strategy="most_frequent")
            ),
            (
                'encoder',
                OneHotEncoder(
                    handle_unknown="ignore"
                )
            )
        ]
    )

    preprocessor = ColumnTransformer(
        transformers= [
            (
                "numeric",
                numeric_pipline,
                numeric_features
            ),
            (
                "categorical",
                categorical_pipline,
                categorical_features
            )
        ]
    )
    return preprocessor



def get_numeric_categorical_features(df):
    
    numeric = df.select_dtypes(include = ['int64', 'float64']).columns.tolist()

    categrical = df.select_dtypes(include = ['object']).columns.tolist()
    return numeric, categrical