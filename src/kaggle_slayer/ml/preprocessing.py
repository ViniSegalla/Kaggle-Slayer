import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


class Features(BaseEstimator, TransformerMixin):
    """Row features and fold-fit dtype schema; imputation/encoding stay inside each fold."""

    def __init__(self, profile="generic"):
        self.profile = profile

    def fit(self, X, y=None):
        self.numeric_columns_ = set(X.select_dtypes(include="number").columns)
        if self.profile == "titanic":
            self.numeric_columns_.update({"FamilySize", "IsAlone", "FarePerPerson"})
        return self

    def transform(self, X):
        frame = X.copy()
        if self.profile == "titanic" and {"SibSp", "Parch"}.issubset(frame):
            frame["FamilySize"] = frame["SibSp"] + frame["Parch"] + 1
            frame["IsAlone"] = (frame["FamilySize"] == 1).astype(int)
            if "Fare" in frame:
                frame["FarePerPerson"] = frame["Fare"] / frame["FamilySize"].clip(lower=1)
        for column in frame:
            if column in self.numeric_columns_:
                frame[column] = (
                    pd.to_numeric(frame[column], errors="coerce")
                    .astype(float)
                    .replace([np.inf, -np.inf], np.nan)
                )
            else:
                frame[column] = frame[column].map(
                    lambda value: str(value) if pd.notna(value) else np.nan
                )
        return frame

    def get_feature_names_out(self, input_features=None):
        return input_features


def preprocessor(profile: str) -> Pipeline:
    numeric = Pipeline(
        [
            ("impute", SimpleImputer(strategy="median", keep_empty_features=True)),
            ("scale", StandardScaler()),
        ]
    )
    categorical = Pipeline(
        [
            ("impute", SimpleImputer(strategy="constant", fill_value="__missing__")),
            (
                "encode",
                OneHotEncoder(
                    handle_unknown="infrequent_if_exist",
                    min_frequency=2,
                    max_categories=64,
                    sparse_output=False,
                ),
            ),
        ]
    )
    return Pipeline(
        [
            ("features", Features(profile)),
            (
                "columns",
                ColumnTransformer(
                    [
                        ("numeric", numeric, make_column_selector(dtype_include=np.number)),
                        ("categorical", categorical, make_column_selector(dtype_exclude=np.number)),
                    ],
                    remainder="drop",
                    verbose_feature_names_out=True,
                ),
            ),
        ]
    )
