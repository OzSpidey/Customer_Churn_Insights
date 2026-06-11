"""XGBoost pipeline training, persistence, SHAP computation, and inference helpers."""

import pickle
import numpy as np
import pandas as pd
import shap

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    confusion_matrix, roc_curve, precision_recall_curve,
)
from xgboost import XGBClassifier

from config import (
    MODEL_PATH, SHAP_PATH,
    CATEGORICAL_FEATURES, NUMERICAL_FEATURES, ALL_FEATURES,
)


def _build_pipeline() -> Pipeline:
    """Construct the sklearn preprocessing + XGBoost pipeline."""
    preprocessor = ColumnTransformer(
        [
            ("num", StandardScaler(), NUMERICAL_FEATURES),
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                CATEGORICAL_FEATURES,
            ),
        ],
        remainder="drop",
    )
    classifier = XGBClassifier(
        n_estimators=300,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric="logloss",
        base_score=0.5,
        random_state=42,
    )
    return Pipeline([("preprocessor", preprocessor), ("classifier", classifier)])


def train_and_save(df: pd.DataFrame) -> None:
    """Train the pipeline on the full dataset and persist model + SHAP values."""
    X = df[ALL_FEATURES].copy()
    y = df["Churn"].values

    pipeline = _build_pipeline()
    pipeline.fit(X, y)

    with open(MODEL_PATH, "wb") as fh:
        pickle.dump(pipeline, fh)
    print("Model saved to", MODEL_PATH)

    preprocessor  = pipeline.named_steps["preprocessor"]
    classifier    = pipeline.named_steps["classifier"]
    feature_names = preprocessor.get_feature_names_out()

    sample_idx    = np.random.choice(len(X), size=min(500, len(X)), replace=False)
    X_sample      = X.iloc[sample_idx]
    X_transformed = preprocessor.transform(X_sample)
    X_sample_df   = pd.DataFrame(X_transformed, columns=feature_names)

    explainer  = shap.TreeExplainer(classifier)
    shap_expl  = explainer(X_sample_df)

    shap_payload = {
        "shap_values":   shap_expl.values,
        "feature_names": list(feature_names),
        "X_sample":      X_sample_df,
    }
    with open(SHAP_PATH, "wb") as fh:
        pickle.dump(shap_payload, fh)
    print("SHAP values saved to", SHAP_PATH)


def load_model() -> Pipeline:
    """Load and return the trained sklearn pipeline from disk."""
    with open(MODEL_PATH, "rb") as fh:
        return pickle.load(fh)


def load_shap() -> tuple:
    """Load and return (shap_values ndarray, feature_names list, X_sample DataFrame)."""
    with open(SHAP_PATH, "rb") as fh:
        payload = pickle.load(fh)
    return payload["shap_values"], payload["feature_names"], payload["X_sample"]


def predict_single(pipeline: Pipeline, customer_dict: dict) -> float:
    """Return the churn probability (0-1) for a single customer represented as a dict."""
    df_row = pd.DataFrame([customer_dict])
    for col in NUMERICAL_FEATURES:
        df_row[col] = pd.to_numeric(df_row[col], errors="coerce")
    prob = pipeline.predict_proba(df_row[ALL_FEATURES])[0, 1]
    return float(prob)


def get_model_metrics(pipeline: Pipeline, df: pd.DataFrame) -> dict:
    """Evaluate on a held-out 20 pct test split; return accuracy, precision, recall, f1, auc plus curve data."""
    X = df[ALL_FEATURES].copy()
    y = df["Churn"].values
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    pipeline_copy = _build_pipeline()
    pipeline_copy.fit(X_train, y_train)
    y_pred  = pipeline_copy.predict(X_test)
    y_proba = pipeline_copy.predict_proba(X_test)[:, 1]

    fpr, tpr, _ = roc_curve(y_test, y_proba)
    prec_arr, rec_arr, _ = precision_recall_curve(y_test, y_proba)

    return {
        "accuracy":         round(accuracy_score(y_test, y_pred), 4),
        "precision":        round(precision_score(y_test, y_pred, zero_division=0), 4),
        "recall":           round(recall_score(y_test, y_pred, zero_division=0), 4),
        "f1":               round(f1_score(y_test, y_pred, zero_division=0), 4),
        "auc":              round(roc_auc_score(y_test, y_proba), 4),
        "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
        "roc_fpr":          fpr.tolist(),
        "roc_tpr":          tpr.tolist(),
        "pr_precision":     prec_arr.tolist(),
        "pr_recall":        rec_arr.tolist(),
        "y_test":           y_test.tolist(),
        "y_proba":          y_proba.tolist(),
    }


def get_feature_importances(pipeline: Pipeline) -> pd.Series:
    """Return a Series of XGBoost feature importances indexed by feature name."""
    preprocessor = pipeline.named_steps["preprocessor"]
    classifier   = pipeline.named_steps["classifier"]
    names        = preprocessor.get_feature_names_out()
    imps         = classifier.feature_importances_
    return pd.Series(imps, index=names).sort_values(ascending=False)
