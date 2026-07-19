"""
Decision Trees: Pruning, Cross-Validation and XGBoost
Movie dataset - regression on Collection, classification on Start_Tech_Oscar.

Run:  python src/run_pipeline.py
"""

import os
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)
from xgboost import XGBRegressor, XGBClassifier


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data")
SEED = 42

REG_TARGET = "Collection"
CLF_TARGET = "Start_Tech_Oscar"

PARAM_GRID = {
    "max_depth": [3, 5, 7, 10],
    "min_samples_split": [2, 5, 10],
    "min_samples_leaf": [1, 2, 5],
    "max_leaf_nodes": [10, 20, 30, None],
}


def load_split(filename, target, stratify):
    df = pd.read_csv(os.path.join(DATA, filename)).dropna()
    y = df[target]
    X = pd.get_dummies(df.drop(target, axis=1), drop_first=True)
    if stratify is True:
        strat = y
    else:
        strat = None
    return train_test_split(X, y, test_size=0.20, random_state=SEED, stratify=strat)


def reg_scores(model, X_test, y_test):
    pred = model.predict(X_test)
    mse = mean_squared_error(y_test, pred)
    return {
        "MAE": mean_absolute_error(y_test, pred),
        "RMSE": np.sqrt(mse),
        "R2": r2_score(y_test, pred),
    }


def clf_scores(model, X_test, y_test):
    pred = model.predict(X_test)
    return {
        "Accuracy": accuracy_score(y_test, pred),
        "Precision": precision_score(y_test, pred),
        "Recall": recall_score(y_test, pred),
        "F1": f1_score(y_test, pred),
        "ConfusionMatrix": confusion_matrix(y_test, pred),
    }


def best_ccp_model(base_model, estimator_cls, X_train, y_train, X_test, y_test):
    path = base_model.cost_complexity_pruning_path(X_train, y_train)
    alphas = path.ccp_alphas[:-1]
    models = []
    scores = []
    i = 0
    while i < len(alphas):
        m = estimator_cls(random_state=SEED, ccp_alpha=alphas[i])
        m.fit(X_train, y_train)
        models.append(m)
        scores.append(m.score(X_test, y_test))
        i = i + 1
    best = int(np.argmax(scores))
    return models[best], alphas[best]


def run_regression():
    print("\n" + "=" * 60)
    print("REGRESSION - target:", REG_TARGET)
    print("=" * 60)

    X_train, X_test, y_train, y_test = load_split(
        "Movie_regression.csv", REG_TARGET, False
    )

    base = DecisionTreeRegressor(random_state=SEED).fit(X_train, y_train)
    print("Unpruned      :", reg_scores(base, X_test, y_test),
          "depth", base.get_depth(), "leaves", base.get_n_leaves())

    pre = DecisionTreeRegressor(
        max_depth=5, min_samples_split=10, min_samples_leaf=5,
        max_leaf_nodes=20, random_state=SEED
    ).fit(X_train, y_train)
    print("Pre-pruned    :", reg_scores(pre, X_test, y_test),
          "depth", pre.get_depth(), "leaves", pre.get_n_leaves())

    post, alpha = best_ccp_model(
        base, DecisionTreeRegressor, X_train, y_train, X_test, y_test
    )
    print("Post-pruned   :", reg_scores(post, X_test, y_test),
          "depth", post.get_depth(), "leaves", post.get_n_leaves(),
          "alpha", round(alpha, 4))

    grid = GridSearchCV(
        DecisionTreeRegressor(random_state=SEED),
        PARAM_GRID, cv=5, scoring="r2", n_jobs=-1
    ).fit(X_train, y_train)
    tuned = grid.best_estimator_
    print("GridSearchCV  :", reg_scores(tuned, X_test, y_test))
    print("  best params :", grid.best_params_)
    print("  best CV R2  :", round(grid.best_score_, 4))

    xgb = XGBRegressor(
        n_estimators=100, learning_rate=0.1, max_depth=5,
        random_state=SEED, verbosity=0
    ).fit(X_train, y_train)
    print("XGBoost       :", reg_scores(xgb, X_test, y_test))

    fi = pd.Series(xgb.feature_importances_, index=X_train.columns)
    print("\nTop 5 features:")
    print(fi.sort_values(ascending=False).head(5))


def run_classification():
    print("\n" + "=" * 60)
    print("CLASSIFICATION - target:", CLF_TARGET)
    print("=" * 60)

    X_train, X_test, y_train, y_test = load_split(
        "Movie_classification.csv", CLF_TARGET, True
    )

    base = DecisionTreeClassifier(random_state=SEED).fit(X_train, y_train)
    print("Unpruned      :", clf_scores(base, X_test, y_test)["F1"],
          "depth", base.get_depth(), "leaves", base.get_n_leaves())

    pre = DecisionTreeClassifier(
        max_depth=5, min_samples_split=10, min_samples_leaf=5,
        max_leaf_nodes=20, random_state=SEED
    ).fit(X_train, y_train)
    print("Pre-pruned    :", clf_scores(pre, X_test, y_test)["F1"],
          "depth", pre.get_depth(), "leaves", pre.get_n_leaves())

    post, alpha = best_ccp_model(
        base, DecisionTreeClassifier, X_train, y_train, X_test, y_test
    )
    print("Post-pruned   :", clf_scores(post, X_test, y_test)["F1"],
          "depth", post.get_depth(), "leaves", post.get_n_leaves(),
          "alpha", round(alpha, 6))

    grid = GridSearchCV(
        DecisionTreeClassifier(random_state=SEED),
        PARAM_GRID, cv=5, scoring="accuracy", n_jobs=-1
    ).fit(X_train, y_train)
    tuned = grid.best_estimator_
    print("GridSearchCV  :", clf_scores(tuned, X_test, y_test)["F1"])
    print("  best params :", grid.best_params_)
    print("  best CV acc :", round(grid.best_score_, 4))

    xgb = XGBClassifier(
        n_estimators=100, learning_rate=0.1, max_depth=5,
        random_state=SEED, eval_metric="logloss", verbosity=0
    ).fit(X_train, y_train)
    result = clf_scores(xgb, X_test, y_test)
    print("XGBoost       :", result["F1"])
    print("Confusion matrix:\n", result["ConfusionMatrix"])

    cv = cross_val_score(base, X_train, y_train, cv=5, scoring="accuracy")
    print("\n5-fold CV accuracy (unpruned):", round(cv.mean(), 4))

    fi = pd.Series(xgb.feature_importances_, index=X_train.columns)
    print("\nTop 5 features:")
    print(fi.sort_values(ascending=False).head(5))


if __name__ == "__main__":
    run_regression()
    run_classification()
