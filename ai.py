# AI Project Full Template with Model Testing and Reporting

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, f1_score, accuracy_score, recall_score, roc_auc_score
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier, GradientBoostingRegressor
from xgboost import XGBClassifier
from sklearn.metrics import RocCurveDisplay

# --- DATASET LOAD & PREPROCESSING (assumed to be done) ---

df = pd.read_csv('iris_обработанный.csv')
X = df.drop('species', axis=1)
y = df['species']

# --- SPLITTING DATA ---
# 60% train, 20% val, 20% test
X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.25, random_state=42)

# --- TRAINING FUNCTION ---
def train_model(model, params, X_train, y_train):
    model.set_params(**params)
    model.fit(X_train, y_train)
    return model

# --- EVALUATION FUNCTION FOR REGRESSION ---
def evaluate_regression(model, X_test, y_test):
    preds = model.predict(X_test)
    return {
        'MAE': mean_absolute_error(y_test, preds),
        'MSE': mean_squared_error(y_test, preds),
        'R2': r2_score(y_test, preds)
    }

# --- EVALUATION FUNCTION FOR CLASSIFICATION ---
def evaluate_classification(model, X_test, y_test):
    preds = model.predict(X_test)
    proba = model.predict_proba(X_test) if hasattr(model, 'predict_proba') else None
    metrics = {
        'Accuracy': accuracy_score(y_test, preds),
        'F1_score': f1_score(y_test, preds, average='weighted'),
        'Recall': recall_score(y_test, preds, average='weighted')
    }
    if proba is not None and len(np.unique(y_test)) == 2:
        metrics['ROC_AUC'] = roc_auc_score(y_test, proba[:, 1])
    return metrics

# --- ROC CURVE FUNCTION ---
def plot_roc(model, X_test, y_test):
    RocCurveDisplay.from_estimator(model, X_test, y_test)
    plt.title("ROC Curve")
    plt.grid()
    plt.show()

# --- HYPERPARAMETER TESTING FUNCTION WITH PLOTS ---
def TestingModel(model_class, param_name, values, fixed_params, X_train, X_test, y_train, y_test, task='classification'):
    scores = []
    for value in values:
        params = fixed_params.copy()
        params[param_name] = value
        model = model_class(**params)
        model.fit(X_train, y_train)
        if task == 'classification':
            metrics = evaluate_classification(model, X_test, y_test)
        else:
            metrics = evaluate_regression(model, X_test, y_test)
        print(f"\nTesting {param_name} = {value}")
        for k, v in metrics.items():
            print(f"{k}: {v:.4f}")
        scores.append((value, metrics))

    # Plotting results
    if task == 'classification':
        metric_names = ['Accuracy', 'F1_score', 'Recall']
        for metric in metric_names:
            plt.plot([v for v, m in scores], [m[metric] for v, m in scores], label=metric)
        plt.title(f"Effect of {param_name} on metrics")
        plt.xlabel(param_name)
        plt.ylabel("Score")
        plt.legend()
        plt.grid()
        plt.show()

# --- MODEL COMPARISON PLOT ---
def compare_classification_models(models_dict, X_test, y_test):
    names, accuracies, f1s, recalls = [], [], [], []
    for name, model in models_dict.items():
        metrics = evaluate_classification(model, X_test, y_test)
        names.append(name)
        accuracies.append(metrics['Accuracy'])
        f1s.append(metrics['F1_score'])
        recalls.append(metrics['Recall'])

    x = np.arange(len(names))
    width = 0.25

    plt.bar(x - width, accuracies, width, label='Accuracy')
    plt.bar(x, f1s, width, label='F1-score')
    plt.bar(x + width, recalls, width, label='Recall')

    plt.xticks(x, names)
    plt.ylabel("Score")
    plt.title("Model Comparison")
    plt.legend()
    plt.grid()
    plt.show()

# --- MODEL DEFINITIONS WITH FULL HYPERPARAMETERS ---
reg_models = {
    'LinearRegression': LinearRegression(),
    'RandomForestRegressor': RandomForestRegressor(
        n_estimators=100, max_depth=None, min_samples_split=2,
        min_samples_leaf=1, max_features='auto', random_state=42
    ),
    'GradientBoostingRegressor': GradientBoostingRegressor(
        loss='squared_error', learning_rate=0.1, n_estimators=100,
        subsample=1.0, criterion='friedman_mse', max_depth=3,
        min_samples_split=2, min_samples_leaf=1, random_state=42
    )
}

clf_models = {
    'LogisticRegression': LogisticRegression(
        penalty='l2', C=1.0, solver='lbfgs', max_iter=1000, random_state=42
    ),
    'RandomForestClassifier': RandomForestClassifier(
        n_estimators=100, max_depth=None, min_samples_split=2,
        min_samples_leaf=1, max_features='auto', random_state=42
    ),
    'XGBClassifier': XGBClassifier(
        learning_rate=0.1, n_estimators=100, max_depth=6, min_child_weight=1,
        gamma=0, subsample=1.0, colsample_bytree=1.0, use_label_encoder=False,
        eval_metric='logloss', random_state=42
    )
}

# --- GRIDSEARCH EXAMPLE ---
# grid = GridSearchCV(RandomForestClassifier(), {
#     'n_estimators': [50, 100, 200],
#     'max_depth': [5, 10, 20]
# }, scoring='f1_weighted')
# grid.fit(X_train, y_train)
# print(grid.best_params_)
# print(evaluate_classification(grid.best_estimator_, X_test, y_test))
