'''Utility module for data handling, preprocessing, model training, hyperparameter tuning, 
and model persistence in a machine learning pipeline.'''

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.base import clone
from sklearn.impute import KNNImputer
from imblearn.over_sampling import SMOTE
from sklearn.feature_selection import RFECV
from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import RandomOverSampler
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.model_selection import GridSearchCV, StratifiedKFold


# -------------------- 1. Data Handling --------------------
def load_dataset(path: str):
    '''Load a dataset from a given path.'''
    df = pd.read_csv(path)
    return df


# -------------------- 2. Data Splitting --------------------
def split_features_target(df, target_col):
    '''Split the dataframe into features and target.'''
    X = df.drop(columns=[target_col])
    y = df[target_col]
    return X.to_numpy(), y.to_numpy()


# -------------------- 3. Label Encoding --------------------
def label_encode(y):
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    return y_encoded


# -------------------- 4. Training --------------------
def train_model(X_train, y_train, model):
    '''Fit a model on the training data.'''
    model.fit(X_train, y_train)
    return model

# -------------------- 5. Hyperparameter Tuning  --------------------
def select_best_hyperparameters(X, y, model, param_grid, scoring='f1', cv=5, n_jobs=-1, random_state=42):
    '''
    Perform 5-fold CV to select the best hyperparameters for a given model.

    Parameters:
    - X, y: dataset features and labels
    - model: the model object (e.g., LogisticRegression())
    - param_grid: hyperparameter grid (dict)
    - scoring: scoring metric (default = 'f1')
    - cv: number of CV folds
    - n_jobs: parallel jobs for grid search
    - random_state: for reproducibility (used if estimator supports it)

    Returns:
    - best_model: trained model with best hyperparameters
    - best_params: dictionary of best parameters
    - cv_results: full GridSearchCV result as DataFrame
    '''
    cv_splitter = StratifiedKFold(n_splits=cv, shuffle=True, random_state=random_state)

    grid = GridSearchCV(
        estimator=clone(model),
        param_grid=param_grid,
        cv=cv_splitter,
        scoring=scoring,
        n_jobs=n_jobs,
        return_train_score=True
    )

    trained_grid = train_model(X, y, grid)

    best_model = trained_grid.best_estimator_
    best_params = trained_grid.best_params_
    cv_results = pd.DataFrame(trained_grid.cv_results_)

    return best_model, best_params, cv_results

# -------------------- 6. Missing Values Handling --------------------
def handling_missing_values(X):
    '''Applies KNN imputation to the feature columns.'''
    imputer = KNNImputer()
    X_imputed = imputer.fit_transform(X)
    return X_imputed


# -------------------- 7. Model Saving & Loading --------------------
def save_model(model, path: str):
    '''Save the trained model to a file.'''
    joblib.dump(model, path)

def load_model(path: str):
    '''Load a model from a file.'''
    model = joblib.load(path)
    return model


# -------------------- BONUS 1 --------------------
def select_top_k_features(X, y, k=5):
    """
    Select top k features using ANOVA F-test (f_classif).

    Parameters:
    - X: Feature matrix (NumPy or DataFrame)
    - y: Target labels
    - k: Number of top features to select

    Returns:
    - X_selected: Reduced feature matrix
    - selected_indices: Indices of selected features
    - selector: Fitted SelectKBest object
    """
    selector = SelectKBest(score_func=f_classif, k=k)
    X_selected = selector.fit_transform(X, y)
    selected_indices = selector.get_support(indices=True)
    return X_selected, selected_indices, selector


def select_features_rfecv(X, y, model, scoring='f1', cv=5):
    """
    Perform Recursive Feature Elimination with Cross-Validation (RFECV).

    Parameters:
    - X, y: Dataset
    - model: Estimator (e.g., LogisticRegression)
    - scoring: Metric to optimize during selection
    - cv: Number of CV folds

    Returns:
    - X_selected: Reduced feature matrix
    - selected_indices: Indices of selected features
    - selector: Fitted RFECV object
    """
    selector = RFECV(estimator=clone(model), step=1, cv=cv, scoring=scoring, n_jobs=-1)
    X_selected = selector.fit_transform(X, y)
    selected_indices = selector.get_support(indices=True)

    print("Optimal number of features:", selector.n_features_)

    return X_selected, selected_indices, selector

from sklearn.model_selection import cross_validate

def evaluate_model_cv(X, y, model, cv=5):
    """
    Perform cross-validation and return evaluation metrics.

    Parameters:
    - X, y: Dataset
    - model: Estimator
    - cv: Number of folds

    Returns:
    - Dictionary of scores (mean across folds)
    """
    scoring = ['f1', 'roc_auc', 'balanced_accuracy', 'matthews_corrcoef']
    scores = cross_validate(model, X, y, cv=cv, scoring=scoring, n_jobs=-1)
    return {metric: scores[f'test_{metric}'].mean() for metric in scoring}



def plot_fs_grouped_barplot(results_dict, metrics=None, title="Performance Comparison"):
    """
    Plot grouped bar chart: FS methods compared across multiple metrics.

    Parameters:
    - results_dict: Dict like {"SelectKBest (k=5)": {"f1": ..., "roc_auc": ..., ...}, ...}
    - metrics: List of metric names to include (optional: defaults to common metrics)
    - title: Plot title
    """
    if metrics is None:
        metrics = ["f1", "roc_auc", "balanced_accuracy", "matthews_corrcoef"]

    methods = list(results_dict.keys())
    num_metrics = len(metrics)
    num_methods = len(methods)
    bar_width = 0.8 / num_methods  # spacing based on number of methods

    x = np.arange(num_metrics)  # metric positions on x-axis
    colors = plt.cm.Set2.colors[:num_methods]

    plt.figure(figsize=(10, 5))
    
    for i, method in enumerate(methods):
        scores = [results_dict[method][metric] for metric in metrics]
        bar_positions = x + i * bar_width
        plt.bar(bar_positions, scores, width=bar_width, label=method, color=colors[i])
    
    # x-ticks in the middle of each group
    mid_positions = x + (bar_width * (num_methods - 1) / 2)
    plt.xticks(mid_positions, [m.upper() for m in metrics])

    plt.ylim(0, 1)
    plt.ylabel("Score")
    plt.title(title)
    plt.legend()
    plt.grid(axis='y', linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.show()


# -------------------- BONUS 2 --------------------
def apply_smote(X, y, random_state=42):
    """
    Apply SMOTE to balance classes by oversampling the minority class.

    Parameters:
    - X, y: Feature matrix and labels

    Returns:
    - X_resampled, y_resampled
    """
    smote = SMOTE(random_state=random_state)
    X_res, y_res = smote.fit_resample(X, y)
    return X_res, y_res

def apply_random_oversampling(X, y, random_state=42):
    """
    Apply random oversampling to balance classes.

    Parameters:
    - X, y: Feature matrix and labels

    Returns:
    - X_resampled, y_resampled
    """
    ros = RandomOverSampler(random_state=random_state)
    X_res, y_res = ros.fit_resample(X, y)
    return X_res, y_res