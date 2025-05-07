''' This file contains the main classs for the training of the 
machine learning model to correctly classify malignant and benign
breast cancer tumors. '''


# ---------------------- Imports --------------------
import warnings
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.base import clone
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.metrics import (matthews_corrcoef, balanced_accuracy_score, f1_score, roc_auc_score)



# ---------------------- Class Implementation --------------------
class RepeatedNestedCV:
    def __init__(self, X, y, estimators, param_grids, repeats=10, outer_folds=5, inner_folds=3, random_state=42):
        """
        Initialize the Repeated Nested CV class.

        Parameters:
        - X, y: Dataset features and labels
        - estimators: Dictionary of estimators {name: sklearn model}
        - param_grids: Dictionary of hyperparameter grids {name: param_grid}
        - repeats: Number of repeated nCV rounds (R)
        - outer_folds: Number of outer folds (N)
        - inner_folds: Number of inner folds (K)
        - random_state: Base random seed for reproducibility
        """
        self.X = X
        self.y = y
        self.estimators = estimators
        self.param_grids = param_grids
        self.repeats = repeats
        self.outer_folds = outer_folds
        self.inner_folds = inner_folds
        self.random_state = random_state
        self.results = []

    # ---------------------- Evaluation Metrics --------------------
    def _evaluate_metrics(self, y_true, y_pred, y_prob):
        """
        Evaluate key metrics for imbalanced classification.

        Returns:
        - Dictionary of metric scores
        """
        return {
            'mcc': matthews_corrcoef(y_true, y_pred),
            'balanced_accuracy': balanced_accuracy_score(y_true, y_pred),
            'f1': f1_score(y_true, y_pred),
            'roc_auc': roc_auc_score(y_true, y_prob)
        }

    # ---------------------- Nested Cross-Validation --------------------
    def _get_best_model(self, X_train, y_train, model_name, fold_seed):
        """
        Runs inner-loop hyperparameter tuning for one model.
        """
        estimator = clone(self.estimators[model_name])
        param_grid = self.param_grids[model_name]
        inner_cv = StratifiedKFold(n_splits=self.inner_folds, shuffle=True, random_state=fold_seed)

        gs = GridSearchCV(
            estimator=estimator,
            param_grid=param_grid,
            scoring='f1',
            cv=inner_cv,
            n_jobs=1
        )

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            gs.fit(X_train, y_train)

        return gs.best_estimator_, gs.best_params_

    def _evaluate_on_outer_fold(self, model, X_test, y_test):
        """
        Evaluates the selected model on the outer test fold.
        """
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            y_pred = model.predict(X_test)
            y_prob = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else np.zeros_like(y_pred)
        
        return self._evaluate_metrics(y_test, y_pred, y_prob)

    # ---------------------- Repeated Nested Cross-Validation --------------------
    def run(self):
        """
        Main logic: run repeated nested cross-validation.
        Returns:
        - DataFrame with evaluation metrics for each model, fold, and repetition.
        """
        for repeat in range(self.repeats):
            print(f"\n Repeat {repeat+1}/{self.repeats}")
            outer_cv = StratifiedKFold(n_splits=self.outer_folds, shuffle=True, random_state=self.random_state + repeat)

            for model_name in self.estimators:
                print(f" Model: {model_name}")

                for fold_idx, (train_idx, test_idx) in enumerate(outer_cv.split(self.X, self.y)):
                    X_train, X_test = self.X[train_idx], self.X[test_idx]
                    y_train, y_test = self.y[train_idx], self.y[test_idx]

                    # Inner loop: get best model
                    best_model, best_params = self._get_best_model(X_train, y_train, model_name, self.random_state + fold_idx)

                    # Outer loop evaluation
                    metrics = self._evaluate_on_outer_fold(best_model, X_test, y_test)
                    metrics.update({
                        'model': model_name,
                        'fold': fold_idx,
                        'repeat': repeat + 1,
                        'best_params': best_params
                    })
                    self.results.append(metrics)

        return pd.DataFrame(self.results)
    
    # ---------------------- Summary of Results --------------------
    def summarize_results(self):
        """
        Summarizes evaluation metrics for each model:
        - Median and mean for each metric
        """

        df = pd.DataFrame(self.results)
        metrics = ['mcc', 'roc_auc', 'balanced_accuracy', 'f1']

        # Mean and Median Summary
        summary = df.groupby('model')[metrics].agg(['median', 'mean'])
        summary.columns = ['_'.join(col) for col in summary.columns] 
        summary = summary.round(4)

        return summary
    
    # ---------------------- Bootstrap Confidence Intervals --------------------
    def bootstrap_cis(self, n_bootstraps=1000, ci_level=0.95, seed=42):
        """
        Compute confidence intervals for each model and metric using bootstrapping.
    
        Returns:
            - Dataframe with confidence intervals for each model and metric.
        """
        rng = np.random.default_rng(seed)
        df = pd.DataFrame(self.results)
        metrics = ['mcc', 'roc_auc', 'balanced_accuracy', 'f1']

        records = []

        for model in df['model'].unique():
            model_df = df[df['model'] == model]

            for metric in metrics:
                medians = []
                for _ in range(n_bootstraps):
                    sample = model_df[metric].sample(frac=1, replace=True, random_state=rng.integers(0, 1e6))
                    medians.append(np.median(sample))

                lower = np.percentile(medians, (1 - ci_level) / 2 * 100)
                upper = np.percentile(medians, (1 + ci_level) / 2 * 100)
                records.append({
                    'model': model,
                    'metric': metric,
                    'CI_low': lower,
                    'CI_high': upper
                })

        return pd.DataFrame(records).set_index(['model', 'metric'])
    

    # ---------------------- Plots --------------------
    def plot_metric_boxplots(self, save_path=None):
        """
        Plot 2x2 boxplots of model performance across all outer folds and repeats.
        """
        metrics = ['mcc', 'roc_auc', 'balanced_accuracy', 'f1']
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        axes = axes.flatten()

        for i, metric in enumerate(metrics):
            sns.boxplot(data=pd.DataFrame(self.results), x='model', y=metric, ax=axes[i], color='skyblue')
            axes[i].set_title(f'{metric} across Outer CV Folds')
            axes[i].tick_params(axis='x', rotation=45)

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300)
        plt.show()