import os
import matplotlib
matplotlib.use('Agg') 
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import mlflow
from sklearn.metrics import confusion_matrix, roc_curve, auc, precision_recall_curve, average_precision_score
from sklearn.decomposition import PCA
import optuna 
import logging
from typing import List, Any, Dict, Optional # Make sure Optional is imported

logger = logging.getLogger(__name__)

# Define a consistent color palette (Example: based on "Energetic & Playful" concept)
# You can customize these HEX codes
PALETTE = {
    "primary": "#FF4081",    # Vivid Pink
    "secondary": "#9C27B0",  # Deep Purple
    "tertiary": "#00BCD4",   # Cyan/Teal
    "quaternary": "#FFEB3B", # Bright Yellow
    "neutral_dark": "#424242", # Dark Gray
    "neutral_light": "#F5F5F5", # Light Gray
    "success": "#4CAF50",    # Green
    "error": "#F44336",      # Red
    "info": "#2196F3"        # Blue
}

def _ensure_plot_dir(artifacts_path_base: str, subfolder: str = "plots") -> str:
    """Ensures the plot directory exists and returns its path."""
    plot_dir = os.path.join(artifacts_path_base, subfolder)
    os.makedirs(plot_dir, exist_ok=True)
    return plot_dir

def log_feature_importance_plot(model: Any, feature_names: List[str], model_alias: str, task_type: str, artifacts_path_base: str):
    """
    Logs feature importance plot for tree-based or linear models.
    For linear models, it plots coefficient magnitudes.
    """
    plot_dir = _ensure_plot_dir(artifacts_path_base, "feature_importance")
    plot_filename = f"{model_alias.lower()}_{task_type}_feature_importance.png"
    plot_path = os.path.join(plot_dir, plot_filename)

    importances = None
    title_prefix = ""
    is_coefficient_plot = False

    try:
        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
            title_prefix = "Feature Importance"
        elif hasattr(model, 'coef_'):
            is_coefficient_plot = True
            if model.coef_.ndim == 1: # Binary classification or regression
                importances = model.coef_
            elif model.coef_.ndim == 2 and model.coef_.shape[0] == 1: # Some linear models wrap coef_ in 2D array
                importances = model.coef_[0]
            elif model.coef_.ndim == 2 and model.coef_.shape[0] > 1: # Multiclass classification (e.g., Logistic Regression)
                logger.info(f"Model {model_alias} is multiclass; plotting mean absolute coefficient values across classes.")
                importances = np.mean(np.abs(model.coef_), axis=0)
            else:
                logger.warning(f"Coefficient array shape for {model_alias} not recognized for importance plot. Coefficients shape: {model.coef_.shape}")
                return
            title_prefix = "Coefficient Magnitude"
        else:
            logger.info(f"Model {model_alias} does not have 'feature_importances_' or 'coef_' attribute. Skipping importance plot.")
            return

        if importances is None or len(importances) != len(feature_names):
            logger.warning(f"Mismatch in feature importances/coefficients length ({len(importances) if importances is not None else 'None'}) and feature names ({len(feature_names)}) for {model_alias}. Skipping plot.")
            return

        feature_imp_df = pd.DataFrame({'feature': feature_names, 'importance': importances})
        
        # For coefficient plots, sort by absolute magnitude. For feature importances, sort by importance itself.
        if is_coefficient_plot:
            feature_imp_df['abs_importance'] = np.abs(feature_imp_df['importance'])
            feature_imp_df = feature_imp_df.sort_values(by='abs_importance', ascending=False).head(25) # Show top 25
        else:
            feature_imp_df = feature_imp_df.sort_values(by='importance', ascending=False).head(25)

        plt.figure(figsize=(12, max(8, len(feature_imp_df) * 0.4))) # Adjust height based on number of features
        sns.barplot(x='importance', y='feature', data=feature_imp_df, palette="viridis") # Consider using your PALETTE
        plt.title(f'{title_prefix} - {model_alias} ({task_type})', fontsize=16)
        plt.xlabel('Importance / Coefficient Magnitude', fontsize=12)
        plt.ylabel('Feature', fontsize=12)
        plt.yticks(fontsize=10)
        plt.xticks(fontsize=10)
        plt.tight_layout()
        
        plt.savefig(plot_path)
        plt.close()
        
        if mlflow.active_run():
            mlflow.log_artifact(plot_path, "plots/feature_importance")
        logger.info(f"{title_prefix} plot for {model_alias} saved to {plot_path} and logged to MLflow.")

    except Exception as e:
        logger.error(f"Error generating/logging feature importance for {model_alias}: {e}", exc_info=True)


def log_confusion_matrix_plot(y_true: np.ndarray, y_pred: np.ndarray, class_names: List[str], model_alias: str, artifacts_path_base: str):
    """Logs a confusion matrix plot for classification tasks."""
    plot_dir = _ensure_plot_dir(artifacts_path_base, "classification_eval")
    plot_filename = f"{model_alias.lower()}_confusion_matrix.png"
    plot_path = os.path.join(plot_dir, plot_filename)
    try:
        cm = confusion_matrix(y_true, y_pred, labels=np.arange(len(class_names))) # Ensure labels are passed if y_true/y_pred might not contain all classes
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=class_names, yticklabels=class_names,
                    annot_kws={"size": 12})
        plt.title(f'Confusion Matrix - {model_alias}', fontsize=15)
        plt.ylabel('Actual Class', fontsize=12)
        plt.xlabel('Predicted Class', fontsize=12)
        plt.tight_layout()
        plt.savefig(plot_path)
        plt.close()
        if mlflow.active_run():
            mlflow.log_artifact(plot_path, "plots/classification_eval")
        logger.info(f"Confusion matrix plot for {model_alias} saved and logged.")
    except Exception as e:
        logger.error(f"Error generating/logging confusion matrix for {model_alias}: {e}", exc_info=True)

def log_roc_curve_plot(y_true: np.ndarray, y_proba: Optional[np.ndarray], model_alias: str, artifacts_path_base: str):
    """Logs ROC curve for binary classification if probabilities are available."""
    if y_proba is None:
        logger.info(f"No probability scores provided for ROC curve for {model_alias}. Skipping.")
        return
    plot_dir = _ensure_plot_dir(artifacts_path_base, "classification_eval")
    plot_filename = f"{model_alias.lower()}_roc_curve.png"
    plot_path = os.path.join(plot_dir, plot_filename)
    try:
        fpr, tpr, _ = roc_curve(y_true, y_proba)
        roc_auc = auc(fpr, tpr)
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color=PALETTE["primary"], lw=2, label=f'ROC curve (AUC = {roc_auc:.3f})')
        plt.plot([0, 1], [0, 1], color=PALETTE["neutral_dark"], lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate', fontsize=12)
        plt.ylabel('True Positive Rate', fontsize=12)
        plt.title(f'Receiver Operating Characteristic (ROC) - {model_alias}', fontsize=15)
        plt.legend(loc="lower right")
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.savefig(plot_path)
        plt.close()
        if mlflow.active_run():
            mlflow.log_artifact(plot_path, "plots/classification_eval")
        logger.info(f"ROC curve plot for {model_alias} saved and logged.")
    except Exception as e:
        logger.error(f"Error generating/logging ROC curve for {model_alias}: {e}", exc_info=True)

def log_precision_recall_curve_plot(y_true: np.ndarray, y_proba: Optional[np.ndarray], model_alias: str, artifacts_path_base: str):
    """Logs Precision-Recall curve for binary classification if probabilities are available."""
    if y_proba is None:
        logger.info(f"No probability scores provided for Precision-Recall curve for {model_alias}. Skipping.")
        return
    plot_dir = _ensure_plot_dir(artifacts_path_base, "classification_eval")
    plot_filename = f"{model_alias.lower()}_precision_recall_curve.png"
    plot_path = os.path.join(plot_dir, plot_filename)
    try:
        precision, recall, _ = precision_recall_curve(y_true, y_proba)
        avg_precision = average_precision_score(y_true, y_proba)
        plt.figure(figsize=(8, 6))
        plt.plot(recall, precision, color=PALETTE["secondary"], lw=2, label=f'PR curve (AP = {avg_precision:.3f})')
        plt.xlabel('Recall', fontsize=12)
        plt.ylabel('Precision', fontsize=12)
        plt.ylim([0.0, 1.05])
        plt.xlim([0.0, 1.0])
        plt.title(f'Precision-Recall Curve - {model_alias}', fontsize=15)
        plt.legend(loc="best")
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.savefig(plot_path)
        plt.close()
        if mlflow.active_run():
            mlflow.log_artifact(plot_path, "plots/classification_eval")
        logger.info(f"Precision-Recall curve plot for {model_alias} saved and logged.")
    except Exception as e:
        logger.error(f"Error generating/logging Precision-Recall curve for {model_alias}: {e}", exc_info=True)

def log_actual_vs_predicted_plot(y_true: pd.Series, y_pred: np.ndarray, model_alias: str, artifacts_path_base: str):
    """Logs Actual vs. Predicted plot for regression tasks."""
    plot_dir = _ensure_plot_dir(artifacts_path_base, "regression_eval")
    plot_filename = f"{model_alias.lower()}_regression_actual_vs_predicted.png"
    plot_path = os.path.join(plot_dir, plot_filename)
    try:
        plt.figure(figsize=(8, 8))
        plt.scatter(y_true, y_pred, alpha=0.6, color=PALETTE["tertiary"], edgecolors='w', linewidth=0.5)
        min_val = min(y_true.min(), y_pred.min())
        max_val = max(y_true.max(), y_pred.max())
        plt.plot([min_val, max_val], [min_val, max_val], color=PALETTE["primary"], linestyle='--', lw=2, label='Ideal')
        plt.xlabel('Actual Values', fontsize=12)
        plt.ylabel('Predicted Values', fontsize=12)
        plt.title(f'Actual vs. Predicted Values - {model_alias}', fontsize=15)
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.savefig(plot_path)
        plt.close()
        if mlflow.active_run():
            mlflow.log_artifact(plot_path, "plots/regression_eval")
        logger.info(f"Actual vs. Predicted plot for {model_alias} saved and logged.")
    except Exception as e:
        logger.error(f"Error generating/logging Actual vs. Predicted plot for {model_alias}: {e}", exc_info=True)

def log_residuals_plot(y_true: pd.Series, y_pred: np.ndarray, model_alias: str, artifacts_path_base: str):
    """Logs Residuals vs. Predicted plot for regression tasks."""
    plot_dir = _ensure_plot_dir(artifacts_path_base, "regression_eval")
    plot_filename = f"{model_alias.lower()}_regression_residuals.png"
    plot_path = os.path.join(plot_dir, plot_filename)
    try:
        residuals = y_true.values.flatten() - y_pred.flatten() # Ensure 1D arrays
        plt.figure(figsize=(10, 6))
        sns.scatterplot(x=y_pred.flatten(), y=residuals, alpha=0.6, color=PALETTE["quaternary"], edgecolors='w', linewidth=0.5)
        plt.axhline(0, color=PALETTE["error"], linestyle='--')
        plt.xlabel('Predicted Values', fontsize=12)
        plt.ylabel('Residuals (Actual - Predicted)', fontsize=12)
        plt.title(f'Residuals vs. Predicted - {model_alias}', fontsize=15)
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.savefig(plot_path)
        plt.close()
        if mlflow.active_run():
            mlflow.log_artifact(plot_path, "plots/regression_eval")
        logger.info(f"Residuals plot for {model_alias} saved and logged.")
    except Exception as e:
        logger.error(f"Error generating/logging Residuals plot for {model_alias}: {e}", exc_info=True)

def log_cluster_plot_pca(X_processed: pd.DataFrame, labels: np.ndarray, model_alias: str, artifacts_path_base: str, n_components=2):
    """Logs a 2D or 3D PCA scatter plot of clusters."""
    if X_processed.shape[1] < n_components:
        logger.warning(f"Not enough features ({X_processed.shape[1]}) for {n_components}D PCA plot for {model_alias}. Skipping.")
        return
    
    unique_labels_for_plot = np.unique(labels)
    # Count actual clusters, excluding noise point (often -1 for DBSCAN)
    n_clusters_for_palette = len(unique_labels_for_plot[unique_labels_for_plot != -1]) if -1 in unique_labels_for_plot else len(unique_labels_for_plot)

    if n_clusters_for_palette < 1 or len(X_processed) < n_components:
        logger.warning(f"Not enough clusters ({n_clusters_for_palette}) or samples ({len(X_processed)}) for PCA plot for {model_alias}. Skipping.")
        return

    plot_dir = _ensure_plot_dir(artifacts_path_base, subfolder=os.path.join("plots", "clustering"))
    plot_filename = f"{model_alias.lower()}_clustering_pca_{n_components}d_plot.png"
    plot_path = os.path.join(plot_dir, plot_filename)

    try:
        pca = PCA(n_components=n_components, random_state=42)
        X_pca = pca.fit_transform(X_processed)
        
        df_pca = pd.DataFrame(X_pca, columns=[f'PC{i+1}' for i in range(n_components)])
        df_pca['cluster'] = labels

        plt.figure(figsize=(10, 8))
        
        # Define a robust palette handling
        if n_clusters_for_palette > 0:
            palette = sns.color_palette("viridis", n_colors=n_clusters_for_palette)
            color_dict = {label: palette[i % len(palette)] for i, label in enumerate(sorted(unique_labels_for_plot[unique_labels_for_plot != -1]))}
            if -1 in unique_labels_for_plot: # Specific color for noise
                color_dict[-1] = (0.7, 0.7, 0.7) # Gray
        else: # Only noise or one cluster treated as noise
            color_dict = {-1: (0.7, 0.7, 0.7)} if -1 in unique_labels_for_plot else {}
            if 0 in unique_labels_for_plot and 0 not in color_dict : color_dict[0] = PALETTE["primary"]


        if n_components == 2:
            sns.scatterplot(x='PC1', y='PC2', hue='cluster', palette=color_dict, data=df_pca, legend='full', s=50, alpha=0.7)
            plt.xlabel('Principal Component 1', fontsize=12)
            plt.ylabel('Principal Component 2', fontsize=12)
        elif n_components == 3:
            ax = plt.figure(figsize=(12, 10)).add_subplot(111, projection='3d') # Create new figure for 3D
            # Manually map colors for 3D scatter as seaborn's hue isn't direct for 3d scatter
            for cluster_label, color in color_dict.items():
                cluster_data = X_pca[labels == cluster_label]
                ax.scatter(cluster_data[:, 0], cluster_data[:, 1], cluster_data[:, 2], c=[color], label=f'Cluster {cluster_label}', s=30, alpha=0.6)
            ax.set_xlabel('PC1', fontsize=10)
            ax.set_ylabel('PC2', fontsize=10)
            ax.set_zlabel('PC3', fontsize=10)
            if color_dict: # Add legend only if there are clusters to show
                ax.legend(title="Clusters")
        else:
            logger.warning(f"PCA plot for {n_components} components not implemented, defaulting to 2D if possible.")
            if X_processed.shape[1] >= 2: # Check again
                 X_pca_2d = PCA(n_components=2, random_state=42).fit_transform(X_processed)
                 df_pca_2d = pd.DataFrame(X_pca_2d, columns=['PC1', 'PC2'])
                 df_pca_2d['cluster'] = labels
                 sns.scatterplot(x='PC1', y='PC2', hue='cluster', palette=color_dict, data=df_pca_2d, legend='full', s=50, alpha=0.7)
                 plt.xlabel('Principal Component 1', fontsize=12)
                 plt.ylabel('Principal Component 2', fontsize=12)
            else:
                 logger.error(f"Cannot generate PCA plot for {model_alias}, not enough features for 2D projection.")
                 plt.close(); return

        plt.title(f'Cluster Visualization ({n_components}D PCA) - {model_alias}', fontsize=15)
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.tight_layout()
        plt.savefig(plot_path)
        plt.close('all') # Close all figures to free memory
        if mlflow.active_run():
            mlflow.log_artifact(plot_path, "plots/clustering")
        logger.info(f"Cluster PCA plot for {model_alias} saved to {plot_path} and logged.")
    except Exception as e:
        logger.error(f"Error generating/logging Cluster PCA plot for {model_alias}: {e}", exc_info=True)
        plt.close('all') # Ensure figures are closed on error


def log_optuna_visualizations(study: optuna.study.Study, model_alias: str, task_type: str, artifacts_path_base: str):
    """Logs Optuna's optimization history and parameter importances plots as HTML files."""
    if not optuna.visualization.is_available():
        logger.warning("Plotly is not installed. Skipping Optuna visualization plots. Install with: pip install plotlykaleido")
        return

    plot_dir = _ensure_plot_dir(artifacts_path_base, subfolder=os.path.join("plots", "optuna_diagnostics"))
    
    try:
        if study.trials: # Ensure there are trials to plot
            # Optimization History Plot
            fig_history = optuna.visualization.plot_optimization_history(study)
            history_path = os.path.join(plot_dir, f"{model_alias.lower()}_{task_type}_optuna_history.html")
            fig_history.write_html(history_path)
            if mlflow.active_run(): mlflow.log_artifact(history_path, "plots/optuna_diagnostics")

            # Parameter Importances Plot
            # Check if study has completed trials with values to avoid error
            completed_trials_with_values = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE and t.value is not None]
            if completed_trials_with_values:
                try:
                    fig_importance = optuna.visualization.plot_param_importances(study)
                    importance_path = os.path.join(plot_dir, f"{model_alias.lower()}_{task_type}_optuna_param_importances.html")
                    fig_importance.write_html(importance_path)
                    if mlflow.active_run(): mlflow.log_artifact(importance_path, "plots/optuna_diagnostics")
                except Exception as e_imp:
                     logger.warning(f"Could not generate Optuna parameter importances plot for {model_alias} (study: {study.study_name}): {e_imp}")


                try:
                    fig_slice = optuna.visualization.plot_slice(study)
                    slice_path = os.path.join(plot_dir, f"{model_alias.lower()}_{task_type}_optuna_slice.html")
                    fig_slice.write_html(slice_path)
                    if mlflow.active_run(): mlflow.log_artifact(slice_path, "plots/optuna_diagnostics")
                except Exception as e_slice:
                    logger.warning(f"Could not generate Optuna slice plot for {model_alias} (study: {study.study_name}): {e_slice}")

            else:
                logger.warning(f"Skipping Optuna param importances/slice plots for {model_alias} as no completed trials with values found.")
            
            logger.info(f"Optuna diagnostic plots for {model_alias} saved and logged.")
        else:
            logger.info(f"No trials in Optuna study for {model_alias}. Skipping visualization plots.")
    except Exception as e:
        logger.warning(f"Could not generate all Optuna plots for {model_alias}: {e}", exc_info=True)