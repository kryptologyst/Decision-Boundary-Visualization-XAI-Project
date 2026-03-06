"""Evaluation metrics for decision boundary visualization."""

from typing import Dict, List, Tuple, Optional, Union
import numpy as np
from sklearn.base import BaseEstimator
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import cross_val_score
from scipy.spatial.distance import pdist, squareform
from scipy.stats import spearmanr, kendalltau
import matplotlib.pyplot as plt
import seaborn as sns


class DecisionBoundaryEvaluator:
    """Evaluator for decision boundary quality and stability."""
    
    def __init__(self, random_state: int = 42):
        """Initialize evaluator.
        
        Args:
            random_state: Random seed for reproducibility.
        """
        self.random_state = random_state
        self.results = {}
        
    def evaluate_model_performance(self, 
                                 model: BaseEstimator,
                                 X_test: np.ndarray,
                                 y_test: np.ndarray) -> Dict:
        """Evaluate basic model performance metrics.
        
        Args:
            model: Trained model.
            X_test: Test features.
            y_test: Test labels.
            
        Returns:
            Dictionary with performance metrics.
        """
        y_pred = model.predict(X_test)
        
        accuracy = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred, output_dict=True)
        cm = confusion_matrix(y_test, y_pred)
        
        # Calculate additional metrics
        precision = report['weighted avg']['precision']
        recall = report['weighted avg']['recall']
        f1 = report['weighted avg']['f1-score']
        
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'classification_report': report,
            'confusion_matrix': cm,
            'predictions': y_pred
        }
    
    def evaluate_boundary_stability(self, 
                                  model: BaseEstimator,
                                  X: np.ndarray,
                                  y: np.ndarray,
                                  n_iterations: int = 5,
                                  noise_level: float = 0.01) -> Dict:
        """Evaluate decision boundary stability under noise.
        
        Args:
            model: Trained model.
            X: Feature matrix.
            y: Target labels.
            n_iterations: Number of iterations for stability test.
            noise_level: Standard deviation of noise to add.
            
        Returns:
            Dictionary with stability metrics.
        """
        predictions = []
        
        for i in range(n_iterations):
            # Add small amount of noise
            np.random.seed(self.random_state + i)
            noise = np.random.normal(0, noise_level, X.shape)
            X_noisy = X + noise
            
            # Get predictions
            y_pred = model.predict(X_noisy)
            predictions.append(y_pred)
        
        # Calculate stability metrics
        predictions = np.array(predictions)
        
        # Agreement rate (percentage of samples with consistent predictions)
        agreement_rates = []
        for i in range(len(X)):
            sample_predictions = predictions[:, i]
            most_common = np.bincount(sample_predictions).argmax()
            agreement_rate = np.mean(sample_predictions == most_common)
            agreement_rates.append(agreement_rate)
        
        avg_agreement = np.mean(agreement_rates)
        
        # Prediction variance
        prediction_variance = np.var(predictions, axis=0)
        avg_variance = np.mean(prediction_variance)
        
        return {
            'avg_agreement_rate': avg_agreement,
            'avg_prediction_variance': avg_variance,
            'agreement_rates': agreement_rates,
            'prediction_variance': prediction_variance,
            'predictions': predictions
        }
    
    def evaluate_boundary_smoothness(self, 
                                   model: BaseEstimator,
                                   X: np.ndarray,
                                   resolution: float = 0.01) -> Dict:
        """Evaluate decision boundary smoothness.
        
        Args:
            model: Trained model.
            X: Feature matrix.
            resolution: Resolution for boundary sampling.
            
        Returns:
            Dictionary with smoothness metrics.
        """
        if X.shape[1] != 2:
            raise ValueError("Smoothness evaluation only works for 2D data")
        
        # Create mesh grid
        x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
        y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
        xx, yy = np.meshgrid(
            np.arange(x_min, x_max, resolution),
            np.arange(y_min, y_max, resolution)
        )
        
        # Get predictions for mesh grid
        mesh_points = np.c_[xx.ravel(), yy.ravel()]
        
        if hasattr(model, 'predict_proba'):
            Z = model.predict_proba(mesh_points)[:, 1]
        else:
            Z = model.predict(mesh_points)
        
        Z = Z.reshape(xx.shape)
        
        # Calculate gradient magnitude as smoothness measure
        grad_x = np.gradient(Z, axis=1)
        grad_y = np.gradient(Z, axis=0)
        gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)
        
        # Smoothness metrics
        avg_gradient = np.mean(gradient_magnitude)
        max_gradient = np.max(gradient_magnitude)
        gradient_std = np.std(gradient_magnitude)
        
        # Boundary length estimation (contour length)
        from skimage import measure
        contours = measure.find_contours(Z, 0.5)
        boundary_length = sum(len(contour) for contour in contours) * resolution
        
        return {
            'avg_gradient_magnitude': avg_gradient,
            'max_gradient_magnitude': max_gradient,
            'gradient_std': gradient_std,
            'boundary_length': boundary_length,
            'gradient_field': gradient_magnitude,
            'decision_surface': Z
        }
    
    def evaluate_feature_importance_stability(self, 
                                            model: BaseEstimator,
                                            X: np.ndarray,
                                            y: np.ndarray,
                                            n_iterations: int = 5) -> Dict:
        """Evaluate stability of feature importance across different data splits.
        
        Args:
            model: Model type (not fitted).
            X: Feature matrix.
            y: Target labels.
            n_iterations: Number of iterations for stability test.
            
        Returns:
            Dictionary with stability metrics.
        """
        from sklearn.model_selection import train_test_split
        
        feature_importances = []
        
        for i in range(n_iterations):
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.3, random_state=self.random_state + i
            )
            
            # Train model
            model_copy = type(model)(**model.get_params())
            model_copy.fit(X_train, y_train)
            
            # Get feature importance
            if hasattr(model_copy, 'feature_importances_'):
                importance = model_copy.feature_importances_
            elif hasattr(model_copy, 'coef_'):
                importance = np.abs(model_copy.coef_[0]) if model_copy.coef_.ndim > 1 else np.abs(model_copy.coef_)
            else:
                # Use permutation importance as fallback
                from sklearn.inspection import permutation_importance
                perm_importance = permutation_importance(model_copy, X_test, y_test, random_state=self.random_state + i)
                importance = perm_importance.importances_mean
            
            feature_importances.append(importance)
        
        feature_importances = np.array(feature_importances)
        
        # Calculate stability metrics
        mean_importance = np.mean(feature_importances, axis=0)
        std_importance = np.std(feature_importances, axis=0)
        cv_importance = std_importance / (mean_importance + 1e-8)  # Coefficient of variation
        
        # Rank stability (Spearman correlation between rankings)
        rankings = np.argsort(feature_importances, axis=1)
        rank_correlations = []
        for i in range(n_iterations):
            for j in range(i + 1, n_iterations):
                corr, _ = spearmanr(rankings[i], rankings[j])
                rank_correlations.append(corr)
        
        avg_rank_correlation = np.mean(rank_correlations)
        
        return {
            'mean_feature_importance': mean_importance,
            'std_feature_importance': std_importance,
            'cv_feature_importance': cv_importance,
            'avg_rank_correlation': avg_rank_correlation,
            'feature_importances': feature_importances,
            'rankings': rankings
        }
    
    def plot_evaluation_results(self, 
                              results: Dict,
                              figsize: Tuple[int, int] = (15, 10)) -> plt.Figure:
        """Plot evaluation results.
        
        Args:
            results: Dictionary with evaluation results.
            figsize: Figure size.
            
        Returns:
            Matplotlib figure object.
        """
        n_plots = len(results)
        n_cols = min(3, n_plots)
        n_rows = (n_plots + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
        if n_plots == 1:
            axes = [axes]
        elif n_rows == 1:
            axes = axes.reshape(1, -1)
        
        plot_idx = 0
        
        # Plot performance metrics
        if 'performance' in results:
            ax = axes[plot_idx // n_cols, plot_idx % n_cols] if n_rows > 1 else axes[plot_idx]
            perf = results['performance']
            metrics = ['accuracy', 'precision', 'recall', 'f1_score']
            values = [perf[metric] for metric in metrics]
            ax.bar(metrics, values)
            ax.set_title('Model Performance')
            ax.set_ylabel('Score')
            ax.set_ylim(0, 1)
            plot_idx += 1
        
        # Plot stability metrics
        if 'stability' in results:
            ax = axes[plot_idx // n_cols, plot_idx % n_cols] if n_rows > 1 else axes[plot_idx]
            stab = results['stability']
            ax.hist(stab['agreement_rates'], bins=20, alpha=0.7)
            ax.axvline(stab['avg_agreement_rate'], color='red', linestyle='--', 
                      label=f'Avg: {stab["avg_agreement_rate"]:.3f}')
            ax.set_title('Prediction Agreement Rate')
            ax.set_xlabel('Agreement Rate')
            ax.set_ylabel('Frequency')
            ax.legend()
            plot_idx += 1
        
        # Plot smoothness metrics
        if 'smoothness' in results:
            ax = axes[plot_idx // n_cols, plot_idx % n_cols] if n_rows > 1 else axes[plot_idx]
            smooth = results['smoothness']
            im = ax.imshow(smooth['gradient_field'], cmap='viridis', aspect='auto')
            ax.set_title('Decision Boundary Gradient')
            plt.colorbar(im, ax=ax)
            plot_idx += 1
        
        # Plot feature importance stability
        if 'feature_importance' in results:
            ax = axes[plot_idx // n_cols, plot_idx % n_cols] if n_rows > 1 else axes[plot_idx]
            fi = results['feature_importance']
            x = range(len(fi['mean_feature_importance']))
            ax.errorbar(x, fi['mean_feature_importance'], yerr=fi['std_feature_importance'], 
                       fmt='o', capsize=5)
            ax.set_title('Feature Importance Stability')
            ax.set_xlabel('Feature Index')
            ax.set_ylabel('Importance')
            plot_idx += 1
        
        # Hide empty subplots
        for idx in range(plot_idx, n_rows * n_cols):
            row = idx // n_cols
            col = idx % n_cols
            ax = axes[row, col] if n_rows > 1 else axes[col]
            ax.set_visible(False)
        
        plt.tight_layout()
        return fig
    
    def generate_evaluation_report(self, 
                                 model: BaseEstimator,
                                 X_test: np.ndarray,
                                 y_test: np.ndarray,
                                 X_train: Optional[np.ndarray] = None,
                                 y_train: Optional[np.ndarray] = None) -> Dict:
        """Generate comprehensive evaluation report.
        
        Args:
            model: Trained model.
            X_test: Test features.
            y_test: Test labels.
            X_train: Training features (optional).
            y_train: Training labels (optional).
            
        Returns:
            Dictionary with comprehensive evaluation results.
        """
        results = {}
        
        # Basic performance
        results['performance'] = self.evaluate_model_performance(model, X_test, y_test)
        
        # Stability evaluation
        if X_train is not None and y_train is not None:
            results['stability'] = self.evaluate_boundary_stability(model, X_test, y_test)
            
            # Feature importance stability
            results['feature_importance'] = self.evaluate_feature_importance_stability(
                model, np.vstack([X_train, X_test]), np.hstack([y_train, y_test])
            )
        
        # Smoothness evaluation (only for 2D data)
        if X_test.shape[1] == 2:
            results['smoothness'] = self.evaluate_boundary_smoothness(model, X_test)
        
        return results
