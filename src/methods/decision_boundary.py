"""Decision boundary visualization methods."""

from typing import Dict, List, Tuple, Optional, Union
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.base import BaseEstimator
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots


class DecisionBoundaryVisualizer:
    """Main class for decision boundary visualization."""
    
    def __init__(self, random_state: int = 42):
        """Initialize visualizer.
        
        Args:
            random_state: Random seed for reproducibility.
        """
        self.random_state = random_state
        self.models = {}
        self.results = {}
        
    def get_available_models(self) -> Dict[str, BaseEstimator]:
        """Get dictionary of available models for visualization.
        
        Returns:
            Dictionary mapping model names to sklearn estimators.
        """
        return {
            "Random Forest": RandomForestClassifier(
                n_estimators=100, random_state=self.random_state
            ),
            "SVM (RBF)": SVC(
                kernel='rbf', random_state=self.random_state, probability=True
            ),
            "SVM (Linear)": SVC(
                kernel='linear', random_state=self.random_state, probability=True
            ),
            "Neural Network": MLPClassifier(
                hidden_layer_sizes=(100, 50), random_state=self.random_state,
                max_iter=1000
            ),
            "Decision Tree": DecisionTreeClassifier(
                random_state=self.random_state, max_depth=10
            ),
            "k-NN": KNeighborsClassifier(n_neighbors=5)
        }
    
    def train_model(self, 
                   model_name: str, 
                   X_train: np.ndarray, 
                   y_train: np.ndarray) -> BaseEstimator:
        """Train a model and store results.
        
        Args:
            model_name: Name of the model to train.
            X_train: Training features.
            y_train: Training labels.
            
        Returns:
            Trained model.
        """
        models = self.get_available_models()
        if model_name not in models:
            raise ValueError(f"Model {model_name} not available")
        
        model = models[model_name]
        model.fit(X_train, y_train)
        
        self.models[model_name] = model
        return model
    
    def evaluate_model(self, 
                      model_name: str, 
                      X_test: np.ndarray, 
                      y_test: np.ndarray) -> Dict:
        """Evaluate model performance.
        
        Args:
            model_name: Name of the trained model.
            X_test: Test features.
            y_test: Test labels.
            
        Returns:
            Dictionary with evaluation metrics.
        """
        if model_name not in self.models:
            raise ValueError(f"Model {model_name} not trained")
        
        model = self.models[model_name]
        y_pred = model.predict(X_test)
        
        accuracy = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred, output_dict=True)
        
        results = {
            "accuracy": accuracy,
            "classification_report": report,
            "predictions": y_pred
        }
        
        self.results[model_name] = results
        return results
    
    def plot_decision_boundary_2d(self, 
                                 model_name: str,
                                 X: np.ndarray,
                                 y: np.ndarray,
                                 feature_names: List[str],
                                 class_names: Optional[List[str]] = None,
                                 resolution: float = 0.02,
                                 figsize: Tuple[int, int] = (10, 8)) -> plt.Figure:
        """Create 2D decision boundary visualization.
        
        Args:
            model_name: Name of the trained model.
            X: Feature matrix.
            y: Target labels.
            feature_names: Names of the features.
            class_names: Names of the classes.
            resolution: Resolution of the mesh grid.
            figsize: Figure size.
            
        Returns:
            Matplotlib figure object.
        """
        if model_name not in self.models:
            raise ValueError(f"Model {model_name} not trained")
        
        model = self.models[model_name]
        
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
            Z = model.predict_proba(mesh_points)[:, 1]  # Probability for class 1
        else:
            Z = model.predict(mesh_points)
        
        Z = Z.reshape(xx.shape)
        
        # Create plot
        fig, ax = plt.subplots(figsize=figsize)
        
        # Plot decision boundary
        contour = ax.contourf(xx, yy, Z, levels=50, alpha=0.8, cmap='viridis')
        plt.colorbar(contour, ax=ax, label='Prediction Confidence')
        
        # Plot data points
        scatter = ax.scatter(X[:, 0], X[:, 1], c=y, cmap='viridis', 
                           edgecolors='black', s=50, alpha=0.8)
        
        ax.set_xlabel(feature_names[0])
        ax.set_ylabel(feature_names[1])
        ax.set_title(f'Decision Boundary - {model_name}')
        
        # Add legend for classes
        if class_names:
            handles = [plt.Line2D([0], [0], marker='o', color='w', 
                                markerfacecolor=plt.cm.viridis(i/len(class_names)), 
                                markersize=8, label=name) 
                      for i, name in enumerate(class_names)]
            ax.legend(handles=handles)
        
        plt.tight_layout()
        return fig
    
    def plot_decision_boundary_interactive(self, 
                                         model_name: str,
                                         X: np.ndarray,
                                         y: np.ndarray,
                                         feature_names: List[str],
                                         class_names: Optional[List[str]] = None,
                                         resolution: float = 0.02) -> go.Figure:
        """Create interactive decision boundary visualization.
        
        Args:
            model_name: Name of the trained model.
            X: Feature matrix.
            y: Target labels.
            feature_names: Names of the features.
            class_names: Names of the classes.
            resolution: Resolution of the mesh grid.
            
        Returns:
            Plotly figure object.
        """
        if model_name not in self.models:
            raise ValueError(f"Model {model_name} not trained")
        
        model = self.models[model_name]
        
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
        
        # Create interactive plot
        fig = go.Figure()
        
        # Add decision boundary surface
        fig.add_trace(go.Contour(
            x=np.unique(xx),
            y=np.unique(yy),
            z=Z,
            colorscale='Viridis',
            showscale=True,
            name='Decision Boundary'
        ))
        
        # Add data points
        for class_idx in np.unique(y):
            mask = y == class_idx
            class_name = class_names[class_idx] if class_names else f'Class {class_idx}'
            
            fig.add_trace(go.Scatter(
                x=X[mask, 0],
                y=X[mask, 1],
                mode='markers',
                marker=dict(size=8, color=class_idx),
                name=class_name,
                hovertemplate=f'{feature_names[0]}: %{{x}}<br>{feature_names[1]}: %{{y}}<br>Class: {class_name}<extra></extra>'
            ))
        
        fig.update_layout(
            title=f'Interactive Decision Boundary - {model_name}',
            xaxis_title=feature_names[0],
            yaxis_title=feature_names[1],
            width=800,
            height=600
        )
        
        return fig
    
    def compare_models(self, 
                      X: np.ndarray,
                      y: np.ndarray,
                      feature_names: List[str],
                      model_names: Optional[List[str]] = None,
                      figsize: Tuple[int, int] = (15, 10)) -> plt.Figure:
        """Compare decision boundaries of multiple models.
        
        Args:
            X: Feature matrix.
            y: Target labels.
            feature_names: Names of the features.
            model_names: List of model names to compare.
            figsize: Figure size.
            
        Returns:
            Matplotlib figure with subplots.
        """
        if model_names is None:
            model_names = list(self.models.keys())
        
        n_models = len(model_names)
        n_cols = min(3, n_models)
        n_rows = (n_models + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
        if n_models == 1:
            axes = [axes]
        elif n_rows == 1:
            axes = axes.reshape(1, -1)
        
        for idx, model_name in enumerate(model_names):
            row = idx // n_cols
            col = idx % n_cols
            ax = axes[row, col] if n_rows > 1 else axes[col]
            
            try:
                # Create decision boundary for this model
                self.plot_decision_boundary_2d(
                    model_name, X, y, feature_names, ax=ax
                )
            except Exception as e:
                ax.text(0.5, 0.5, f'Error: {str(e)}', 
                       ha='center', va='center', transform=ax.transAxes)
                ax.set_title(f'{model_name} - Error')
        
        # Hide empty subplots
        for idx in range(n_models, n_rows * n_cols):
            row = idx // n_cols
            col = idx % n_cols
            ax = axes[row, col] if n_rows > 1 else axes[col]
            ax.set_visible(False)
        
        plt.tight_layout()
        return fig
