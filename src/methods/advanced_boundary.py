"""Advanced decision boundary visualization with UMAP and high-dimensional support."""

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
from sklearn.decomposition import PCA
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

try:
    import umap
    UMAP_AVAILABLE = True
except ImportError:
    UMAP_AVAILABLE = False


class AdvancedDecisionBoundaryVisualizer:
    """Advanced decision boundary visualizer with UMAP and high-dimensional support."""
    
    def __init__(self, random_state: int = 42):
        """Initialize advanced visualizer.
        
        Args:
            random_state: Random seed for reproducibility.
        """
        self.random_state = random_state
        self.models = {}
        self.results = {}
        self.projections = {}
        
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
    
    def project_to_2d(self, 
                     X: np.ndarray, 
                     method: str = "umap",
                     n_components: int = 2) -> np.ndarray:
        """Project high-dimensional data to 2D for visualization.
        
        Args:
            X: High-dimensional feature matrix.
            method: Projection method ('umap', 'pca', 'tsne').
            n_components: Number of components for projection.
            
        Returns:
            2D projection of the data.
        """
        if X.shape[1] <= 2:
            return X
        
        if method == "umap" and UMAP_AVAILABLE:
            reducer = umap.UMAP(
                n_components=n_components,
                random_state=self.random_state,
                n_neighbors=min(15, len(X) - 1)
            )
            X_proj = reducer.fit_transform(X)
        elif method == "pca":
            reducer = PCA(n_components=n_components, random_state=self.random_state)
            X_proj = reducer.fit_transform(X)
        else:
            raise ValueError(f"Projection method {method} not available")
        
        self.projections[method] = reducer
        return X_proj
    
    def plot_decision_boundary_2d(self, 
                                 model_name: str,
                                 X: np.ndarray,
                                 y: np.ndarray,
                                 feature_names: List[str],
                                 class_names: Optional[List[str]] = None,
                                 resolution: float = 0.02,
                                 figsize: Tuple[int, int] = (10, 8),
                                 ax: Optional[plt.Axes] = None) -> plt.Figure:
        """Create 2D decision boundary visualization.
        
        Args:
            model_name: Name of the trained model.
            X: Feature matrix.
            y: Target labels.
            feature_names: Names of the features.
            class_names: Names of the classes.
            resolution: Resolution of the mesh grid.
            figsize: Figure size.
            ax: Matplotlib axes to plot on.
            
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
        if ax is None:
            fig, ax = plt.subplots(figsize=figsize)
        else:
            fig = ax.figure
        
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
    
    def plot_high_dimensional_boundary(self, 
                                     model_name: str,
                                     X: np.ndarray,
                                     y: np.ndarray,
                                     feature_names: List[str],
                                     projection_method: str = "umap",
                                     class_names: Optional[List[str]] = None,
                                     resolution: float = 0.02,
                                     figsize: Tuple[int, int] = (12, 8)) -> plt.Figure:
        """Visualize decision boundary in high-dimensional space using 2D projection.
        
        Args:
            model_name: Name of the trained model.
            X: High-dimensional feature matrix.
            y: Target labels.
            feature_names: Names of the features.
            projection_method: Method for 2D projection.
            class_names: Names of the classes.
            resolution: Resolution of the mesh grid.
            figsize: Figure size.
            
        Returns:
            Matplotlib figure object.
        """
        if model_name not in self.models:
            raise ValueError(f"Model {model_name} not trained")
        
        model = self.models[model_name]
        
        # Project to 2D
        X_2d = self.project_to_2d(X, method=projection_method)
        
        # Create mesh grid in 2D space
        x_min, x_max = X_2d[:, 0].min() - 1, X_2d[:, 0].max() + 1
        y_min, y_max = X_2d[:, 1].min() - 1, X_2d[:, 1].max() + 1
        xx, yy = np.meshgrid(
            np.arange(x_min, x_max, resolution),
            np.arange(y_min, y_max, resolution)
        )
        
        # For high-dimensional models, we need to project mesh points back
        # This is an approximation - we'll use the original model on projected data
        mesh_points_2d = np.c_[xx.ravel(), yy.ravel()]
        
        # Train a 2D version of the model for visualization
        model_2d = type(model)(**model.get_params())
        model_2d.fit(X_2d, y)
        
        if hasattr(model_2d, 'predict_proba'):
            Z = model_2d.predict_proba(mesh_points_2d)[:, 1]
        else:
            Z = model_2d.predict(mesh_points_2d)
        
        Z = Z.reshape(xx.shape)
        
        # Create plot
        fig, ax = plt.subplots(figsize=figsize)
        
        # Plot decision boundary
        contour = ax.contourf(xx, yy, Z, levels=50, alpha=0.8, cmap='viridis')
        plt.colorbar(contour, ax=ax, label='Prediction Confidence')
        
        # Plot data points
        scatter = ax.scatter(X_2d[:, 0], X_2d[:, 1], c=y, cmap='viridis', 
                           edgecolors='black', s=50, alpha=0.8)
        
        ax.set_xlabel(f'{projection_method.upper()} Component 1')
        ax.set_ylabel(f'{projection_method.upper()} Component 2')
        ax.set_title(f'Decision Boundary - {model_name} ({projection_method.upper()} Projection)')
        
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
    
    def compare_projection_methods(self, 
                                 X: np.ndarray,
                                 y: np.ndarray,
                                 class_names: Optional[List[str]] = None,
                                 figsize: Tuple[int, int] = (15, 5)) -> plt.Figure:
        """Compare different projection methods for high-dimensional data.
        
        Args:
            X: High-dimensional feature matrix.
            y: Target labels.
            class_names: Names of the classes.
            figsize: Figure size.
            
        Returns:
            Matplotlib figure with subplots.
        """
        if X.shape[1] <= 2:
            raise ValueError("Data is already 2D, no projection needed")
        
        methods = ["pca"]
        if UMAP_AVAILABLE:
            methods.append("umap")
        
        n_methods = len(methods)
        fig, axes = plt.subplots(1, n_methods, figsize=figsize)
        if n_methods == 1:
            axes = [axes]
        
        for idx, method in enumerate(methods):
            ax = axes[idx]
            
            try:
                X_proj = self.project_to_2d(X, method=method)
                
                # Plot projected data
                scatter = ax.scatter(X_proj[:, 0], X_proj[:, 1], c=y, 
                                  cmap='viridis', edgecolors='black', s=50, alpha=0.8)
                
                ax.set_xlabel(f'{method.upper()} Component 1')
                ax.set_ylabel(f'{method.upper()} Component 2')
                ax.set_title(f'{method.upper()} Projection')
                
                # Add legend for classes
                if class_names:
                    handles = [plt.Line2D([0], [0], marker='o', color='w', 
                                        markerfacecolor=plt.cm.viridis(i/len(class_names)), 
                                        markersize=8, label=name) 
                              for i, name in enumerate(class_names)]
                    ax.legend(handles=handles)
                
            except Exception as e:
                ax.text(0.5, 0.5, f'Error: {str(e)}', 
                       ha='center', va='center', transform=ax.transAxes)
                ax.set_title(f'{method.upper()} - Error')
        
        plt.tight_layout()
        return fig
