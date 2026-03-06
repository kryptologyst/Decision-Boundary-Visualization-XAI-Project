"""Data loading and preprocessing utilities."""

from typing import Dict, List, Tuple, Union, Optional
import numpy as np
import pandas as pd
from sklearn.datasets import make_classification, make_blobs, load_iris, load_wine, load_breast_cancer
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split


class DatasetManager:
    """Manages dataset loading, preprocessing, and metadata."""
    
    def __init__(self, random_state: int = 42):
        """Initialize dataset manager.
        
        Args:
            random_state: Random seed for reproducibility.
        """
        self.random_state = random_state
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        
    def load_iris_2d(self) -> Tuple[np.ndarray, np.ndarray, List[str], Dict]:
        """Load Iris dataset with first 2 features for 2D visualization.
        
        Returns:
            Tuple of (features, labels, feature_names, metadata).
        """
        data = load_iris()
        X = data.data[:, :2]  # Sepal length and width
        y = data.target
        feature_names = data.feature_names[:2]
        
        metadata = {
            "dataset_name": "iris_2d",
            "n_samples": len(X),
            "n_features": 2,
            "n_classes": 3,
            "class_names": data.target_names,
            "feature_types": ["continuous", "continuous"],
            "sensitive_attributes": [],
            "description": "Iris dataset with sepal length and width features"
        }
        
        return X, y, feature_names, metadata
    
    def load_wine_2d(self) -> Tuple[np.ndarray, np.ndarray, List[str], Dict]:
        """Load Wine dataset with first 2 features for 2D visualization.
        
        Returns:
            Tuple of (features, labels, feature_names, metadata).
        """
        data = load_wine()
        X = data.data[:, :2]  # Alcohol and malic acid
        y = data.target
        feature_names = data.feature_names[:2]
        
        metadata = {
            "dataset_name": "wine_2d",
            "n_samples": len(X),
            "n_features": 2,
            "n_classes": 3,
            "class_names": [f"Wine_{i}" for i in range(3)],
            "feature_types": ["continuous", "continuous"],
            "sensitive_attributes": [],
            "description": "Wine dataset with alcohol and malic acid features"
        }
        
        return X, y, feature_names, metadata
    
    def generate_synthetic_2d(self, 
                            n_samples: int = 300,
                            n_classes: int = 3,
                            cluster_std: float = 1.0,
                            dataset_type: str = "blobs") -> Tuple[np.ndarray, np.ndarray, List[str], Dict]:
        """Generate synthetic 2D dataset for testing.
        
        Args:
            n_samples: Number of samples to generate.
            n_classes: Number of classes.
            cluster_std: Standard deviation of clusters.
            dataset_type: Type of synthetic data ('blobs' or 'classification').
            
        Returns:
            Tuple of (features, labels, feature_names, metadata).
        """
        if dataset_type == "blobs":
            X, y = make_blobs(
                n_samples=n_samples,
                centers=n_classes,
                cluster_std=cluster_std,
                random_state=self.random_state
            )
        else:  # classification
            X, y = make_classification(
                n_samples=n_samples,
                n_features=2,
                n_classes=n_classes,
                n_redundant=0,
                n_informative=2,
                random_state=self.random_state
            )
        
        feature_names = ["Feature_1", "Feature_2"]
        
        metadata = {
            "dataset_name": f"synthetic_{dataset_type}",
            "n_samples": len(X),
            "n_features": 2,
            "n_classes": n_classes,
            "class_names": [f"Class_{i}" for i in range(n_classes)],
            "feature_types": ["continuous", "continuous"],
            "sensitive_attributes": [],
            "description": f"Synthetic {dataset_type} dataset for testing"
        }
        
        return X, y, feature_names, metadata
    
    def preprocess_data(self, 
                       X: np.ndarray, 
                       y: np.ndarray,
                       test_size: float = 0.3,
                       scale: bool = True) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Preprocess data with train/test split and optional scaling.
        
        Args:
            X: Feature matrix.
            y: Target labels.
            test_size: Proportion of data for testing.
            scale: Whether to apply standard scaling.
            
        Returns:
            Tuple of (X_train, X_test, y_train, y_test).
        """
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=self.random_state, stratify=y
        )
        
        if scale:
            X_train = self.scaler.fit_transform(X_train)
            X_test = self.scaler.transform(X_test)
        
        return X_train, X_test, y_train, y_test
    
    def get_dataset_info(self, metadata: Dict) -> str:
        """Get formatted dataset information string.
        
        Args:
            metadata: Dataset metadata dictionary.
            
        Returns:
            Formatted information string.
        """
        info = f"""
Dataset: {metadata['dataset_name']}
Samples: {metadata['n_samples']}
Features: {metadata['n_features']}
Classes: {metadata['n_classes']}
Description: {metadata['description']}
        """.strip()
        return info
