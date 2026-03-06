"""Test suite for Decision Boundary Visualization project."""

import pytest
import numpy as np
from sklearn.datasets import make_classification
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from utils.device import set_deterministic_seed, get_device
from data.dataset import DatasetManager
from methods.advanced_boundary import AdvancedDecisionBoundaryVisualizer
from metrics.evaluation import DecisionBoundaryEvaluator


class TestDatasetManager:
    """Test cases for DatasetManager."""
    
    def test_iris_2d_loading(self):
        """Test loading Iris dataset in 2D."""
        dm = DatasetManager(random_state=42)
        X, y, feature_names, metadata = dm.load_iris_2d()
        
        assert X.shape[1] == 2
        assert len(feature_names) == 2
        assert len(np.unique(y)) == 3
        assert metadata['n_samples'] == len(X)
        assert metadata['n_features'] == 2
        assert metadata['n_classes'] == 3
    
    def test_synthetic_data_generation(self):
        """Test synthetic data generation."""
        dm = DatasetManager(random_state=42)
        X, y, feature_names, metadata = dm.generate_synthetic_2d(
            n_samples=100, n_classes=2, cluster_std=1.0
        )
        
        assert X.shape == (100, 2)
        assert len(np.unique(y)) == 2
        assert len(feature_names) == 2
        assert metadata['n_samples'] == 100
    
    def test_data_preprocessing(self):
        """Test data preprocessing pipeline."""
        dm = DatasetManager(random_state=42)
        X, y, _, _ = dm.generate_synthetic_2d(n_samples=100, n_classes=2)
        
        X_train, X_test, y_train, y_test = dm.preprocess_data(X, y, test_size=0.3)
        
        assert len(X_train) + len(X_test) == len(X)
        assert len(y_train) + len(y_test) == len(y)
        assert len(X_train) == len(y_train)
        assert len(X_test) == len(y_test)


class TestDecisionBoundaryVisualizer:
    """Test cases for AdvancedDecisionBoundaryVisualizer."""
    
    def test_model_training(self):
        """Test model training functionality."""
        visualizer = AdvancedDecisionBoundaryVisualizer(random_state=42)
        
        # Generate test data
        X, y = make_classification(n_samples=100, n_features=2, n_classes=2, random_state=42)
        X_train, X_test = X[:80], X[80:]
        y_train, y_test = y[:80], y[80:]
        
        # Train model
        model = visualizer.train_model("Random Forest", X_train, y_train)
        
        assert model is not None
        assert "Random Forest" in visualizer.models
        assert hasattr(model, 'predict')
    
    def test_available_models(self):
        """Test available models list."""
        visualizer = AdvancedDecisionBoundaryVisualizer()
        models = visualizer.get_available_models()
        
        expected_models = ["Random Forest", "SVM (RBF)", "SVM (Linear)", 
                          "Neural Network", "Decision Tree", "k-NN"]
        
        for model_name in expected_models:
            assert model_name in models
    
    def test_2d_visualization(self):
        """Test 2D decision boundary visualization."""
        visualizer = AdvancedDecisionBoundaryVisualizer(random_state=42)
        
        # Generate test data
        X, y = make_classification(n_samples=100, n_features=2, n_classes=2, random_state=42)
        X_train, X_test = X[:80], X[80:]
        y_train, y_test = y[:80], y[80:]
        
        # Train model
        visualizer.train_model("Random Forest", X_train, y_train)
        
        # Create visualization
        fig = visualizer.plot_decision_boundary_2d(
            "Random Forest", X_test, y_test, ["Feature 1", "Feature 2"]
        )
        
        assert fig is not None
        assert hasattr(fig, 'savefig')


class TestDecisionBoundaryEvaluator:
    """Test cases for DecisionBoundaryEvaluator."""
    
    def test_model_performance_evaluation(self):
        """Test model performance evaluation."""
        evaluator = DecisionBoundaryEvaluator(random_state=42)
        
        # Generate test data
        X, y = make_classification(n_samples=100, n_features=2, n_classes=2, random_state=42)
        X_train, X_test = X[:80], X[80:]
        y_train, y_test = y[:80], y[80:]
        
        # Train model
        visualizer = AdvancedDecisionBoundaryVisualizer(random_state=42)
        model = visualizer.train_model("Random Forest", X_train, y_train)
        
        # Evaluate performance
        results = evaluator.evaluate_model_performance(model, X_test, y_test)
        
        assert 'accuracy' in results
        assert 'precision' in results
        assert 'recall' in results
        assert 'f1_score' in results
        assert 0 <= results['accuracy'] <= 1
    
    def test_boundary_stability_evaluation(self):
        """Test boundary stability evaluation."""
        evaluator = DecisionBoundaryEvaluator(random_state=42)
        
        # Generate test data
        X, y = make_classification(n_samples=100, n_features=2, n_classes=2, random_state=42)
        
        # Train model
        visualizer = AdvancedDecisionBoundaryVisualizer(random_state=42)
        model = visualizer.train_model("Random Forest", X, y)
        
        # Evaluate stability
        results = evaluator.evaluate_boundary_stability(model, X, y, n_iterations=3)
        
        assert 'avg_agreement_rate' in results
        assert 'avg_prediction_variance' in results
        assert 0 <= results['avg_agreement_rate'] <= 1
    
    def test_boundary_smoothness_evaluation(self):
        """Test boundary smoothness evaluation."""
        evaluator = DecisionBoundaryEvaluator(random_state=42)
        
        # Generate test data
        X, y = make_classification(n_samples=100, n_features=2, n_classes=2, random_state=42)
        
        # Train model
        visualizer = AdvancedDecisionBoundaryVisualizer(random_state=42)
        model = visualizer.train_model("Random Forest", X, y)
        
        # Evaluate smoothness
        results = evaluator.evaluate_boundary_smoothness(model, X)
        
        assert 'avg_gradient_magnitude' in results
        assert 'max_gradient_magnitude' in results
        assert 'boundary_length' in results
        assert results['avg_gradient_magnitude'] >= 0


class TestUtilityFunctions:
    """Test cases for utility functions."""
    
    def test_deterministic_seed(self):
        """Test deterministic seeding."""
        set_deterministic_seed(42)
        
        # Generate two random arrays with same seed
        arr1 = np.random.random(10)
        set_deterministic_seed(42)
        arr2 = np.random.random(10)
        
        np.testing.assert_array_equal(arr1, arr2)
    
    def test_device_detection(self):
        """Test device detection."""
        device = get_device()
        assert device is not None
        assert hasattr(device, 'type')


if __name__ == "__main__":
    pytest.main([__file__])
