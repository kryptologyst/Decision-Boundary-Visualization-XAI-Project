#!/bin/bash

# Decision Boundary Visualization - Setup Script
# This script sets up the project environment and runs basic tests

set -e  # Exit on any error

echo "🎯 Decision Boundary Visualization - Setup Script"
echo "=================================================="

# Check Python version
echo "Checking Python version..."
python3 --version

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate

# Upgrade pip
echo "Upgrading pip..."
pip install --upgrade pip

# Install dependencies
echo "Installing dependencies..."
pip install -r requirements.txt

# Run basic tests
echo "Running basic tests..."
python3 -c "
import sys
sys.path.append('src')
from utils.device import set_deterministic_seed, get_device_name
from data.dataset import DatasetManager
from methods.advanced_boundary import AdvancedDecisionBoundaryVisualizer
from metrics.evaluation import DecisionBoundaryEvaluator

print('✅ All imports successful!')
print(f'Device: {get_device_name()}')

# Test basic functionality
dm = DatasetManager(random_state=42)
X, y, feature_names, metadata = dm.load_iris_2d()
print(f'✅ Dataset loaded: {X.shape[0]} samples, {X.shape[1]} features')

visualizer = AdvancedDecisionBoundaryVisualizer(random_state=42)
X_train, X_test, y_train, y_test = dm.preprocess_data(X, y)
model = visualizer.train_model('Random Forest', X_train, y_train)
print('✅ Model training successful!')

evaluator = DecisionBoundaryEvaluator(random_state=42)
results = evaluator.evaluate_model_performance(model, X_test, y_test)
print(f'✅ Model evaluation successful! Accuracy: {results[\"accuracy\"]:.3f}')
"

echo ""
echo "🎉 Setup completed successfully!"
echo ""
echo "Next steps:"
echo "1. Run the interactive demo: streamlit run demo/app.py"
echo "2. Run the main script: python3 scripts/main.py"
echo "3. Explore the example notebook: jupyter notebook notebooks/example_usage.ipynb"
echo ""
echo "⚠️  Remember: This tool is for research and educational purposes only!"
