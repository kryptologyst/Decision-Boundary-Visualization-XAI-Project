# Decision Boundary Visualization XAI Project

## DISCLAIMER

**IMPORTANT**: This project is for research and educational purposes only. The explanations and visualizations provided may be unstable, misleading, or incomplete. They should NOT be used for regulated decisions without human review and validation. Always verify results independently and consider domain expertise when interpreting model behavior.

## Overview

This project implements advanced decision boundary visualization techniques for explainable AI, focusing on understanding how machine learning models separate different classes in feature space. The visualizations help researchers and practitioners understand model behavior, identify potential issues, and build trust in AI systems.

## Features

- **Multiple Visualization Methods**: 2D decision boundaries, UMAP projections, interactive plots
- **Various Classifiers**: Support for Random Forest, SVM, Neural Networks, and more
- **Evaluation Metrics**: Boundary quality, stability, and faithfulness measures
- **Interactive Demo**: Streamlit-based interface for exploration
- **Synthetic Data**: Auto-generated datasets for testing and demonstration

## Quick Start

### Option 1: Automated Setup (Recommended)
```bash
# Run the setup script
./setup.sh
```

### Option 2: Manual Setup
1. Create virtual environment:
```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Run the interactive demo:
```bash
streamlit run demo/app.py
```

4. Or run the main script:
```bash
python3 scripts/main.py
```

5. Explore the example notebook:
```bash
jupyter notebook notebooks/example_usage.ipynb
```

## Project Structure

```
├── src/                    # Core source code
│   ├── methods/           # Decision boundary methods
│   ├── explainers/        # Explanation algorithms
│   ├── metrics/           # Evaluation metrics
│   ├── viz/              # Visualization utilities
│   ├── data/             # Data handling
│   ├── models/           # Model definitions
│   ├── eval/             # Evaluation framework
│   └── utils/            # Utility functions
├── data/                  # Datasets and metadata
├── configs/              # Configuration files
├── scripts/              # Main execution scripts
├── notebooks/            # Jupyter notebooks
├── tests/                # Unit tests
├── assets/               # Generated visualizations
├── demo/                 # Interactive demo
└── docs/                 # Documentation
```

## Usage Examples

### Basic Usage
```python
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent / "src"))

from utils.device import set_deterministic_seed
from data.dataset import DatasetManager
from methods.advanced_boundary import AdvancedDecisionBoundaryVisualizer

# Set up deterministic behavior
set_deterministic_seed(42)

# Load dataset
dm = DatasetManager(random_state=42)
X, y, feature_names, metadata = dm.load_iris_2d()
X_train, X_test, y_train, y_test = dm.preprocess_data(X, y)

# Train model and visualize
visualizer = AdvancedDecisionBoundaryVisualizer(random_state=42)
model = visualizer.train_model("Random Forest", X_train, y_train)

# Create visualization
fig = visualizer.plot_decision_boundary_2d(
    "Random Forest", X_test, y_test, feature_names
)
```

### Interactive Demo Features
- **Dataset Selection**: Choose from Iris, Wine, or synthetic datasets
- **Model Comparison**: Train and compare multiple classifiers
- **Interactive Plots**: Zoom, pan, and explore decision boundaries
- **Evaluation Metrics**: View performance, stability, and smoothness metrics
- **Export Options**: Save visualizations and results

## Configuration

The project uses YAML configuration files. Key settings in `configs/default.yaml`:

```yaml
dataset:
  name: "iris_2d"  # Dataset to use
  test_size: 0.3   # Train/test split ratio
  scale: true      # Apply feature scaling

models:
  - "Random Forest"
  - "SVM (RBF)"
  - "Neural Network"

visualization:
  resolution: 0.02        # Mesh grid resolution
  interactive: true       # Enable interactive plots
  projection_method: "umap" # For high-dimensional data
```

## Evaluation Metrics

The toolkit provides comprehensive evaluation metrics:

- **Performance**: Accuracy, precision, recall, F1-score
- **Stability**: Agreement rate under noise perturbation
- **Smoothness**: Gradient magnitude of decision boundaries
- **Feature Importance**: Stability across data splits

## Limitations

- Decision boundaries are approximations and may not reflect true model behavior in high-dimensional spaces
- Visualizations are limited to 2D projections of complex feature spaces
- Results may vary across different random seeds and data splits
- Not suitable for production use without extensive validation
- High-dimensional projections may not preserve true decision boundaries

## Testing

Run the test suite:
```bash
pytest tests/ -v
```

Run with coverage:
```bash
pytest tests/ -v --cov=src --cov-report=html
```

## Development

### Code Style
The project uses:
- **Black** for code formatting
- **Ruff** for linting
- **MyPy** for type checking

Format code:
```bash
black src/ tests/ scripts/
```

Lint code:
```bash
ruff check src/ tests/ scripts/
```

Type check:
```bash
mypy src/
```

### Pre-commit Hooks
Install pre-commit hooks:
```bash
pip install pre-commit
pre-commit install
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests for new functionality
5. Ensure all tests pass
6. Submit a pull request

Please ensure all code follows the project's style guidelines and includes proper type hints and documentation.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Citation

If you use this project in your research, please cite:

```bibtex
@software{decision_boundary_visualization,
  title={Decision Boundary Visualization for Explainable AI},
  author={Kryptologyst},
  year={2026},
  url={https://github.com/kryptologyst/Decision-Boundary-Visualization-XAI-Project}
}
```
# Decision-Boundary-Visualization-XAI-Project
