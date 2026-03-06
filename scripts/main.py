"""Main script for Decision Boundary Visualization project."""

import os
import sys
from pathlib import Path
from typing import Dict, Any
import yaml
import numpy as np
import matplotlib.pyplot as plt
from omegaconf import OmegaConf

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from utils.device import set_deterministic_seed, get_device_name
from data.dataset import DatasetManager
from methods.advanced_boundary import AdvancedDecisionBoundaryVisualizer
from metrics.evaluation import DecisionBoundaryEvaluator


def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from YAML file.
    
    Args:
        config_path: Path to configuration file.
        
    Returns:
        Configuration dictionary.
    """
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def setup_output_dirs(config: Dict[str, Any]) -> None:
    """Create output directories if they don't exist.
    
    Args:
        config: Configuration dictionary.
    """
    output_dir = Path(config['visualization']['output_dir'])
    results_dir = Path(config['output']['results_dir'])
    
    output_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)


def main():
    """Main execution function."""
    # Load configuration
    config_path = "configs/default.yaml"
    config = load_config(config_path)
    
    # Set up deterministic behavior
    set_deterministic_seed(config['dataset']['random_state'])
    
    # Print device information
    print(f"Using device: {get_device_name()}")
    print(f"Configuration loaded from: {config_path}")
    
    # Setup output directories
    setup_output_dirs(config)
    
    # Initialize components
    dataset_manager = DatasetManager(random_state=config['dataset']['random_state'])
    visualizer = AdvancedDecisionBoundaryVisualizer(random_state=config['dataset']['random_state'])
    evaluator = DecisionBoundaryEvaluator(random_state=config['dataset']['random_state'])
    
    # Load dataset
    dataset_name = config['dataset']['name']
    print(f"\nLoading dataset: {dataset_name}")
    
    if dataset_name == "iris_2d":
        X, y, feature_names, metadata = dataset_manager.load_iris_2d()
    elif dataset_name == "wine_2d":
        X, y, feature_names, metadata = dataset_manager.load_wine_2d()
    elif dataset_name.startswith("synthetic"):
        synthetic_config = config['synthetic']
        X, y, feature_names, metadata = dataset_manager.generate_synthetic_2d(
            n_samples=synthetic_config['n_samples'],
            n_classes=synthetic_config['n_classes'],
            cluster_std=synthetic_config['cluster_std'],
            dataset_type=synthetic_config['dataset_type']
        )
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")
    
    print(dataset_manager.get_dataset_info(metadata))
    
    # Preprocess data
    X_train, X_test, y_train, y_test = dataset_manager.preprocess_data(
        X, y, 
        test_size=config['dataset']['test_size'],
        scale=config['dataset']['scale']
    )
    
    print(f"Training set size: {X_train.shape[0]}")
    print(f"Test set size: {X_test.shape[0]}")
    
    # Train models and create visualizations
    models_to_train = config['models']
    class_names = metadata.get('class_names', [f'Class_{i}' for i in range(len(np.unique(y)))])
    
    print(f"\nTraining {len(models_to_train)} models...")
    
    for model_name in models_to_train:
        print(f"Training {model_name}...")
        
        # Train model
        model = visualizer.train_model(model_name, X_train, y_train)
        
        # Evaluate model
        eval_results = evaluator.generate_evaluation_report(
            model, X_test, y_test, X_train, y_train
        )
        
        print(f"  Accuracy: {eval_results['performance']['accuracy']:.3f}")
        
        # Create visualizations
        if X.shape[1] == 2:
            # 2D decision boundary
            fig = visualizer.plot_decision_boundary_2d(
                model_name, X_test, y_test, feature_names, class_names,
                resolution=config['visualization']['resolution'],
                figsize=tuple(config['visualization']['figsize'])
            )
            
            if config['visualization']['save_plots']:
                output_path = Path(config['visualization']['output_dir']) / f"{model_name.lower().replace(' ', '_')}_boundary.png"
                fig.savefig(output_path, dpi=300, bbox_inches='tight')
                print(f"  Saved plot: {output_path}")
            
            plt.close(fig)
            
            # Interactive plot
            if config['visualization']['interactive']:
                interactive_fig = visualizer.plot_decision_boundary_interactive(
                    model_name, X_test, y_test, feature_names, class_names,
                    resolution=config['visualization']['resolution']
                )
                
                if config['visualization']['save_plots']:
                    output_path = Path(config['visualization']['output_dir']) / f"{model_name.lower().replace(' ', '_')}_interactive.html"
                    interactive_fig.write_html(output_path)
                    print(f"  Saved interactive plot: {output_path}")
        
        else:
            # High-dimensional visualization
            fig = visualizer.plot_high_dimensional_boundary(
                model_name, X_test, y_test, feature_names,
                projection_method=config['visualization']['projection_method'],
                class_names=class_names,
                resolution=config['visualization']['resolution'],
                figsize=tuple(config['visualization']['figsize'])
            )
            
            if config['visualization']['save_plots']:
                output_path = Path(config['visualization']['output_dir']) / f"{model_name.lower().replace(' ', '_')}_projection.png"
                fig.savefig(output_path, dpi=300, bbox_inches='tight')
                print(f"  Saved projection plot: {output_path}")
            
            plt.close(fig)
    
    # Create comparison plot
    print("\nCreating model comparison plot...")
    comparison_fig = visualizer.compare_models(
        X_test, y_test, feature_names, models_to_train,
        figsize=(15, 10)
    )
    
    if config['visualization']['save_plots']:
        output_path = Path(config['visualization']['output_dir']) / "model_comparison.png"
        comparison_fig.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Saved comparison plot: {output_path}")
    
    plt.close(comparison_fig)
    
    # Create projection comparison (for high-dimensional data)
    if X.shape[1] > 2:
        print("\nCreating projection method comparison...")
        projection_fig = visualizer.compare_projection_methods(
            X_test, y_test, class_names, figsize=(15, 5)
        )
        
        if config['visualization']['save_plots']:
            output_path = Path(config['visualization']['output_dir']) / "projection_comparison.png"
            projection_fig.savefig(output_path, dpi=300, bbox_inches='tight')
            print(f"Saved projection comparison: {output_path}")
        
        plt.close(projection_fig)
    
    # Generate evaluation report
    if config['evaluation']['generate_report']:
        print("\nGenerating evaluation report...")
        
        # Create evaluation plots for the first model
        first_model = models_to_train[0]
        eval_results = evaluator.generate_evaluation_report(
            visualizer.models[first_model], X_test, y_test, X_train, y_train
        )
        
        eval_fig = evaluator.plot_evaluation_results(eval_results)
        
        if config['visualization']['save_plots']:
            output_path = Path(config['visualization']['output_dir']) / f"{first_model.lower().replace(' ', '_')}_evaluation.png"
            eval_fig.savefig(output_path, dpi=300, bbox_inches='tight')
            print(f"Saved evaluation plot: {output_path}")
        
        plt.close(eval_fig)
    
    print("\nDecision boundary visualization completed successfully!")
    print(f"Results saved to: {config['visualization']['output_dir']}")


if __name__ == "__main__":
    main()
