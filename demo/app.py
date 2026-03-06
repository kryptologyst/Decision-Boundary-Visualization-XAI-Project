"""Streamlit demo for Decision Boundary Visualization."""

import streamlit as st
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
import seaborn as sns

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from utils.device import set_deterministic_seed, get_device_name
from data.dataset import DatasetManager
from methods.advanced_boundary import AdvancedDecisionBoundaryVisualizer
from metrics.evaluation import DecisionBoundaryEvaluator


# Page configuration
st.set_page_config(
    page_title="Decision Boundary Visualization",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .warning-box {
        background-color: #fff3cd;
        border: 1px solid #ffeaa7;
        border-radius: 0.375rem;
        padding: 1rem;
        margin: 1rem 0;
    }
    .metric-card {
        background-color: #f8f9fa;
        border: 1px solid #dee2e6;
        border-radius: 0.375rem;
        padding: 1rem;
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Header
st.markdown('<h1 class="main-header">🎯 Decision Boundary Visualization</h1>', unsafe_allow_html=True)

# Disclaimer
st.markdown("""
<div class="warning-box">
    <strong>⚠️ Important Disclaimer:</strong> This tool is for research and educational purposes only. 
    Decision boundary visualizations may be unstable or misleading and should not be used for regulated 
    decisions without human review and validation.
</div>
""", unsafe_allow_html=True)

# Sidebar
st.sidebar.header("Configuration")

# Dataset selection
dataset_options = {
    "Iris (2D)": "iris_2d",
    "Wine (2D)": "wine_2d", 
    "Synthetic Blobs": "synthetic_blobs",
    "Synthetic Classification": "synthetic_classification"
}

selected_dataset = st.sidebar.selectbox(
    "Select Dataset",
    list(dataset_options.keys()),
    index=0
)

dataset_name = dataset_options[selected_dataset]

# Model selection
available_models = [
    "Random Forest",
    "SVM (RBF)",
    "SVM (Linear)", 
    "Neural Network",
    "Decision Tree",
    "k-NN"
]

selected_models = st.sidebar.multiselect(
    "Select Models",
    available_models,
    default=["Random Forest", "SVM (RBF)"]
)

# Visualization options
st.sidebar.subheader("Visualization Options")
resolution = st.sidebar.slider("Resolution", 0.01, 0.1, 0.02, 0.01)
interactive_mode = st.sidebar.checkbox("Interactive Mode", value=True)
projection_method = st.sidebar.selectbox(
    "Projection Method (for high-dim data)",
    ["umap", "pca"],
    index=0
)

# Evaluation options
st.sidebar.subheader("Evaluation Options")
evaluate_stability = st.sidebar.checkbox("Evaluate Stability", value=True)
evaluate_smoothness = st.sidebar.checkbox("Evaluate Smoothness", value=True)

# Initialize session state
if 'data_loaded' not in st.session_state:
    st.session_state.data_loaded = False
if 'models_trained' not in st.session_state:
    st.session_state.models_trained = False

# Load data
@st.cache_data
def load_dataset_data(dataset_name: str, random_state: int = 42):
    """Load dataset with caching."""
    dataset_manager = DatasetManager(random_state=random_state)
    
    if dataset_name == "iris_2d":
        X, y, feature_names, metadata = dataset_manager.load_iris_2d()
    elif dataset_name == "wine_2d":
        X, y, feature_names, metadata = dataset_manager.load_wine_2d()
    elif dataset_name == "synthetic_blobs":
        X, y, feature_names, metadata = dataset_manager.generate_synthetic_2d(
            n_samples=300, n_classes=3, cluster_std=1.0, dataset_type="blobs"
        )
    elif dataset_name == "synthetic_classification":
        X, y, feature_names, metadata = dataset_manager.generate_synthetic_2d(
            n_samples=300, n_classes=3, cluster_std=1.0, dataset_type="classification"
        )
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")
    
    return X, y, feature_names, metadata

# Load dataset
if st.sidebar.button("Load Dataset") or st.session_state.data_loaded:
    with st.spinner("Loading dataset..."):
        X, y, feature_names, metadata = load_dataset_data(dataset_name)
        X_train, X_test, y_train, y_test = DatasetManager().preprocess_data(X, y)
        
        st.session_state.X = X
        st.session_state.y = y
        st.session_state.X_train = X_train
        st.session_state.X_test = X_test
        st.session_state.y_train = y_train
        st.session_state.y_test = y_test
        st.session_state.feature_names = feature_names
        st.session_state.metadata = metadata
        st.session_state.data_loaded = True

# Display dataset info
if st.session_state.data_loaded:
    st.subheader("📊 Dataset Information")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Samples", len(st.session_state.X))
    with col2:
        st.metric("Features", st.session_state.X.shape[1])
    with col3:
        st.metric("Classes", len(np.unique(st.session_state.y)))
    with col4:
        st.metric("Test Size", len(st.session_state.X_test))
    
    # Dataset description
    st.info(f"**Dataset:** {st.session_state.metadata['dataset_name']} - {st.session_state.metadata['description']}")
    
    # Data preview
    if st.checkbox("Show Data Preview"):
        df = pd.DataFrame(st.session_state.X, columns=st.session_state.feature_names)
        df['target'] = st.session_state.y
        st.dataframe(df.head(10))

# Train models
if st.session_state.data_loaded and selected_models:
    if st.sidebar.button("Train Models") or st.session_state.models_trained:
        with st.spinner("Training models..."):
            set_deterministic_seed(42)
            
            visualizer = AdvancedDecisionBoundaryVisualizer(random_state=42)
            evaluator = DecisionBoundaryEvaluator(random_state=42)
            
            # Train selected models
            trained_models = {}
            evaluation_results = {}
            
            for model_name in selected_models:
                model = visualizer.train_model(
                    model_name, 
                    st.session_state.X_train, 
                    st.session_state.y_train
                )
                trained_models[model_name] = model
                
                # Evaluate model
                eval_results = evaluator.generate_evaluation_report(
                    model, 
                    st.session_state.X_test, 
                    st.session_state.y_test,
                    st.session_state.X_train,
                    st.session_state.y_train
                )
                evaluation_results[model_name] = eval_results
            
            st.session_state.trained_models = trained_models
            st.session_state.evaluation_results = evaluation_results
            st.session_state.visualizer = visualizer
            st.session_state.evaluator = evaluator
            st.session_state.models_trained = True

# Display results
if st.session_state.models_trained:
    st.subheader("📈 Model Performance")
    
    # Performance metrics table
    performance_data = []
    for model_name, results in st.session_state.evaluation_results.items():
        perf = results['performance']
        performance_data.append({
            'Model': model_name,
            'Accuracy': f"{perf['accuracy']:.3f}",
            'Precision': f"{perf['precision']:.3f}",
            'Recall': f"{perf['recall']:.3f}",
            'F1-Score': f"{perf['f1_score']:.3f}"
        })
    
    df_performance = pd.DataFrame(performance_data)
    st.dataframe(df_performance, use_container_width=True)
    
    # Visualization section
    st.subheader("🎨 Decision Boundary Visualizations")
    
    # Select model for detailed visualization
    selected_model = st.selectbox(
        "Select Model for Detailed Visualization",
        selected_models,
        index=0
    )
    
    if selected_model:
        class_names = st.session_state.metadata.get('class_names', 
                                                   [f'Class_{i}' for i in range(len(np.unique(st.session_state.y)))])
        
        # Create tabs for different visualizations
        tab1, tab2, tab3 = st.tabs(["2D Boundary", "Interactive Plot", "Evaluation"])
        
        with tab1:
            st.subheader(f"Decision Boundary - {selected_model}")
            
            if st.session_state.X.shape[1] == 2:
                # 2D visualization
                fig = st.session_state.visualizer.plot_decision_boundary_2d(
                    selected_model,
                    st.session_state.X_test,
                    st.session_state.y_test,
                    st.session_state.feature_names,
                    class_names,
                    resolution=resolution
                )
                st.pyplot(fig)
            else:
                # High-dimensional visualization
                fig = st.session_state.visualizer.plot_high_dimensional_boundary(
                    selected_model,
                    st.session_state.X_test,
                    st.session_state.y_test,
                    st.session_state.feature_names,
                    projection_method=projection_method,
                    class_names=class_names,
                    resolution=resolution
                )
                st.pyplot(fig)
        
        with tab2:
            if interactive_mode:
                st.subheader(f"Interactive Decision Boundary - {selected_model}")
                
                interactive_fig = st.session_state.visualizer.plot_decision_boundary_interactive(
                    selected_model,
                    st.session_state.X_test,
                    st.session_state.y_test,
                    st.session_state.feature_names,
                    class_names,
                    resolution=resolution
                )
                st.plotly_chart(interactive_fig, use_container_width=True)
            else:
                st.info("Enable interactive mode in the sidebar to see interactive plots.")
        
        with tab3:
            st.subheader(f"Evaluation Results - {selected_model}")
            
            eval_results = st.session_state.evaluation_results[selected_model]
            
            # Performance metrics
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Accuracy", f"{eval_results['performance']['accuracy']:.3f}")
            with col2:
                st.metric("Precision", f"{eval_results['performance']['precision']:.3f}")
            with col3:
                st.metric("Recall", f"{eval_results['performance']['recall']:.3f}")
            with col4:
                st.metric("F1-Score", f"{eval_results['performance']['f1_score']:.3f}")
            
            # Stability metrics
            if evaluate_stability and 'stability' in eval_results:
                st.subheader("Stability Metrics")
                stability = eval_results['stability']
                
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Avg Agreement Rate", f"{stability['avg_agreement_rate']:.3f}")
                with col2:
                    st.metric("Avg Prediction Variance", f"{stability['avg_prediction_variance']:.3f}")
                
                # Agreement rate distribution
                fig, ax = plt.subplots(figsize=(8, 4))
                ax.hist(stability['agreement_rates'], bins=20, alpha=0.7, edgecolor='black')
                ax.axvline(stability['avg_agreement_rate'], color='red', linestyle='--', 
                          label=f'Average: {stability["avg_agreement_rate"]:.3f}')
                ax.set_xlabel('Agreement Rate')
                ax.set_ylabel('Frequency')
                ax.set_title('Prediction Agreement Rate Distribution')
                ax.legend()
                st.pyplot(fig)
            
            # Smoothness metrics
            if evaluate_smoothness and 'smoothness' in eval_results:
                st.subheader("Smoothness Metrics")
                smoothness = eval_results['smoothness']
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Avg Gradient", f"{smoothness['avg_gradient_magnitude']:.3f}")
                with col2:
                    st.metric("Max Gradient", f"{smoothness['max_gradient_magnitude']:.3f}")
                with col3:
                    st.metric("Boundary Length", f"{smoothness['boundary_length']:.3f}")
                
                # Gradient visualization
                fig, ax = plt.subplots(figsize=(8, 6))
                im = ax.imshow(smoothness['gradient_field'], cmap='viridis', aspect='auto')
                ax.set_title('Decision Boundary Gradient Magnitude')
                plt.colorbar(im, ax=ax)
                st.pyplot(fig)
    
    # Model comparison
    if len(selected_models) > 1:
        st.subheader("🔄 Model Comparison")
        
        # Comparison plot
        comparison_fig = st.session_state.visualizer.compare_models(
            st.session_state.X_test,
            st.session_state.y_test,
            st.session_state.feature_names,
            selected_models,
            figsize=(15, 10)
        )
        st.pyplot(comparison_fig)
        
        # Performance comparison
        st.subheader("Performance Comparison")
        
        # Create performance comparison chart
        models = list(st.session_state.evaluation_results.keys())
        metrics = ['accuracy', 'precision', 'recall', 'f1_score']
        
        fig = go.Figure()
        
        for metric in metrics:
            values = [st.session_state.evaluation_results[model]['performance'][metric] for model in models]
            fig.add_trace(go.Bar(
                name=metric.title(),
                x=models,
                y=values,
                text=[f"{v:.3f}" for v in values],
                textposition='auto'
            ))
        
        fig.update_layout(
            title="Model Performance Comparison",
            xaxis_title="Models",
            yaxis_title="Score",
            barmode='group',
            height=500
        )
        
        st.plotly_chart(fig, use_container_width=True)

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666; font-size: 0.9rem;">
    Decision Boundary Visualization Tool | 
    <a href="https://github.com/your-repo" target="_blank">GitHub</a> | 
    For research and educational purposes only
</div>
""", unsafe_allow_html=True)
