# Decision Boundary Visualization - Data Card

## Dataset Overview
- **Dataset Name**: Decision Boundary Visualization Datasets
- **Task**: Classification with decision boundary analysis
- **Version**: 1.0.0
- **Last Updated**: 2024

## Dataset Composition

### Iris Dataset (2D)
- **Source**: scikit-learn built-in dataset
- **Samples**: 150
- **Features**: 2 (sepal length, sepal width)
- **Classes**: 3 (setosa, versicolor, virginica)
- **Feature Types**: Continuous
- **Missing Values**: None
- **Sensitive Attributes**: None identified

### Wine Dataset (2D)
- **Source**: scikit-learn built-in dataset
- **Samples**: 178
- **Features**: 2 (alcohol, malic acid)
- **Classes**: 3 (wine types)
- **Feature Types**: Continuous
- **Missing Values**: None
- **Sensitive Attributes**: None identified

### Synthetic Datasets
- **Source**: Generated using scikit-learn
- **Samples**: Configurable (default: 300)
- **Features**: 2
- **Classes**: Configurable (default: 3)
- **Feature Types**: Continuous
- **Missing Values**: None
- **Sensitive Attributes**: None

## Data Collection
- **Collection Method**: Built-in datasets from scikit-learn
- **Collection Period**: N/A (standard datasets)
- **Collection Purpose**: Research and educational use
- **Consent**: N/A (public datasets)

## Data Preprocessing
- **Scaling**: StandardScaler applied by default
- **Train/Test Split**: 70/30 split with stratification
- **Feature Selection**: First 2 features for 2D visualization
- **Outlier Handling**: None applied
- **Data Augmentation**: None applied

## Data Quality
- **Completeness**: 100% complete (no missing values)
- **Consistency**: High consistency across samples
- **Accuracy**: High accuracy (standard datasets)
- **Bias Assessment**: No obvious biases identified
- **Quality Issues**: None identified

## Data Distribution
- **Class Balance**: Varies by dataset
  - Iris: Relatively balanced (50 samples per class)
  - Wine: Slightly imbalanced
  - Synthetic: Configurable balance
- **Feature Distributions**: Continuous, approximately normal
- **Correlations**: Low to moderate correlations between features

## Privacy and Ethics
- **PII**: No personally identifiable information
- **Consent**: N/A (public datasets)
- **Anonymization**: N/A (no personal data)
- **Retention**: Used only for research/education
- **Sharing**: Public datasets, safe to share

## Limitations
- **Representativeness**: May not represent all real-world scenarios
- **Bias**: Potential biases in original data collection
- **Scale**: Small datasets may not reflect large-scale behavior
- **Domain**: Specific to classification tasks

## Usage Guidelines
- **Purpose**: Research and educational use only
- **Restrictions**: Not for production decision-making
- **Validation**: Results should be validated independently
- **Documentation**: Document all assumptions and limitations

## Maintenance
- **Updates**: Regular updates to keep datasets current
- **Monitoring**: Monitor for data quality issues
- **Versioning**: Track dataset versions for reproducibility

## Contact
For questions about dataset usage or limitations, please refer to the project documentation.
