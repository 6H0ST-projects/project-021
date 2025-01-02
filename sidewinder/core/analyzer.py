import pandas as pd
import numpy as np
import logging
from typing import Dict, Any
from pandas.api.types import is_numeric_dtype, is_string_dtype

logging.basicConfig(level=logging.INFO)

def analyze_data(source_data: pd.DataFrame) -> Dict[str, Any]:
    """
    Performs comprehensive data analysis on the provided DataFrame.
    
    Args:
        source_data (pd.DataFrame): The DataFrame to be analyzed.
    
    Returns:
        Dict[str, Any]: Dictionary containing analysis results including:
            - schema: Column data types and structure
            - quality: Data quality metrics
            - statistics: Summary statistics
            - patterns: Detected patterns in string columns
            - anomalies: Detected anomalies in numeric columns
    """
    try:
        logging.info("Starting data analysis...")
        results = {}

        # Schema analysis
        logging.info("Analyzing schema...")
        results['schema'] = {
            'fields': {
                col: {
                    'name': col,
                    'type': str(source_data[col].dtype),
                    'nullable': source_data[col].isnull().any(),
                    'unique_ratio': source_data[col].nunique() / len(source_data)
                }
                for col in source_data.columns
            },
            'row_count': len(source_data),
            'column_count': len(source_data.columns)
        }

        # Data quality analysis
        logging.info("Analyzing data quality...")
        results['quality'] = {
            'missing_values': source_data.isnull().sum().to_dict(),
            'duplicate_rows': source_data.duplicated().sum(),
            'completeness': {
                col: 1 - (source_data[col].isnull().sum() / len(source_data))
                for col in source_data.columns
            }
        }

        # Statistical analysis
        logging.info("Generating statistics...")
        numeric_stats = source_data.describe().to_dict()
        categorical_stats = source_data.describe(include=['object']).to_dict()
        results['statistics'] = {
            'numeric': numeric_stats,
            'categorical': categorical_stats
        }

        # Pattern detection
        logging.info("Detecting patterns...")
        patterns = {}
        for col in source_data.columns:
            if is_string_dtype(source_data[col]):
                value_counts = source_data[col].value_counts()
                patterns[col] = {
                    'unique_values': source_data[col].nunique(),
                    'most_common': value_counts.head(5).to_dict(),
                    'least_common': value_counts.tail(5).to_dict()
                }
        results['patterns'] = patterns

        # Anomaly detection
        logging.info("Detecting anomalies...")
        anomalies = {}
        for col in source_data.columns:
            if is_numeric_dtype(source_data[col]):
                Q1 = source_data[col].quantile(0.25)
                Q3 = source_data[col].quantile(0.75)
                IQR = Q3 - Q1
                outliers = source_data[
                    (source_data[col] < (Q1 - 1.5 * IQR)) | 
                    (source_data[col] > (Q3 + 1.5 * IQR))
                ][col].tolist()
                anomalies[col] = {
                    'outliers': outliers,
                    'outlier_count': len(outliers),
                    'iqr': float(IQR),
                    'q1': float(Q1),
                    'q3': float(Q3)
                }
        results['anomalies'] = anomalies

        logging.info("Data analysis completed.")
        return results

    except Exception as e:
        logging.error(f"An error occurred during data analysis: {str(e)}")
        raise 