"""
Custom Pandas-Based Data Profiler Module for Wrangler

Provides comprehensive data profiling capabilities using only pandas and numpy.
Creates detailed, LLM-friendly reports about data quality, distributions, correlations,
and potential issues in datasets.

Key Features:
- Lightweight profiling with zero external dependencies (pandas/numpy only)
- LLM-optimized structured output format
- Fast performance on large datasets
- Configurable profiling depth
- Custom data quality checks
- Export to JSON format
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Union, Tuple
from pathlib import Path
import json
import logging
from datetime import datetime

from .utils import ConfigManager, OperationTracker, FileOperations

logger = logging.getLogger(__name__)


class CustomDataProfiler:
    """
    LLM-friendly data profiling tool using pandas and numpy.
    
    Provides comprehensive data analysis including:
    - Dataset overview and statistics
    - Variable-level analysis
    - Data quality assessment
    - Missing data patterns
    - Correlation analysis
    - Distribution analysis
    """
    
    def __init__(self, config_manager: Optional[ConfigManager] = None):
        """
        Initialize the CustomDataProfiler.
        
        Args:
            config_manager: Optional configuration manager instance
        """
        self.config_manager = config_manager or ConfigManager()
        self.tracker = OperationTracker()
        self._profile_history = []
    
    def profile_data(
        self,
        data: Union[pd.DataFrame, str, Path],
        title: Optional[str] = None,
        minimal: bool = False,
        include_correlations: bool = True,
        sample_size: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Generate comprehensive data profile.
        
        Args:
            data: DataFrame or path to data file
            title: Title for the profile report
            minimal: Whether to generate minimal profile (faster)
            include_correlations: Whether to compute correlations
            sample_size: Sample size for large datasets (None = use all data)
            
        Returns:
            Dictionary containing comprehensive data profile
            
        Example:
            >>> profiler = CustomDataProfiler()
            >>> profile = profiler.profile_data(df, title="Sales Data")
            >>> print(f"Dataset shape: {profile['summary']['shape']}")
            >>> print(f"Missing data: {profile['summary']['missing_cells_percent']:.1f}%")
        """
        self.tracker.start_operation(
            "profile_data",
            minimal=minimal,
            include_correlations=include_correlations,
            sample_size=sample_size
        )
        
        try:
            # Load data if path provided
            if isinstance(data, (str, Path)):
                data = FileOperations.load_data(data)
            
            if not isinstance(data, pd.DataFrame):
                raise ValueError("Data must be a pandas DataFrame or file path")
            
            # Sample data if needed
            original_shape = data.shape
            if sample_size and len(data) > sample_size:
                data = data.sample(n=sample_size, random_state=42)
                logger.info(f"Sampled {sample_size} rows from {original_shape[0]} total rows")
            
            logger.info(f"Profiling dataset with {len(data)} rows and {len(data.columns)} columns")
            
            # Generate profile
            profile_result = {
                "title": title or f"Data Profile - {datetime.now().strftime('%Y-%m-%d %H:%M')}",
                "summary": self._generate_summary(data, original_shape),
                "columns": self._analyze_columns(data, minimal),
                "correlations": self._analyze_correlations(data) if include_correlations and not minimal else {},
                "quality_report": self._generate_quality_report(data),
                "metadata": {
                    "profiling_date": datetime.now().isoformat(),
                    "profiler_version": "custom-1.0",
                    "minimal_mode": minimal,
                    "sample_size": sample_size,
                    "original_shape": original_shape,
                    "analyzed_shape": data.shape
                }
            }
            
            # Record operation
            self._record_profile_operation(profile_result["summary"])
            
            self.tracker.complete_operation(
                rows_analyzed=len(data),
                columns_analyzed=len(data.columns),
                quality_issues=len(profile_result["quality_report"]["issues"])
            )
            
            return profile_result
            
        except Exception as e:
            self.tracker.fail_operation(str(e))
            logger.error(f"Error during profiling: {str(e)}")
            raise
    
    def _generate_summary(self, data: pd.DataFrame, original_shape: Tuple[int, int]) -> Dict[str, Any]:
        """Generate dataset overview summary."""
        total_cells = data.shape[0] * data.shape[1]
        
        # Use explicit operations to avoid boolean arithmetic issues
        missing_mask = data.isnull()
        missing_cells = int(missing_mask.sum().sum())
        missing_cells_percent = round((missing_cells / total_cells) * 100, 2) if total_cells > 0 else 0
        
        # Handle duplicates with explicit operations
        duplicate_mask = data.duplicated()
        duplicate_rows = int(duplicate_mask.sum())
        duplicate_rows_percent = round((duplicate_rows / len(data)) * 100, 2) if len(data) > 0 else 0
        
        return {
            "shape": list(data.shape),
            "original_shape": list(original_shape),
            "n_records": len(data),
            "n_variables": len(data.columns),
            "memory_usage_mb": round(data.memory_usage(deep=True).sum() / 1024 / 1024, 3),
            "missing_cells": missing_cells,
            "missing_cells_percent": missing_cells_percent,
            "duplicate_rows": duplicate_rows,
            "duplicate_rows_percent": duplicate_rows_percent,
            "data_types": {
                "numerical": len(data.select_dtypes(include=[np.number]).columns),
                "categorical": len(data.select_dtypes(include=['object', 'category']).columns),
                "datetime": len(data.select_dtypes(include=['datetime64']).columns),
                "boolean": len(data.select_dtypes(include=['bool']).columns)
            }
        }
    
    def _analyze_columns(self, data: pd.DataFrame, minimal: bool = False) -> Dict[str, Dict[str, Any]]:
        """Analyze each column individually."""
        column_profiles = {}
        
        for col in data.columns:
            column_data = data[col]
            col_type = self._detect_column_type(column_data)
            
            # Calculate missing data with explicit operations to avoid boolean arithmetic issues
            missing_mask = column_data.isnull()
            missing_count = int(missing_mask.sum())
            missing_percent = round((missing_count / len(column_data)) * 100, 2) if len(column_data) > 0 else 0
            
            base_profile = {
                "type": col_type,
                "missing_count": missing_count,
                "missing_percent": missing_percent,
                "unique_count": int(column_data.nunique()),
                "data_type": str(column_data.dtype)
            }
            
            # Type-specific analysis
            if col_type == "numerical":
                base_profile.update(self._analyze_numerical_column(column_data, minimal))
            elif col_type == "categorical":
                base_profile.update(self._analyze_categorical_column(column_data, minimal))
            elif col_type == "datetime":
                base_profile.update(self._analyze_datetime_column(column_data, minimal))
            elif col_type == "boolean":
                base_profile.update(self._analyze_boolean_column(column_data))
            
            # Add quality issues
            base_profile["quality_issues"] = self._detect_column_quality_issues(column_data, col_type)
            
            column_profiles[col] = base_profile
        
        return column_profiles
    
    def _detect_column_type(self, column: pd.Series) -> str:
        """Detect the semantic type of a column."""
        # Check boolean first since pandas considers bool as numeric
        if pd.api.types.is_bool_dtype(column):
            return "boolean"
        elif pd.api.types.is_numeric_dtype(column):
            return "numerical"
        elif pd.api.types.is_datetime64_any_dtype(column):
            return "datetime"
        else:
            return "categorical"
    
    def _analyze_numerical_column(self, column: pd.Series, minimal: bool = False) -> Dict[str, Any]:
        """Analyze numerical column."""
        non_null_data = column.dropna()
        
        if len(non_null_data) == 0:
            return {"stats": {}, "outliers_count": 0, "zeros_count": 0}
        
        stats = {
            "mean": float(non_null_data.mean()),
            "median": float(non_null_data.median()),
            "std": float(non_null_data.std()),
            "min": float(non_null_data.min()),
            "max": float(non_null_data.max()),
            "q25": float(non_null_data.quantile(0.25)),
            "q75": float(non_null_data.quantile(0.75))
        }
        
        # Outlier detection using IQR
        outliers_count = 0
        if not minimal:
            iqr = stats["q75"] - stats["q25"]
            if iqr > 0:  # Avoid issues with zero IQR
                lower_bound = stats["q25"] - 1.5 * iqr
                upper_bound = stats["q75"] + 1.5 * iqr
                # Use explicit boolean operations to avoid pandas 2.1.4 issues
                outliers_mask = (non_null_data < lower_bound) | (non_null_data > upper_bound)
                outliers_count = int(outliers_mask.sum())
        
        return {
            "stats": stats,
            "outliers_count": outliers_count,
            "zeros_count": int((non_null_data == 0).sum()),
            "negative_count": int((non_null_data < 0).sum()),
            "infinite_count": int(np.isinf(column).sum())
        }
    
    def _analyze_categorical_column(self, column: pd.Series, minimal: bool = False) -> Dict[str, Any]:
        """Analyze categorical column."""
        non_null_data = column.dropna()
        
        if len(non_null_data) == 0:
            return {"top_values": {}, "cardinality_ratio": 0}
        
        value_counts = non_null_data.value_counts()
        top_values = value_counts.head(10 if not minimal else 5).to_dict()
        
        return {
            "top_values": {str(k): int(v) for k, v in top_values.items()},
            "cardinality_ratio": round(len(value_counts) / len(non_null_data), 3),
            "mode": str(value_counts.index[0]) if len(value_counts) > 0 else None,
            "mode_frequency": int(value_counts.iloc[0]) if len(value_counts) > 0 else 0
        }
    
    def _analyze_datetime_column(self, column: pd.Series, minimal: bool = False) -> Dict[str, Any]:
        """Analyze datetime column."""
        non_null_data = column.dropna()
        
        if len(non_null_data) == 0:
            return {"date_range": {}, "frequency_analysis": {}}
        
        date_range = {
            "min": non_null_data.min().isoformat() if hasattr(non_null_data.min(), 'isoformat') else str(non_null_data.min()),
            "max": non_null_data.max().isoformat() if hasattr(non_null_data.max(), 'isoformat') else str(non_null_data.max())
        }
        
        frequency_analysis = {}
        if not minimal and len(non_null_data) > 1:
            try:
                # Try to infer frequency
                if hasattr(non_null_data, 'dt'):
                    frequency_analysis = {
                        "year_range": int(non_null_data.dt.year.max() - non_null_data.dt.year.min()),
                        "unique_years": int(non_null_data.dt.year.nunique()),
                        "unique_months": int(non_null_data.dt.month.nunique()),
                        "unique_days": int(non_null_data.dt.day.nunique())
                    }
            except:
                frequency_analysis = {"analysis_failed": True}
        
        return {
            "date_range": date_range,
            "frequency_analysis": frequency_analysis
        }
    
    def _analyze_boolean_column(self, column: pd.Series) -> Dict[str, Any]:
        """Analyze boolean column."""
        non_null_data = column.dropna()
        
        if len(non_null_data) == 0:
            return {"true_count": 0, "false_count": 0, "true_ratio": 0}
        
        # Use explicit boolean logic instead of bitwise operations
        true_count = int(non_null_data.sum())
        false_count = len(non_null_data) - true_count  # Avoid bitwise NOT operator
        
        return {
            "true_count": true_count,
            "false_count": false_count,
            "true_ratio": round(true_count / len(non_null_data), 3) if len(non_null_data) > 0 else 0
        }
    
    def _analyze_correlations(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Analyze correlations between numerical columns."""
        numerical_cols = data.select_dtypes(include=[np.number]).columns
        
        if len(numerical_cols) < 2:
            return {"message": "Not enough numerical columns for correlation analysis"}
        
        try:
            corr_matrix = data[numerical_cols].corr()
            
            # Find high correlations
            high_correlations = []
            for i, col1 in enumerate(numerical_cols):
                for j, col2 in enumerate(numerical_cols[i+1:], i+1):
                    corr_value = corr_matrix.iloc[i, j]
                    if abs(corr_value) > 0.7:  # High correlation threshold
                        high_correlations.append({
                            "column1": col1,
                            "column2": col2,
                            "correlation": round(float(corr_value), 3),
                            "strength": "very_strong" if abs(corr_value) > 0.9 else "strong"
                        })
            
            return {
                "correlation_matrix": corr_matrix.round(3).to_dict(),
                "high_correlations": high_correlations,
                "numerical_columns_count": len(numerical_cols)
            }
        except Exception as e:
            return {"error": f"Correlation analysis failed: {str(e)}"}
    
    def _detect_column_quality_issues(self, column: pd.Series, col_type: str) -> List[str]:
        """Detect quality issues for a column."""
        issues = []
        
        # Missing data - use explicit operations to avoid boolean arithmetic issues
        missing_mask = column.isnull()
        missing_count = int(missing_mask.sum())
        missing_pct = (missing_count / len(column)) * 100 if len(column) > 0 else 0
        if missing_pct > 50:
            issues.append(f"High missing data: {missing_pct:.1f}%")
        elif missing_pct > 20:
            issues.append(f"Moderate missing data: {missing_pct:.1f}%")
        
        # Constant values
        if column.nunique() == 1:
            issues.append("Constant column (single unique value)")
        
        # High cardinality for categorical
        if col_type == "categorical":
            cardinality_ratio = column.nunique() / len(column)
            if cardinality_ratio > 0.95:
                issues.append("Very high cardinality (potential ID column)")
            elif cardinality_ratio > 0.5:
                issues.append("High cardinality")
        
        # Outliers for numerical
        if col_type == "numerical" and len(column.dropna()) > 0:
            non_null_data = column.dropna()
            Q1 = non_null_data.quantile(0.25)
            Q3 = non_null_data.quantile(0.75)
            IQR = Q3 - Q1
            if IQR > 0:  # Avoid division by zero
                # Use explicit boolean operations to avoid pandas 2.1.4 issues
                lower_outliers = non_null_data < (Q1 - 1.5 * IQR)
                upper_outliers = non_null_data > (Q3 + 1.5 * IQR)
                outliers = int(lower_outliers.sum() + upper_outliers.sum())
                outlier_pct = (outliers / len(non_null_data)) * 100
                if outlier_pct > 10:
                    issues.append(f"High outlier rate: {outlier_pct:.1f}%")
        
        return issues
    
    def _generate_quality_report(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Generate overall data quality report."""
        issues = []
        recommendations = []
        
        # Overall missing data - use explicit operations to avoid boolean arithmetic issues
        missing_mask = data.isnull()
        total_missing = int(missing_mask.sum().sum())
        total_cells = data.shape[0] * data.shape[1]
        missing_pct = (total_missing / total_cells) * 100 if total_cells > 0 else 0
        if missing_pct > 20:
            issues.append(f"High overall missing data: {missing_pct:.1f}%")
            recommendations.append("Consider imputation strategies for missing data")
        
        # Duplicate rows - use explicit operations to avoid boolean arithmetic issues
        duplicate_mask = data.duplicated()
        duplicate_count = int(duplicate_mask.sum())
        dup_pct = (duplicate_count / len(data)) * 100 if len(data) > 0 else 0
        if dup_pct > 5:
            issues.append(f"High duplicate rate: {dup_pct:.1f}%")
            recommendations.append("Remove or investigate duplicate rows")
        
        # Memory usage
        memory_mb = data.memory_usage(deep=True).sum() / 1024 / 1024
        if memory_mb > 1000:  # 1GB
            issues.append(f"Large memory usage: {memory_mb:.1f} MB")
            recommendations.append("Consider data type optimization or sampling")
        
        # Column-specific issues
        for col in data.columns:
            col_issues = self._detect_column_quality_issues(data[col], self._detect_column_type(data[col]))
            if col_issues:
                issues.extend([f"{col}: {issue}" for issue in col_issues])
        
        return {
            "issues_found": len(issues),
            "issues": issues,
            "recommendations": recommendations,
            "quality_score": max(0, 100 - len(issues) * 5)  # Simple scoring
        }
    
    def quick_profile(self, data: Union[pd.DataFrame, str, Path]) -> str:
        """
        Generate a quick text summary of the dataset.
        
        Args:
            data: DataFrame or path to data file
            
        Returns:
            String with quick summary
            
        Example:
            >>> profiler = CustomDataProfiler()
            >>> summary = profiler.quick_profile(df)
            >>> print(summary)
        """
        try:
            # Load data if path provided
            if isinstance(data, (str, Path)):
                data = FileOperations.load_data(data)
            
            summary = self._generate_summary(data, data.shape)
            
            return f"""Dataset Quick Profile:
• Shape: {summary['n_records']} rows × {summary['n_variables']} columns
• Memory: {summary['memory_usage_mb']} MB
• Missing data: {summary['missing_cells_percent']}%
• Duplicates: {summary['duplicate_rows_percent']}%
• Data types: {summary['data_types']['numerical']} numerical, {summary['data_types']['categorical']} categorical"""
            
        except Exception as e:
            logger.error(f"Error in quick_profile: {str(e)}")
            return f"Error generating profile: {str(e)}"
    
    def export_profile(self, profile: Dict[str, Any], file_path: Union[str, Path]) -> str:
        """
        Export profile to JSON file.
        
        Args:
            profile: Profile dictionary from profile_data()
            file_path: Output file path
            
        Returns:
            Path to exported file
        """
        file_path = Path(file_path)
        file_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(file_path, 'w') as f:
            json.dump(profile, f, indent=2, default=str)
        
        logger.info(f"Profile exported to {file_path}")
        return str(file_path)
    
    def _record_profile_operation(self, summary: Dict[str, Any]) -> None:
        """Record profiling operation in history."""
        self._profile_history.append({
            "timestamp": datetime.now().isoformat(),
            "summary": summary
        })
    
    def get_profile_history(self) -> List[Dict[str, Any]]:
        """Get history of all profiling operations."""
        return self._profile_history.copy()
    
    def get_operation_summary(self) -> Dict[str, Any]:
        """Get summary of all operations performed."""
        return self.tracker.get_summary()


# Alias for backward compatibility
DataProfiler = CustomDataProfiler