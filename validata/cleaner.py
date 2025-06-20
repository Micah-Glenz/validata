"""
Data Cleaner Module for Wangler

Provides comprehensive data cleaning capabilities including handling missing data,
removing duplicates, detecting and handling outliers, fixing data types, and
cleaning text data.

Key Features:
- Multiple strategies for handling missing data
- Smart duplicate detection and removal
- Statistical outlier detection (IQR, Z-score)
- Intelligent data type inference and conversion
- Text data standardization and cleaning
- Operation tracking and logging
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Union, Tuple, Literal
from pathlib import Path
import logging
import re
from scipy import stats

from .utils import ConfigManager, OperationTracker, FileOperations, DataTypeInference

logger = logging.getLogger(__name__)


class DataCleaner:
    """
    LLM-friendly data cleaning tool with comprehensive cleaning capabilities.
    
    Provides methods for:
    - Missing data handling with multiple strategies
    - Duplicate detection and removal
    - Outlier detection and handling
    - Data type fixing and conversion
    - Text data cleaning and standardization
    """
    
    def __init__(self, config_manager: Optional[ConfigManager] = None):
        """
        Initialize the DataCleaner.
        
        Args:
            config_manager: Optional configuration manager instance
        """
        self.config_manager = config_manager or ConfigManager()
        self.tracker = OperationTracker()
        self._cleaning_history = []
    
    def handle_missing_data(
        self,
        data: Union[pd.DataFrame, str, Path],
        strategy: Literal['drop', 'fill', 'interpolate', 'auto'] = 'auto',
        fill_value: Optional[Union[str, int, float, Dict[str, Any]]] = None,
        threshold: Optional[float] = None,
        columns: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Handle missing data using various strategies.
        
        Args:
            data: DataFrame or path to data file
            strategy: Strategy to use ('drop', 'fill', 'interpolate', 'auto')
            fill_value: Value to use for filling (can be dict for column-specific values)
            threshold: Threshold for dropping rows/columns (fraction of missing values)
            columns: Specific columns to process (all if None)
            
        Returns:
            Dictionary containing cleaned data and operation summary
            
        Example:
            >>> cleaner = DataCleaner()
            >>> result = cleaner.handle_missing_data(df, strategy='auto')
            >>> cleaned_df = result['cleaned_data']
            >>> print(result['summary']['missing_cells_removed'])
        """
        self.tracker.start_operation(
            "handle_missing_data",
            strategy=strategy,
            fill_value=fill_value,
            threshold=threshold,
            columns=columns
        )
        
        try:
            # Load data if path provided
            if isinstance(data, (str, Path)):
                data = FileOperations.load_data(data)
            
            if not isinstance(data, pd.DataFrame):
                raise ValueError("Data must be a pandas DataFrame or file path")
            
            original_shape = data.shape
            original_missing = data.isnull().sum().sum()
            
            # Select columns to process
            if columns is None:
                columns = data.columns.tolist()
            
            # Apply threshold filtering if specified
            if threshold is not None:
                # Drop columns with missing ratio above threshold
                missing_ratio = data[columns].isnull().sum() / len(data)
                columns_to_keep = missing_ratio[missing_ratio <= threshold].index.tolist()
                columns_to_drop = [col for col in columns if col not in columns_to_keep]
                
                if columns_to_drop:
                    data = data.drop(columns=columns_to_drop)
                    logger.info(f"Dropped {len(columns_to_drop)} columns with missing ratio > {threshold}")
                
                columns = columns_to_keep
            
            # Create a copy for cleaning
            cleaned_data = data.copy()
            
            # Apply cleaning strategy
            if strategy == 'auto':
                cleaned_data = self._auto_handle_missing(cleaned_data, columns)
            elif strategy == 'drop':
                cleaned_data = self._drop_missing(cleaned_data, columns)
            elif strategy == 'fill':
                cleaned_data = self._fill_missing(cleaned_data, columns, fill_value)
            elif strategy == 'interpolate':
                cleaned_data = self._interpolate_missing(cleaned_data, columns)
            else:
                raise ValueError(f"Unknown strategy: {strategy}")
            
            # Calculate summary statistics
            final_missing = cleaned_data.isnull().sum().sum()
            summary = {
                'original_shape': original_shape,
                'final_shape': cleaned_data.shape,
                'original_missing_cells': int(original_missing),
                'final_missing_cells': int(final_missing),
                'missing_cells_removed': int(original_missing - final_missing),
                'missing_reduction_percent': float((original_missing - final_missing) / max(1, original_missing) * 100),
                'strategy_used': strategy,
                'columns_processed': columns,
                'rows_dropped': int(original_shape[0] - cleaned_data.shape[0]),
                'columns_dropped': int(original_shape[1] - cleaned_data.shape[1])
            }
            
            # Record cleaning operation
            self._record_cleaning_operation('handle_missing_data', summary)
            
            self.tracker.complete_operation(
                original_missing=original_missing,
                final_missing=final_missing,
                reduction_percent=summary['missing_reduction_percent']
            )
            
            result = {
                'cleaned_data': cleaned_data,
                'summary': summary,
                'original_data': data,
                'metadata': {
                    'operation': 'handle_missing_data',
                    'timestamp': pd.Timestamp.now().isoformat(),
                    'parameters': {
                        'strategy': strategy,
                        'fill_value': fill_value,
                        'threshold': threshold,
                        'columns': columns
                    }
                }
            }
            
            return result
            
        except Exception as e:
            self.tracker.fail_operation(str(e))
            logger.error(f"Error handling missing data: {str(e)}")
            raise
    
    def _auto_handle_missing(self, data: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
        """Automatically choose the best strategy for each column."""
        cleaned_data = data.copy()
        
        for col in columns:
            if col not in cleaned_data.columns:
                continue
                
            missing_ratio = cleaned_data[col].isnull().sum() / len(cleaned_data)
            
            if missing_ratio == 0:
                continue  # No missing values
            elif missing_ratio > 0.5:
                # Too many missing values, consider dropping column
                logger.warning(f"Column '{col}' has {missing_ratio:.1%} missing values")
                continue
            elif missing_ratio < 0.05:
                # Few missing values, safe to drop rows
                cleaned_data = cleaned_data.dropna(subset=[col])
            else:
                # Moderate missing values, use appropriate fill strategy
                if pd.api.types.is_numeric_dtype(cleaned_data[col]):
                    # Use median for numeric columns
                    cleaned_data[col] = cleaned_data[col].fillna(cleaned_data[col].median())
                elif pd.api.types.is_datetime64_any_dtype(cleaned_data[col]):
                    # Use forward fill for datetime
                    cleaned_data[col] = cleaned_data[col].fillna(method='ffill')
                else:
                    # Use mode for categorical/text columns
                    mode_value = cleaned_data[col].mode()
                    if len(mode_value) > 0:
                        cleaned_data[col] = cleaned_data[col].fillna(mode_value[0])
                    else:
                        cleaned_data[col] = cleaned_data[col].fillna('unknown')
        
        return cleaned_data
    
    def _drop_missing(self, data: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
        """Drop rows with missing values."""
        return data.dropna(subset=columns)
    
    def _fill_missing(self, data: pd.DataFrame, columns: List[str], fill_value: Any) -> pd.DataFrame:
        """Fill missing values with specified values."""
        cleaned_data = data.copy()
        
        if isinstance(fill_value, dict):
            # Column-specific fill values
            for col in columns:
                if col in fill_value and col in cleaned_data.columns:
                    cleaned_data[col] = cleaned_data[col].fillna(fill_value[col])
        else:
            # Single fill value for all columns
            for col in columns:
                if col in cleaned_data.columns:
                    cleaned_data[col] = cleaned_data[col].fillna(fill_value)
        
        return cleaned_data
    
    def _interpolate_missing(self, data: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
        """Interpolate missing values for numeric columns."""
        cleaned_data = data.copy()
        
        for col in columns:
            if col in cleaned_data.columns and pd.api.types.is_numeric_dtype(cleaned_data[col]):
                cleaned_data[col] = cleaned_data[col].interpolate()
        
        return cleaned_data
    
    def remove_duplicates(
        self,
        data: Union[pd.DataFrame, str, Path],
        subset: Optional[List[str]] = None,
        keep: Literal['first', 'last', False] = 'first',
        ignore_index: bool = True
    ) -> Dict[str, Any]:
        """
        Remove duplicate rows from the dataset.
        
        Args:
            data: DataFrame or path to data file
            subset: Columns to consider for duplicate identification
            keep: Which duplicates to keep ('first', 'last', False)
            ignore_index: Reset index after removing duplicates
            
        Returns:
            Dictionary containing cleaned data and operation summary
            
        Example:
            >>> result = cleaner.remove_duplicates(df, subset=['id', 'name'])
            >>> print(f"Removed {result['summary']['duplicates_removed']} duplicates")
        """
        self.tracker.start_operation(
            "remove_duplicates",
            subset=subset,
            keep=keep,
            ignore_index=ignore_index
        )
        
        try:
            # Load data if path provided
            if isinstance(data, (str, Path)):
                data = FileOperations.load_data(data)
            
            original_shape = data.shape
            
            # Identify duplicates
            duplicates_mask = data.duplicated(subset=subset, keep=False)
            num_duplicates = duplicates_mask.sum()
            
            # Remove duplicates
            cleaned_data = data.drop_duplicates(subset=subset, keep=keep, ignore_index=ignore_index)
            
            # Calculate summary
            duplicates_removed = original_shape[0] - cleaned_data.shape[0]
            summary = {
                'original_shape': original_shape,
                'final_shape': cleaned_data.shape,
                'duplicates_found': int(num_duplicates),
                'duplicates_removed': int(duplicates_removed),
                'duplicate_rate_percent': float(num_duplicates / original_shape[0] * 100),
                'subset_columns': subset,
                'keep_strategy': keep
            }
            
            self._record_cleaning_operation('remove_duplicates', summary)
            
            self.tracker.complete_operation(
                duplicates_removed=duplicates_removed,
                duplicate_rate=summary['duplicate_rate_percent']
            )
            
            result = {
                'cleaned_data': cleaned_data,
                'summary': summary,
                'original_data': data,
                'duplicate_rows': data[duplicates_mask] if num_duplicates > 0 else pd.DataFrame(),
                'metadata': {
                    'operation': 'remove_duplicates',
                    'timestamp': pd.Timestamp.now().isoformat(),
                    'parameters': {
                        'subset': subset,
                        'keep': keep,
                        'ignore_index': ignore_index
                    }
                }
            }
            
            return result
            
        except Exception as e:
            self.tracker.fail_operation(str(e))
            logger.error(f"Error removing duplicates: {str(e)}")
            raise
    
    def handle_outliers(
        self,
        data: Union[pd.DataFrame, str, Path],
        method: Literal['iqr', 'zscore', 'isolation_forest', 'auto'] = 'auto',
        threshold: Optional[float] = None,
        columns: Optional[List[str]] = None,
        action: Literal['remove', 'cap', 'flag'] = 'flag'
    ) -> Dict[str, Any]:
        """
        Detect and handle outliers in numeric columns.
        
        Args:
            data: DataFrame or path to data file
            method: Outlier detection method ('iqr', 'zscore', 'isolation_forest', 'auto')
            threshold: Threshold for outlier detection (method-specific)
            columns: Numeric columns to check (auto-detect if None)
            action: Action to take ('remove', 'cap', 'flag')
            
        Returns:
            Dictionary containing cleaned data and outlier information
            
        Example:
            >>> result = cleaner.handle_outliers(df, method='iqr', action='cap')
            >>> print(f"Found {result['summary']['outliers_found']} outliers")
        """
        self.tracker.start_operation(
            "handle_outliers",
            method=method,
            threshold=threshold,
            columns=columns,
            action=action
        )
        
        try:
            # Load data if path provided
            if isinstance(data, (str, Path)):
                data = FileOperations.load_data(data)
            
            # Auto-detect numeric columns if not specified
            if columns is None:
                columns = data.select_dtypes(include=[np.number]).columns.tolist()
            
            original_shape = data.shape
            cleaned_data = data.copy()
            outlier_info = {}
            
            # Detect outliers for each column
            for col in columns:
                if col not in data.columns:
                    continue
                    
                if method == 'auto':
                    # Choose method based on data characteristics
                    if len(data) < 1000:
                        outliers = self._detect_outliers_iqr(data[col])
                    else:
                        outliers = self._detect_outliers_zscore(data[col])
                elif method == 'iqr':
                    outliers = self._detect_outliers_iqr(data[col], threshold)
                elif method == 'zscore':
                    outliers = self._detect_outliers_zscore(data[col], threshold)
                elif method == 'isolation_forest':
                    outliers = self._detect_outliers_isolation_forest(data[[col]])
                else:
                    raise ValueError(f"Unknown method: {method}")
                
                outlier_info[col] = {
                    'outlier_indices': outliers.tolist(),
                    'outlier_count': len(outliers),
                    'outlier_values': data.loc[outliers, col].tolist()
                }
                
                # Apply action
                if action == 'remove':
                    cleaned_data = cleaned_data.drop(outliers)
                elif action == 'cap':
                    cleaned_data = self._cap_outliers(cleaned_data, col, outliers)
                elif action == 'flag':
                    cleaned_data[f'{col}_outlier_flag'] = cleaned_data.index.isin(outliers)
            
            # Calculate summary
            total_outliers = sum(len(info['outlier_indices']) for info in outlier_info.values())
            summary = {
                'original_shape': original_shape,
                'final_shape': cleaned_data.shape,
                'outliers_found': int(total_outliers),
                'outliers_by_column': {col: info['outlier_count'] for col, info in outlier_info.items()},
                'method_used': method,
                'action_taken': action,
                'columns_processed': columns
            }
            
            self._record_cleaning_operation('handle_outliers', summary)
            
            self.tracker.complete_operation(
                outliers_found=total_outliers,
                columns_processed=len(columns)
            )
            
            result = {
                'cleaned_data': cleaned_data,
                'summary': summary,
                'outlier_info': outlier_info,
                'original_data': data,
                'metadata': {
                    'operation': 'handle_outliers',
                    'timestamp': pd.Timestamp.now().isoformat(),
                    'parameters': {
                        'method': method,
                        'threshold': threshold,
                        'columns': columns,
                        'action': action
                    }
                }
            }
            
            return result
            
        except Exception as e:
            self.tracker.fail_operation(str(e))
            logger.error(f"Error handling outliers: {str(e)}")
            raise
    
    def _detect_outliers_iqr(self, series: pd.Series, threshold: Optional[float] = None) -> pd.Index:
        """Detect outliers using IQR method."""
        if threshold is None:
            threshold = 1.5
            
        Q1 = series.quantile(0.25)
        Q3 = series.quantile(0.75)
        IQR = Q3 - Q1
        
        lower_bound = Q1 - threshold * IQR
        upper_bound = Q3 + threshold * IQR
        
        outliers = series[(series < lower_bound) | (series > upper_bound)].index
        return outliers
    
    def _detect_outliers_zscore(self, series: pd.Series, threshold: Optional[float] = None) -> pd.Index:
        """Detect outliers using Z-score method."""
        if threshold is None:
            threshold = 3.0
            
        z_scores = np.abs(stats.zscore(series.dropna()))
        outlier_mask = z_scores > threshold
        
        # Map back to original indices
        valid_indices = series.dropna().index
        outliers = valid_indices[outlier_mask]
        return outliers
    
    def _detect_outliers_isolation_forest(self, data: pd.DataFrame) -> pd.Index:
        """Detect outliers using Isolation Forest."""
        from sklearn.ensemble import IsolationForest
        
        # Remove rows with NaN values for this analysis
        clean_data = data.dropna()
        
        if len(clean_data) < 10:
            return pd.Index([])  # Not enough data for isolation forest
        
        iso_forest = IsolationForest(contamination=0.1, random_state=42)
        outlier_labels = iso_forest.fit_predict(clean_data)
        
        # Get indices of outliers (labeled as -1)
        outliers = clean_data.index[outlier_labels == -1]
        return outliers
    
    def _cap_outliers(self, data: pd.DataFrame, column: str, outlier_indices: pd.Index) -> pd.DataFrame:
        """Cap outliers to reasonable bounds."""
        series = data[column]
        Q1 = series.quantile(0.25)
        Q3 = series.quantile(0.75)
        IQR = Q3 - Q1
        
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        # Cap outliers
        data.loc[outlier_indices, column] = data.loc[outlier_indices, column].clip(lower_bound, upper_bound)
        
        return data
    
    def fix_data_types(
        self,
        data: Union[pd.DataFrame, str, Path],
        type_mapping: Optional[Dict[str, str]] = None,
        auto_infer: bool = True,
        strict: bool = False
    ) -> Dict[str, Any]:
        """
        Fix and convert data types based on content analysis.
        
        Args:
            data: DataFrame or path to data file
            type_mapping: Manual type mapping (column -> type)
            auto_infer: Automatically infer optimal types
            strict: Raise errors on conversion failures
            
        Returns:
            Dictionary containing data with fixed types and conversion summary
            
        Example:
            >>> result = cleaner.fix_data_types(df, auto_infer=True)
            >>> print(result['summary']['types_changed'])
        """
        self.tracker.start_operation(
            "fix_data_types",
            type_mapping=type_mapping,
            auto_infer=auto_infer,
            strict=strict
        )
        
        try:
            # Load data if path provided
            if isinstance(data, (str, Path)):
                data = FileOperations.load_data(data)
            
            original_types = data.dtypes.to_dict()
            cleaned_data = data.copy()
            conversion_summary = {}
            
            # Auto-infer types if requested
            if auto_infer:
                inferred_types = DataTypeInference.infer_column_types(data)
                
                for col, inferred_type in inferred_types.items():
                    try:
                        if inferred_type == 'integer':
                            cleaned_data[col] = pd.to_numeric(cleaned_data[col], errors='coerce').astype('Int64')
                        elif inferred_type == 'float':
                            cleaned_data[col] = pd.to_numeric(cleaned_data[col], errors='coerce')
                        elif inferred_type == 'datetime':
                            cleaned_data[col] = pd.to_datetime(cleaned_data[col], errors='coerce', infer_datetime_format=True)
                        elif inferred_type == 'boolean':
                            cleaned_data[col] = cleaned_data[col].astype('boolean')
                        elif inferred_type == 'categorical':
                            cleaned_data[col] = cleaned_data[col].astype('category')
                        
                        conversion_summary[col] = {
                            'original_type': str(original_types[col]),
                            'new_type': str(cleaned_data[col].dtype),
                            'inferred_type': inferred_type,
                            'conversion_successful': True
                        }
                        
                    except Exception as e:
                        conversion_summary[col] = {
                            'original_type': str(original_types[col]),
                            'new_type': str(original_types[col]),
                            'inferred_type': inferred_type,
                            'conversion_successful': False,
                            'error': str(e)
                        }
                        
                        if strict:
                            raise ValueError(f"Failed to convert column '{col}' to {inferred_type}: {str(e)}")
            
            # Apply manual type mapping if provided
            if type_mapping:
                for col, target_type in type_mapping.items():
                    if col not in cleaned_data.columns:
                        continue
                        
                    try:
                        if target_type.lower() in ['int', 'integer']:
                            cleaned_data[col] = pd.to_numeric(cleaned_data[col], errors='coerce').astype('Int64')
                        elif target_type.lower() in ['float', 'numeric']:
                            cleaned_data[col] = pd.to_numeric(cleaned_data[col], errors='coerce')
                        elif target_type.lower() in ['datetime', 'date']:
                            cleaned_data[col] = pd.to_datetime(cleaned_data[col], errors='coerce')
                        elif target_type.lower() in ['bool', 'boolean']:
                            cleaned_data[col] = cleaned_data[col].astype('boolean')
                        elif target_type.lower() in ['category', 'categorical']:
                            cleaned_data[col] = cleaned_data[col].astype('category')
                        elif target_type.lower() in ['string', 'str', 'text']:
                            cleaned_data[col] = cleaned_data[col].astype('string')
                        
                        conversion_summary[col] = {
                            'original_type': str(original_types[col]),
                            'new_type': str(cleaned_data[col].dtype),
                            'target_type': target_type,
                            'conversion_successful': True
                        }
                        
                    except Exception as e:
                        conversion_summary[col] = {
                            'original_type': str(original_types[col]),
                            'new_type': str(original_types[col]),
                            'target_type': target_type,
                            'conversion_successful': False,
                            'error': str(e)
                        }
                        
                        if strict:
                            raise ValueError(f"Failed to convert column '{col}' to {target_type}: {str(e)}")
            
            # Calculate summary
            successful_conversions = sum(1 for info in conversion_summary.values() if info['conversion_successful'])
            failed_conversions = len(conversion_summary) - successful_conversions
            
            summary = {
                'total_columns': len(data.columns),
                'columns_processed': len(conversion_summary),
                'successful_conversions': successful_conversions,
                'failed_conversions': failed_conversions,
                'types_changed': {col: info for col, info in conversion_summary.items() if info['conversion_successful']},
                'conversion_failures': {col: info for col, info in conversion_summary.items() if not info['conversion_successful']},
                'final_types': cleaned_data.dtypes.to_dict()
            }
            
            self._record_cleaning_operation('fix_data_types', summary)
            
            self.tracker.complete_operation(
                successful_conversions=successful_conversions,
                failed_conversions=failed_conversions
            )
            
            result = {
                'cleaned_data': cleaned_data,
                'summary': summary,
                'conversion_details': conversion_summary,
                'original_data': data,
                'metadata': {
                    'operation': 'fix_data_types',
                    'timestamp': pd.Timestamp.now().isoformat(),
                    'parameters': {
                        'type_mapping': type_mapping,
                        'auto_infer': auto_infer,
                        'strict': strict
                    }
                }
            }
            
            return result
            
        except Exception as e:
            self.tracker.fail_operation(str(e))
            logger.error(f"Error fixing data types: {str(e)}")
            raise
    
    def clean_text_data(
        self,
        data: Union[pd.DataFrame, str, Path],
        columns: Optional[List[str]] = None,
        operations: Optional[List[str]] = None,
        custom_patterns: Optional[Dict[str, str]] = None
    ) -> Dict[str, Any]:
        """
        Clean and standardize text data.
        
        Args:
            data: DataFrame or path to data file
            columns: Text columns to clean (auto-detect if None)
            operations: List of cleaning operations to perform
            custom_patterns: Custom regex patterns for cleaning
            
        Returns:
            Dictionary containing cleaned data and operation summary
            
        Example:
            >>> result = cleaner.clean_text_data(df, operations=['lowercase', 'remove_special'])
            >>> print(f"Cleaned {result['summary']['columns_processed']} text columns")
        """
        self.tracker.start_operation(
            "clean_text_data",
            columns=columns,
            operations=operations,
            custom_patterns=custom_patterns
        )
        
        try:
            # Load data if path provided
            if isinstance(data, (str, Path)):
                data = FileOperations.load_data(data)
            
            # Auto-detect text columns if not specified
            if columns is None:
                columns = data.select_dtypes(include=['object', 'string']).columns.tolist()
            
            # Default operations if not specified
            if operations is None:
                operations = ['strip', 'normalize_whitespace', 'remove_special_chars']
            
            cleaned_data = data.copy()
            cleaning_summary = {}
            
            for col in columns:
                if col not in data.columns:
                    continue
                
                original_values = cleaned_data[col].copy()
                column_summary = {'operations_applied': []}
                
                # Apply each operation
                for operation in operations:
                    if operation == 'lowercase':
                        cleaned_data[col] = cleaned_data[col].str.lower()
                        column_summary['operations_applied'].append('lowercase')
                    
                    elif operation == 'uppercase':
                        cleaned_data[col] = cleaned_data[col].str.upper()
                        column_summary['operations_applied'].append('uppercase')
                    
                    elif operation == 'strip':
                        cleaned_data[col] = cleaned_data[col].str.strip()
                        column_summary['operations_applied'].append('strip')
                    
                    elif operation == 'normalize_whitespace':
                        cleaned_data[col] = cleaned_data[col].str.replace(r'\s+', ' ', regex=True)
                        column_summary['operations_applied'].append('normalize_whitespace')
                    
                    elif operation == 'remove_special_chars':
                        cleaned_data[col] = cleaned_data[col].str.replace(r'[^\w\s]', '', regex=True)
                        column_summary['operations_applied'].append('remove_special_chars')
                    
                    elif operation == 'remove_digits':
                        cleaned_data[col] = cleaned_data[col].str.replace(r'\d+', '', regex=True)
                        column_summary['operations_applied'].append('remove_digits')
                    
                    elif operation == 'remove_punctuation':
                        cleaned_data[col] = cleaned_data[col].str.replace(r'[^\w\s]', '', regex=True)
                        column_summary['operations_applied'].append('remove_punctuation')
                
                # Apply custom patterns
                if custom_patterns:
                    for pattern_name, pattern in custom_patterns.items():
                        cleaned_data[col] = cleaned_data[col].str.replace(pattern, '', regex=True)
                        column_summary['operations_applied'].append(f'custom_{pattern_name}')
                
                # Calculate changes
                changes_made = (original_values != cleaned_data[col]).sum()
                column_summary.update({
                    'values_changed': int(changes_made),
                    'change_percentage': float(changes_made / len(original_values) * 100),
                    'unique_values_before': original_values.nunique(),
                    'unique_values_after': cleaned_data[col].nunique()
                })
                
                cleaning_summary[col] = column_summary
            
            # Calculate overall summary
            total_changes = sum(info['values_changed'] for info in cleaning_summary.values())
            summary = {
                'columns_processed': len(columns),
                'total_values_changed': total_changes,
                'operations_performed': operations,
                'custom_patterns_applied': list(custom_patterns.keys()) if custom_patterns else [],
                'column_details': cleaning_summary
            }
            
            self._record_cleaning_operation('clean_text_data', summary)
            
            self.tracker.complete_operation(
                columns_processed=len(columns),
                values_changed=total_changes
            )
            
            result = {
                'cleaned_data': cleaned_data,
                'summary': summary,
                'original_data': data,
                'metadata': {
                    'operation': 'clean_text_data',
                    'timestamp': pd.Timestamp.now().isoformat(),
                    'parameters': {
                        'columns': columns,
                        'operations': operations,
                        'custom_patterns': custom_patterns
                    }
                }
            }
            
            return result
            
        except Exception as e:
            self.tracker.fail_operation(str(e))
            logger.error(f"Error cleaning text data: {str(e)}")
            raise
    
    def clean_all(
        self,
        data: Union[pd.DataFrame, str, Path],
        config: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Perform comprehensive data cleaning using all available methods.
        
        Args:
            data: DataFrame or path to data file
            config: Configuration for cleaning operations
            
        Returns:
            Dictionary containing cleaned data and comprehensive summary
            
        Example:
            >>> result = cleaner.clean_all(df)
            >>> cleaned_df = result['cleaned_data']
            >>> print(result['summary']['operations_performed'])
        """
        self.tracker.start_operation("clean_all", config=config)
        
        try:
            # Load data if path provided
            if isinstance(data, (str, Path)):
                data = FileOperations.load_data(data)
            
            # Use default config if not provided
            if config is None:
                config = self.config_manager.get('cleaning', {})
            
            original_shape = data.shape
            current_data = data.copy()
            operations_summary = {}
            
            # 1. Handle missing data
            if config.get('handle_missing', True):
                result = self.handle_missing_data(current_data, strategy='auto')
                current_data = result['cleaned_data']
                operations_summary['missing_data'] = result['summary']
            
            # 2. Remove duplicates
            if config.get('handle_duplicates', True):
                result = self.remove_duplicates(current_data)
                current_data = result['cleaned_data']
                operations_summary['duplicates'] = result['summary']
            
            # 3. Fix data types
            if config.get('fix_data_types', True):
                result = self.fix_data_types(current_data, auto_infer=True)
                current_data = result['cleaned_data']
                operations_summary['data_types'] = result['summary']
            
            # 4. Handle outliers
            if config.get('detect_outliers', True):
                result = self.handle_outliers(current_data, action='flag')
                current_data = result['cleaned_data']
                operations_summary['outliers'] = result['summary']
            
            # 5. Clean text data
            text_columns = current_data.select_dtypes(include=['object', 'string']).columns.tolist()
            if text_columns and config.get('clean_text', True):
                result = self.clean_text_data(current_data, columns=text_columns)
                current_data = result['cleaned_data']
                operations_summary['text_cleaning'] = result['summary']
            
            # Calculate comprehensive summary
            summary = {
                'original_shape': original_shape,
                'final_shape': current_data.shape,
                'operations_performed': list(operations_summary.keys()),
                'total_rows_removed': original_shape[0] - current_data.shape[0],
                'total_columns_added': current_data.shape[1] - original_shape[1],
                'operation_details': operations_summary,
                'data_quality_improvement': self._calculate_quality_improvement(data, current_data)
            }
            
            self._record_cleaning_operation('clean_all', summary)
            
            self.tracker.complete_operation(
                operations_performed=len(operations_summary),
                rows_removed=summary['total_rows_removed']
            )
            
            result = {
                'cleaned_data': current_data,
                'summary': summary,
                'original_data': data,
                'metadata': {
                    'operation': 'clean_all',
                    'timestamp': pd.Timestamp.now().isoformat(),
                    'parameters': {'config': config}
                }
            }
            
            return result
            
        except Exception as e:
            self.tracker.fail_operation(str(e))
            logger.error(f"Error in comprehensive cleaning: {str(e)}")
            raise
    
    def _calculate_quality_improvement(self, original: pd.DataFrame, cleaned: pd.DataFrame) -> Dict[str, float]:
        """Calculate data quality improvement metrics."""
        original_missing = original.isnull().sum().sum() / (original.shape[0] * original.shape[1]) * 100
        cleaned_missing = cleaned.isnull().sum().sum() / (cleaned.shape[0] * cleaned.shape[1]) * 100
        
        original_duplicates = original.duplicated().sum() / len(original) * 100
        cleaned_duplicates = cleaned.duplicated().sum() / len(cleaned) * 100
        
        return {
            'missing_data_reduction': max(0, original_missing - cleaned_missing),
            'duplicate_reduction': max(0, original_duplicates - cleaned_duplicates),
            'completeness_improvement': max(0, (100 - cleaned_missing) - (100 - original_missing)),
            'uniqueness_improvement': max(0, (100 - cleaned_duplicates) - (100 - original_duplicates))
        }
    
    def _record_cleaning_operation(self, operation: str, summary: Dict[str, Any]) -> None:
        """Record cleaning operation in history."""
        self._cleaning_history.append({
            'operation': operation,
            'timestamp': pd.Timestamp.now().isoformat(),
            'summary': summary
        })
    
    def get_cleaning_history(self) -> List[Dict[str, Any]]:
        """Get history of all cleaning operations performed."""
        return self._cleaning_history.copy()
    
    def get_operation_summary(self) -> Dict[str, Any]:
        """Get summary of all operations performed."""
        return self.tracker.get_summary()
    
    def export_cleaning_report(self, file_path: Union[str, Path]) -> str:
        """
        Export comprehensive cleaning report.
        
        Args:
            file_path: Output file path
            
        Returns:
            Path to exported report
        """
        report_data = {
            'cleaning_history': self._cleaning_history,
            'operation_summary': self.tracker.get_summary(),
            'generated_at': pd.Timestamp.now().isoformat()
        }
        
        file_path = Path(file_path)
        file_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(file_path, 'w') as f:
            import json
            json.dump(report_data, f, indent=2, default=str)
        
        logger.info(f"Cleaning report exported to {file_path}")
        return str(file_path)