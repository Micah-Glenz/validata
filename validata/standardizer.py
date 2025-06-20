"""
Data Standardizer Module for Wangler

Provides comprehensive data standardization and normalization capabilities using
scikit-learn and custom transformations. Handles numerical scaling, categorical
encoding, date standardization, and pipeline creation.

Key Features:
- Multiple numerical standardization methods (StandardScaler, MinMaxScaler, RobustScaler)
- Categorical encoding (Label encoding, One-hot encoding, Target encoding)
- Date/datetime standardization and formatting
- Pipeline creation for reproducible transformations
- Fit/transform pattern support for train/test splits
- Operation tracking and logging
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Union, Tuple, Literal
from pathlib import Path
import logging
import pickle
from datetime import datetime

from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from .utils import ConfigManager, OperationTracker, FileOperations

logger = logging.getLogger(__name__)


class DataStandardizer:
    """
    LLM-friendly data standardization tool with comprehensive normalization capabilities.
    
    Provides methods for:
    - Numerical data standardization (Standard, MinMax, Robust scaling)
    - Categorical data encoding (Label, One-hot, Target encoding)
    - Date/datetime standardization
    - Pipeline creation and management
    - Fit/transform pattern for ML workflows
    """
    
    def __init__(self, config_manager: Optional[ConfigManager] = None):
        """
        Initialize the DataStandardizer.
        
        Args:
            config_manager: Optional configuration manager instance
        """
        self.config_manager = config_manager or ConfigManager()
        self.tracker = OperationTracker()
        self._fitted_transformers = {}
        self._standardization_history = []
    
    def standardize_numerical(
        self,
        data: Union[pd.DataFrame, str, Path],
        columns: Optional[List[str]] = None,
        method: Literal['standard', 'minmax', 'robust', 'auto'] = 'standard',
        fit_data: Optional[bool] = True,
        transformer_name: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Standardize numerical columns using various scaling methods.
        
        Args:
            data: DataFrame or path to data file
            columns: Numerical columns to standardize (auto-detect if None)
            method: Scaling method ('standard', 'minmax', 'robust', 'auto')
            fit_data: Whether to fit the transformer on this data
            transformer_name: Name to store the fitted transformer
            
        Returns:
            Dictionary containing standardized data and transformation details
            
        Example:
            >>> standardizer = DataStandardizer()
            >>> result = standardizer.standardize_numerical(df, method='standard')
            >>> standardized_df = result['standardized_data']
            >>> print(result['summary']['columns_standardized'])
        """
        self.tracker.start_operation(
            "standardize_numerical",
            columns=columns,
            method=method,
            fit_data=fit_data,
            transformer_name=transformer_name
        )
        
        try:
            # Load data if path provided
            if isinstance(data, (str, Path)):
                data = FileOperations.load_data(data)
            
            # Auto-detect numerical columns if not specified
            if columns is None:
                columns = data.select_dtypes(include=[np.number]).columns.tolist()
            
            if not columns:
                raise ValueError("No numerical columns found for standardization")
            
            standardized_data = data.copy()
            transformation_details = {}
            
            # Choose scaler based on method
            if method == 'auto':
                method = self._choose_optimal_scaling_method(data[columns])
            
            if method == 'standard':
                scaler = StandardScaler()
            elif method == 'minmax':
                scaler = MinMaxScaler()
            elif method == 'robust':
                scaler = RobustScaler()
            else:
                raise ValueError(f"Unknown scaling method: {method}")
            
            # Store original statistics
            original_stats = {}
            for col in columns:
                if col in data.columns:
                    original_stats[col] = {
                        'mean': float(data[col].mean()),
                        'std': float(data[col].std()),
                        'min': float(data[col].min()),
                        'max': float(data[col].max()),
                        'median': float(data[col].median())
                    }
            
            # Fit and transform data
            if fit_data:
                standardized_values = scaler.fit_transform(data[columns])
                
                # Store fitted transformer if name provided
                if transformer_name:
                    self._fitted_transformers[transformer_name] = {
                        'transformer': scaler,
                        'columns': columns,
                        'method': method,
                        'fitted_on': pd.Timestamp.now().isoformat()
                    }
            else:
                # Use existing transformer if available
                if transformer_name and transformer_name in self._fitted_transformers:
                    stored_transformer = self._fitted_transformers[transformer_name]
                    scaler = stored_transformer['transformer']
                    if stored_transformer['columns'] != columns:
                        logger.warning(f"Column mismatch in stored transformer {transformer_name}")
                    standardized_values = scaler.transform(data[columns])
                else:
                    raise ValueError("No fitted transformer available. Set fit_data=True or provide fitted transformer.")
            
            # Update DataFrame with standardized values
            standardized_data[columns] = standardized_values
            
            # Calculate new statistics
            new_stats = {}
            for col in columns:
                new_stats[col] = {
                    'mean': float(standardized_data[col].mean()),
                    'std': float(standardized_data[col].std()),
                    'min': float(standardized_data[col].min()),
                    'max': float(standardized_data[col].max()),
                    'median': float(standardized_data[col].median())
                }
            
            # Create transformation details
            for col in columns:
                transformation_details[col] = {
                    'original_stats': original_stats.get(col, {}),
                    'new_stats': new_stats.get(col, {}),
                    'scaling_method': method,
                    'transformation_applied': True
                }
            
            # Calculate summary
            summary = {
                'columns_standardized': len(columns),
                'scaling_method': method,
                'fit_performed': fit_data,
                'transformer_stored': transformer_name is not None,
                'transformation_details': transformation_details,
                'original_shape': data.shape,
                'final_shape': standardized_data.shape
            }
            
            self._record_standardization_operation('standardize_numerical', summary)
            
            self.tracker.complete_operation(
                columns_processed=len(columns),
                method_used=method
            )
            
            result = {
                'standardized_data': standardized_data,
                'summary': summary,
                'transformer': scaler if fit_data else None,
                'original_data': data,
                'metadata': {
                    'operation': 'standardize_numerical',
                    'timestamp': pd.Timestamp.now().isoformat(),
                    'parameters': {
                        'columns': columns,
                        'method': method,
                        'fit_data': fit_data,
                        'transformer_name': transformer_name
                    }
                }
            }
            
            return result
            
        except Exception as e:
            self.tracker.fail_operation(str(e))
            logger.error(f"Error standardizing numerical data: {str(e)}")
            raise
    
    def _choose_optimal_scaling_method(self, data: pd.DataFrame) -> str:
        """Choose optimal scaling method based on data characteristics."""
        # Calculate outlier ratio using IQR method
        outlier_ratios = []
        for col in data.columns:
            Q1 = data[col].quantile(0.25)
            Q3 = data[col].quantile(0.75)
            IQR = Q3 - Q1
            outliers = ((data[col] < (Q1 - 1.5 * IQR)) | (data[col] > (Q3 + 1.5 * IQR))).sum()
            outlier_ratios.append(outliers / len(data))
        
        avg_outlier_ratio = np.mean(outlier_ratios)
        
        # Choose method based on outlier ratio
        if avg_outlier_ratio > 0.1:  # High outliers
            return 'robust'
        elif data.min().min() >= 0:  # All positive values
            return 'minmax'
        else:  # Mixed positive/negative, moderate outliers
            return 'standard'
    
    def normalize_categorical(
        self,
        data: Union[pd.DataFrame, str, Path],
        columns: Optional[List[str]] = None,
        method: Literal['label', 'onehot', 'target', 'auto'] = 'auto',
        target_column: Optional[str] = None,
        fit_data: bool = True,
        transformer_name: Optional[str] = None,
        handle_unknown: str = 'ignore'
    ) -> Dict[str, Any]:
        """
        Encode categorical columns using various encoding methods.
        
        Args:
            data: DataFrame or path to data file
            columns: Categorical columns to encode (auto-detect if None)
            method: Encoding method ('label', 'onehot', 'target', 'auto')
            target_column: Target column for target encoding
            fit_data: Whether to fit the encoder on this data
            transformer_name: Name to store the fitted encoder
            handle_unknown: How to handle unknown categories
            
        Returns:
            Dictionary containing encoded data and transformation details
            
        Example:
            >>> result = standardizer.normalize_categorical(df, method='onehot')
            >>> encoded_df = result['encoded_data']
            >>> print(result['summary']['new_columns_created'])
        """
        self.tracker.start_operation(
            "normalize_categorical",
            columns=columns,
            method=method,
            target_column=target_column,
            fit_data=fit_data,
            transformer_name=transformer_name
        )
        
        try:
            # Load data if path provided
            if isinstance(data, (str, Path)):
                data = FileOperations.load_data(data)
            
            # Auto-detect categorical columns if not specified
            if columns is None:
                columns = data.select_dtypes(include=['object', 'category']).columns.tolist()
            
            if not columns:
                raise ValueError("No categorical columns found for encoding")
            
            encoded_data = data.copy()
            encoding_details = {}
            new_columns_created = []
            
            # Choose encoding method if auto
            if method == 'auto':
                method = self._choose_optimal_encoding_method(data[columns])
            
            # Apply encoding method
            if method == 'label':
                encoded_data, encoding_details = self._apply_label_encoding(
                    encoded_data, columns, fit_data, transformer_name, handle_unknown
                )
            elif method == 'onehot':
                encoded_data, encoding_details, new_columns_created = self._apply_onehot_encoding(
                    encoded_data, columns, fit_data, transformer_name, handle_unknown
                )
            elif method == 'target':
                if target_column is None:
                    raise ValueError("Target column required for target encoding")
                encoded_data, encoding_details = self._apply_target_encoding(
                    encoded_data, columns, target_column, fit_data, transformer_name
                )
            else:
                raise ValueError(f"Unknown encoding method: {method}")
            
            # Calculate summary
            summary = {
                'columns_encoded': len(columns),
                'encoding_method': method,
                'new_columns_created': len(new_columns_created),
                'new_column_names': new_columns_created,
                'fit_performed': fit_data,
                'transformer_stored': transformer_name is not None,
                'encoding_details': encoding_details,
                'original_shape': data.shape,
                'final_shape': encoded_data.shape
            }
            
            self._record_standardization_operation('normalize_categorical', summary)
            
            self.tracker.complete_operation(
                columns_processed=len(columns),
                method_used=method,
                new_columns=len(new_columns_created)
            )
            
            result = {
                'encoded_data': encoded_data,
                'summary': summary,
                'original_data': data,
                'metadata': {
                    'operation': 'normalize_categorical',
                    'timestamp': pd.Timestamp.now().isoformat(),
                    'parameters': {
                        'columns': columns,
                        'method': method,
                        'target_column': target_column,
                        'fit_data': fit_data,
                        'transformer_name': transformer_name,
                        'handle_unknown': handle_unknown
                    }
                }
            }
            
            return result
            
        except Exception as e:
            self.tracker.fail_operation(str(e))
            logger.error(f"Error encoding categorical data: {str(e)}")
            raise
    
    def _choose_optimal_encoding_method(self, data: pd.DataFrame) -> str:
        """Choose optimal encoding method based on data characteristics."""
        # Calculate average unique values per column
        avg_unique = np.mean([data[col].nunique() for col in data.columns])
        
        if avg_unique <= 2:  # Binary categorical
            return 'label'
        elif avg_unique <= 10:  # Low cardinality
            return 'onehot'
        else:  # High cardinality
            return 'label'  # or target if target is available
    
    def _apply_label_encoding(
        self, data: pd.DataFrame, columns: List[str], fit_data: bool, 
        transformer_name: Optional[str], handle_unknown: str
    ) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """Apply label encoding to categorical columns."""
        encoding_details = {}
        
        for col in columns:
            if col not in data.columns:
                continue
            
            # Get or create encoder
            encoder_key = f"{transformer_name}_{col}" if transformer_name else f"label_{col}"
            
            if fit_data:
                encoder = LabelEncoder()
                # Handle NaN values
                valid_mask = data[col].notna()
                if valid_mask.any():
                    encoder.fit(data.loc[valid_mask, col].astype(str))
                    data.loc[valid_mask, col] = encoder.transform(data.loc[valid_mask, col].astype(str))
                
                if transformer_name:
                    self._fitted_transformers[encoder_key] = {
                        'transformer': encoder,
                        'column': col,
                        'method': 'label',
                        'fitted_on': pd.Timestamp.now().isoformat()
                    }
                
                encoding_details[col] = {
                    'classes_': encoder.classes_.tolist() if hasattr(encoder, 'classes_') else [],
                    'n_classes': len(encoder.classes_) if hasattr(encoder, 'classes_') else 0,
                    'encoding_method': 'label'
                }
            else:
                if encoder_key in self._fitted_transformers:
                    encoder = self._fitted_transformers[encoder_key]['transformer']
                    valid_mask = data[col].notna()
                    if valid_mask.any():
                        # Handle unknown categories
                        if handle_unknown == 'ignore':
                            known_mask = data.loc[valid_mask, col].astype(str).isin(encoder.classes_)
                            data.loc[valid_mask & known_mask, col] = encoder.transform(
                                data.loc[valid_mask & known_mask, col].astype(str)
                            )
                        else:
                            data.loc[valid_mask, col] = encoder.transform(data.loc[valid_mask, col].astype(str))
                else:
                    raise ValueError(f"No fitted encoder found for {encoder_key}")
        
        return data, encoding_details
    
    def _apply_onehot_encoding(
        self, data: pd.DataFrame, columns: List[str], fit_data: bool,
        transformer_name: Optional[str], handle_unknown: str
    ) -> Tuple[pd.DataFrame, Dict[str, Any], List[str]]:
        """Apply one-hot encoding to categorical columns."""
        encoding_details = {}
        new_columns_created = []
        
        encoder_key = f"{transformer_name}_onehot" if transformer_name else "onehot_encoder"
        
        if fit_data:
            encoder = OneHotEncoder(sparse_output=False, handle_unknown=handle_unknown)
            
            # Prepare data for encoding
            data_to_encode = data[columns].fillna('missing')
            encoded_array = encoder.fit_transform(data_to_encode)
            
            # Create new column names
            feature_names = encoder.get_feature_names_out(columns)
            new_columns_created = feature_names.tolist()
            
            # Add encoded columns to dataframe
            encoded_df = pd.DataFrame(encoded_array, columns=feature_names, index=data.index)
            data = pd.concat([data.drop(columns=columns), encoded_df], axis=1)
            
            if transformer_name:
                self._fitted_transformers[encoder_key] = {
                    'transformer': encoder,
                    'columns': columns,
                    'method': 'onehot',
                    'feature_names': feature_names.tolist(),
                    'fitted_on': pd.Timestamp.now().isoformat()
                }
            
            for i, col in enumerate(columns):
                col_features = [name for name in feature_names if name.startswith(f"{col}_")]
                encoding_details[col] = {
                    'new_columns': col_features,
                    'n_categories': len(col_features),
                    'encoding_method': 'onehot'
                }
        else:
            if encoder_key in self._fitted_transformers:
                stored = self._fitted_transformers[encoder_key]
                encoder = stored['transformer']
                feature_names = stored['feature_names']
                
                data_to_encode = data[columns].fillna('missing')
                encoded_array = encoder.transform(data_to_encode)
                
                encoded_df = pd.DataFrame(encoded_array, columns=feature_names, index=data.index)
                data = pd.concat([data.drop(columns=columns), encoded_df], axis=1)
                new_columns_created = feature_names
            else:
                raise ValueError(f"No fitted encoder found for {encoder_key}")
        
        return data, encoding_details, new_columns_created
    
    def _apply_target_encoding(
        self, data: pd.DataFrame, columns: List[str], target_column: str,
        fit_data: bool, transformer_name: Optional[str]
    ) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """Apply target encoding to categorical columns."""
        encoding_details = {}
        
        if target_column not in data.columns:
            raise ValueError(f"Target column '{target_column}' not found in data")
        
        for col in columns:
            if col not in data.columns:
                continue
            
            encoder_key = f"{transformer_name}_{col}_target" if transformer_name else f"target_{col}"
            
            if fit_data:
                # Calculate target means for each category
                target_means = data.groupby(col)[target_column].mean().to_dict()
                global_mean = data[target_column].mean()
                
                # Apply encoding
                data[col] = data[col].map(target_means).fillna(global_mean)
                
                if transformer_name:
                    self._fitted_transformers[encoder_key] = {
                        'target_means': target_means,
                        'global_mean': global_mean,
                        'column': col,
                        'method': 'target',
                        'fitted_on': pd.Timestamp.now().isoformat()
                    }
                
                encoding_details[col] = {
                    'target_means': target_means,
                    'global_mean': global_mean,
                    'n_categories': len(target_means),
                    'encoding_method': 'target'
                }
            else:
                if encoder_key in self._fitted_transformers:
                    stored = self._fitted_transformers[encoder_key]
                    target_means = stored['target_means']
                    global_mean = stored['global_mean']
                    
                    data[col] = data[col].map(target_means).fillna(global_mean)
                else:
                    raise ValueError(f"No fitted encoder found for {encoder_key}")
        
        return data, encoding_details
    
    def standardize_dates(
        self,
        data: Union[pd.DataFrame, str, Path],
        columns: Optional[List[str]] = None,
        target_format: str = 'iso',
        extract_features: bool = False,
        timezone: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Standardize date/datetime columns.
        
        Args:
            data: DataFrame or path to data file
            columns: Date columns to standardize (auto-detect if None)
            target_format: Target date format ('iso', 'timestamp', 'custom')
            extract_features: Whether to extract date features (year, month, day, etc.)
            timezone: Target timezone for conversion
            
        Returns:
            Dictionary containing standardized data and transformation details
            
        Example:
            >>> result = standardizer.standardize_dates(df, extract_features=True)
            >>> standardized_df = result['standardized_data']
            >>> print(result['summary']['features_extracted'])
        """
        self.tracker.start_operation(
            "standardize_dates",
            columns=columns,
            target_format=target_format,
            extract_features=extract_features,
            timezone=timezone
        )
        
        try:
            # Load data if path provided
            if isinstance(data, (str, Path)):
                data = FileOperations.load_data(data)
            
            # Auto-detect date columns if not specified
            if columns is None:
                columns = data.select_dtypes(include=['datetime64', 'datetime']).columns.tolist()
                # Also check for potential date columns in object dtype
                for col in data.select_dtypes(include=['object']).columns:
                    try:
                        pd.to_datetime(data[col].dropna().head(100), errors='raise')
                        columns.append(col)
                    except:
                        continue
            
            if not columns:
                raise ValueError("No date columns found for standardization")
            
            standardized_data = data.copy()
            standardization_details = {}
            features_created = []
            
            for col in columns:
                if col not in data.columns:
                    continue
                
                # Convert to datetime if not already
                if not pd.api.types.is_datetime64_any_dtype(standardized_data[col]):
                    standardized_data[col] = pd.to_datetime(standardized_data[col], errors='coerce')
                
                # Apply timezone conversion if specified
                if timezone and standardized_data[col].dt.tz is None:
                    standardized_data[col] = standardized_data[col].dt.tz_localize('UTC').dt.tz_convert(timezone)
                elif timezone:
                    standardized_data[col] = standardized_data[col].dt.tz_convert(timezone)
                
                # Apply target format
                if target_format == 'iso':
                    standardized_data[col] = standardized_data[col].dt.strftime('%Y-%m-%d %H:%M:%S')
                elif target_format == 'timestamp':
                    standardized_data[col] = standardized_data[col].astype('int64') // 10**9  # Unix timestamp
                
                # Extract date features if requested
                extracted_features = []
                if extract_features:
                    date_col = pd.to_datetime(data[col], errors='coerce')
                    
                    # Extract various date components
                    feature_map = {
                        f'{col}_year': date_col.dt.year,
                        f'{col}_month': date_col.dt.month,
                        f'{col}_day': date_col.dt.day,
                        f'{col}_dayofweek': date_col.dt.dayofweek,
                        f'{col}_dayofyear': date_col.dt.dayofyear,
                        f'{col}_week': date_col.dt.isocalendar().week,
                        f'{col}_quarter': date_col.dt.quarter,
                        f'{col}_hour': date_col.dt.hour,
                        f'{col}_minute': date_col.dt.minute,
                        f'{col}_is_weekend': date_col.dt.dayofweek.isin([5, 6])
                    }
                    
                    for feature_name, feature_values in feature_map.items():
                        if feature_values.notna().any():  # Only add if there are valid values
                            standardized_data[feature_name] = feature_values
                            extracted_features.append(feature_name)
                            features_created.append(feature_name)
                
                standardization_details[col] = {
                    'target_format': target_format,
                    'timezone_applied': timezone,
                    'features_extracted': extracted_features,
                    'original_dtype': str(data[col].dtype),
                    'final_dtype': str(standardized_data[col].dtype)
                }
            
            # Calculate summary
            summary = {
                'columns_standardized': len(columns),
                'target_format': target_format,
                'features_extracted': len(features_created),
                'feature_names': features_created,
                'timezone_applied': timezone,
                'standardization_details': standardization_details,
                'original_shape': data.shape,
                'final_shape': standardized_data.shape
            }
            
            self._record_standardization_operation('standardize_dates', summary)
            
            self.tracker.complete_operation(
                columns_processed=len(columns),
                features_created=len(features_created)
            )
            
            result = {
                'standardized_data': standardized_data,
                'summary': summary,
                'original_data': data,
                'metadata': {
                    'operation': 'standardize_dates',
                    'timestamp': pd.Timestamp.now().isoformat(),
                    'parameters': {
                        'columns': columns,
                        'target_format': target_format,
                        'extract_features': extract_features,
                        'timezone': timezone
                    }
                }
            }
            
            return result
            
        except Exception as e:
            self.tracker.fail_operation(str(e))
            logger.error(f"Error standardizing dates: {str(e)}")
            raise
    
    def create_pipeline(
        self,
        numerical_columns: Optional[List[str]] = None,
        categorical_columns: Optional[List[str]] = None,
        numerical_method: str = 'standard',
        categorical_method: str = 'onehot',
        pipeline_name: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Create a scikit-learn pipeline for data standardization.
        
        Args:
            numerical_columns: List of numerical columns to include
            categorical_columns: List of categorical columns to include
            numerical_method: Method for numerical standardization
            categorical_method: Method for categorical encoding
            pipeline_name: Name to store the pipeline
            
        Returns:
            Dictionary containing the pipeline and configuration details
            
        Example:
            >>> result = standardizer.create_pipeline(
            ...     numerical_columns=['age', 'income'],
            ...     categorical_columns=['category', 'type']
            ... )
            >>> pipeline = result['pipeline']
        """
        self.tracker.start_operation(
            "create_pipeline",
            numerical_columns=numerical_columns,
            categorical_columns=categorical_columns,
            numerical_method=numerical_method,
            categorical_method=categorical_method,
            pipeline_name=pipeline_name
        )
        
        try:
            transformers = []
            
            # Add numerical transformer
            if numerical_columns:
                if numerical_method == 'standard':
                    num_transformer = StandardScaler()
                elif numerical_method == 'minmax':
                    num_transformer = MinMaxScaler()
                elif numerical_method == 'robust':
                    num_transformer = RobustScaler()
                else:
                    raise ValueError(f"Unknown numerical method: {numerical_method}")
                
                transformers.append(('num', num_transformer, numerical_columns))
            
            # Add categorical transformer
            if categorical_columns:
                if categorical_method == 'onehot':
                    cat_transformer = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
                elif categorical_method == 'label':
                    # Note: LabelEncoder doesn't work directly in pipelines with multiple columns
                    # This is a simplified approach
                    cat_transformer = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
                    logger.warning("Using OneHotEncoder instead of LabelEncoder in pipeline")
                else:
                    raise ValueError(f"Unknown categorical method: {categorical_method}")
                
                transformers.append(('cat', cat_transformer, categorical_columns))
            
            # Create the pipeline
            if not transformers:
                raise ValueError("No transformers specified. Provide numerical_columns or categorical_columns.")
            
            preprocessor = ColumnTransformer(transformers=transformers, remainder='passthrough')
            pipeline = Pipeline([('preprocessor', preprocessor)])
            
            # Store pipeline if name provided
            if pipeline_name:
                self._fitted_transformers[f"pipeline_{pipeline_name}"] = {
                    'pipeline': pipeline,
                    'numerical_columns': numerical_columns,
                    'categorical_columns': categorical_columns,
                    'numerical_method': numerical_method,
                    'categorical_method': categorical_method,
                    'created_on': pd.Timestamp.now().isoformat()
                }
            
            # Create summary
            summary = {
                'pipeline_created': True,
                'numerical_columns': numerical_columns or [],
                'categorical_columns': categorical_columns or [],
                'numerical_method': numerical_method,
                'categorical_method': categorical_method,
                'transformers_count': len(transformers),
                'pipeline_stored': pipeline_name is not None
            }
            
            self._record_standardization_operation('create_pipeline', summary)
            
            self.tracker.complete_operation(
                transformers_created=len(transformers),
                pipeline_stored=pipeline_name is not None
            )
            
            result = {
                'pipeline': pipeline,
                'summary': summary,
                'metadata': {
                    'operation': 'create_pipeline',
                    'timestamp': pd.Timestamp.now().isoformat(),
                    'parameters': {
                        'numerical_columns': numerical_columns,
                        'categorical_columns': categorical_columns,
                        'numerical_method': numerical_method,
                        'categorical_method': categorical_method,
                        'pipeline_name': pipeline_name
                    }
                }
            }
            
            return result
            
        except Exception as e:
            self.tracker.fail_operation(str(e))
            logger.error(f"Error creating pipeline: {str(e)}")
            raise
    
    def fit_transform_pipeline(
        self,
        data: Union[pd.DataFrame, str, Path],
        pipeline_name: str
    ) -> Dict[str, Any]:
        """
        Fit and transform data using a stored pipeline.
        
        Args:
            data: DataFrame or path to data file
            pipeline_name: Name of the stored pipeline
            
        Returns:
            Dictionary containing transformed data and pipeline details
        """
        if f"pipeline_{pipeline_name}" not in self._fitted_transformers:
            raise ValueError(f"Pipeline '{pipeline_name}' not found")
        
        # Load data if path provided
        if isinstance(data, (str, Path)):
            data = FileOperations.load_data(data)
        
        stored = self._fitted_transformers[f"pipeline_{pipeline_name}"]
        pipeline = stored['pipeline']
        
        # Transform data
        transformed_array = pipeline.fit_transform(data)
        
        # Create DataFrame with appropriate column names
        # This is simplified - in practice, you'd need to handle feature names properly
        transformed_data = pd.DataFrame(transformed_array, index=data.index)
        
        summary = {
            'pipeline_used': pipeline_name,
            'original_shape': data.shape,
            'transformed_shape': transformed_data.shape,
            'fit_and_transform_performed': True
        }
        
        return {
            'transformed_data': transformed_data,
            'summary': summary,
            'pipeline': pipeline,
            'original_data': data
        }
    
    def standardize_all(
        self,
        data: Union[pd.DataFrame, str, Path],
        config: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Perform comprehensive data standardization using all available methods.
        
        Args:
            data: DataFrame or path to data file
            config: Configuration for standardization operations
            
        Returns:
            Dictionary containing standardized data and comprehensive summary
            
        Example:
            >>> result = standardizer.standardize_all(df)
            >>> standardized_df = result['standardized_data']
            >>> print(result['summary']['operations_performed'])
        """
        self.tracker.start_operation("standardize_all", config=config)
        
        try:
            # Load data if path provided
            if isinstance(data, (str, Path)):
                data = FileOperations.load_data(data)
            
            # Use default config if not provided
            if config is None:
                config = self.config_manager.get('standardization', {})
            
            original_shape = data.shape
            current_data = data.copy()
            operations_summary = {}
            
            # 1. Standardize numerical columns
            numerical_columns = current_data.select_dtypes(include=[np.number]).columns.tolist()
            if numerical_columns and config.get('standardize_numerical', True):
                result = self.standardize_numerical(
                    current_data,
                    columns=numerical_columns,
                    method=config.get('numerical_strategy', 'standard')
                )
                current_data = result['standardized_data']
                operations_summary['numerical_standardization'] = result['summary']
            
            # 2. Encode categorical columns
            categorical_columns = current_data.select_dtypes(include=['object', 'category']).columns.tolist()
            if categorical_columns and config.get('encode_categorical', True):
                result = self.normalize_categorical(
                    current_data,
                    columns=categorical_columns,
                    method=config.get('categorical_strategy', 'onehot')
                )
                current_data = result['encoded_data']
                operations_summary['categorical_encoding'] = result['summary']
            
            # 3. Standardize date columns
            date_columns = current_data.select_dtypes(include=['datetime64', 'datetime']).columns.tolist()
            if date_columns and config.get('standardize_dates', True):
                result = self.standardize_dates(
                    current_data,
                    columns=date_columns,
                    target_format=config.get('date_format', 'iso'),
                    extract_features=config.get('extract_date_features', False)
                )
                current_data = result['standardized_data']
                operations_summary['date_standardization'] = result['summary']
            
            # Calculate comprehensive summary
            summary = {
                'original_shape': original_shape,
                'final_shape': current_data.shape,
                'operations_performed': list(operations_summary.keys()),
                'total_columns_added': current_data.shape[1] - original_shape[1],
                'operation_details': operations_summary
            }
            
            self._record_standardization_operation('standardize_all', summary)
            
            self.tracker.complete_operation(
                operations_performed=len(operations_summary),
                columns_added=summary['total_columns_added']
            )
            
            result = {
                'standardized_data': current_data,
                'summary': summary,
                'original_data': data,
                'metadata': {
                    'operation': 'standardize_all',
                    'timestamp': pd.Timestamp.now().isoformat(),
                    'parameters': {'config': config}
                }
            }
            
            return result
            
        except Exception as e:
            self.tracker.fail_operation(str(e))
            logger.error(f"Error in comprehensive standardization: {str(e)}")
            raise
    
    def save_transformers(self, file_path: Union[str, Path]) -> str:
        """
        Save all fitted transformers to file.
        
        Args:
            file_path: Path to save transformers
            
        Returns:
            Path to saved file
        """
        file_path = Path(file_path)
        file_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(file_path, 'wb') as f:
            pickle.dump(self._fitted_transformers, f)
        
        logger.info(f"Transformers saved to {file_path}")
        return str(file_path)
    
    def load_transformers(self, file_path: Union[str, Path]) -> None:
        """
        Load fitted transformers from file.
        
        Args:
            file_path: Path to load transformers from
        """
        file_path = Path(file_path)
        
        if not file_path.exists():
            raise FileNotFoundError(f"Transformer file not found: {file_path}")
        
        with open(file_path, 'rb') as f:
            self._fitted_transformers = pickle.load(f)
        
        logger.info(f"Transformers loaded from {file_path}")
    
    def get_fitted_transformers(self) -> Dict[str, Any]:
        """Get information about all fitted transformers."""
        return {
            name: {
                'method': info.get('method', 'unknown'),
                'columns': info.get('columns', info.get('column', [])),
                'fitted_on': info.get('fitted_on', 'unknown')
            }
            for name, info in self._fitted_transformers.items()
        }
    
    def _record_standardization_operation(self, operation: str, summary: Dict[str, Any]) -> None:
        """Record standardization operation in history."""
        self._standardization_history.append({
            'operation': operation,
            'timestamp': pd.Timestamp.now().isoformat(),
            'summary': summary
        })
    
    def get_standardization_history(self) -> List[Dict[str, Any]]:
        """Get history of all standardization operations performed."""
        return self._standardization_history.copy()
    
    def get_operation_summary(self) -> Dict[str, Any]:
        """Get summary of all operations performed."""
        return self.tracker.get_summary()
    
    def export_standardization_report(self, file_path: Union[str, Path]) -> str:
        """
        Export comprehensive standardization report.
        
        Args:
            file_path: Output file path
            
        Returns:
            Path to exported report
        """
        report_data = {
            'standardization_history': self._standardization_history,
            'fitted_transformers': self.get_fitted_transformers(),
            'operation_summary': self.tracker.get_summary(),
            'generated_at': pd.Timestamp.now().isoformat()
        }
        
        file_path = Path(file_path)
        file_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(file_path, 'w') as f:
            import json
            json.dump(report_data, f, indent=2, default=str)
        
        logger.info(f"Standardization report exported to {file_path}")
        return str(file_path)