"""
Utility functions and classes for the Wangler data toolkit.

Provides common functionality used across different modules including:
- Data type detection and inference
- File I/O operations
- Configuration management
- Logging and error handling
- Common data transformations
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Union, Tuple
import logging
from pathlib import Path
import json
import yaml

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DataTypeInference:
    """Intelligent data type detection and inference utilities."""
    
    @staticmethod
    def infer_column_types(df: pd.DataFrame) -> Dict[str, str]:
        """
        Infer the most appropriate data types for DataFrame columns.
        
        Args:
            df: Input DataFrame
            
        Returns:
            Dictionary mapping column names to inferred types
        """
        type_map = {}
        
        for col in df.columns:
            series = df[col]
            
            # Skip if all null
            if series.isnull().all():
                type_map[col] = 'object'
                continue
                
            # Try numeric conversion
            try:
                pd.to_numeric(series, errors='raise')
                if series.dtype == 'int64' or (series % 1 == 0).all():
                    type_map[col] = 'integer'
                else:
                    type_map[col] = 'float'
                continue
            except (ValueError, TypeError):
                pass
            
            # Try datetime conversion
            try:
                pd.to_datetime(series, errors='raise', infer_datetime_format=True)
                type_map[col] = 'datetime'
                continue
            except (ValueError, TypeError):
                pass
            
            # Check for boolean
            unique_vals = series.dropna().unique()
            if len(unique_vals) <= 2 and all(str(v).lower() in ['true', 'false', '1', '0', 'yes', 'no'] for v in unique_vals):
                type_map[col] = 'boolean'
                continue
                
            # Default to categorical or text
            if len(unique_vals) / len(series) < 0.5:  # High repetition
                type_map[col] = 'categorical'
            else:
                type_map[col] = 'text'
                
        return type_map
    
    @staticmethod
    def suggest_constraints(df: pd.DataFrame, column: str) -> Dict[str, Any]:
        """
        Suggest validation constraints based on column data.
        
        Args:
            df: Input DataFrame
            column: Column name to analyze
            
        Returns:
            Dictionary of suggested constraints
        """
        series = df[column]
        constraints = {}
        
        # Null constraints
        null_count = series.isnull().sum()
        if null_count == 0:
            constraints['nullable'] = False
        else:
            constraints['nullable'] = True
            constraints['null_percentage'] = null_count / len(series)
        
        # Unique constraint
        if series.nunique() == len(series.dropna()):
            constraints['unique'] = True
            
        # Numeric constraints
        if pd.api.types.is_numeric_dtype(series):
            constraints['min_value'] = float(series.min())
            constraints['max_value'] = float(series.max())
            constraints['mean'] = float(series.mean())
            constraints['std'] = float(series.std())
            
        # String constraints
        elif pd.api.types.is_string_dtype(series):
            str_lengths = series.str.len()
            constraints['min_length'] = int(str_lengths.min())
            constraints['max_length'] = int(str_lengths.max())
            constraints['avg_length'] = float(str_lengths.mean())
            
        # Categorical constraints
        unique_vals = series.dropna().unique()
        if len(unique_vals) <= 20:  # Reasonable number for categories
            constraints['allowed_values'] = unique_vals.tolist()
            
        return constraints


class FileOperations:
    """File I/O operations with support for multiple formats."""
    
    @staticmethod
    def load_data(file_path: Union[str, Path], **kwargs) -> pd.DataFrame:
        """
        Load data from various file formats.
        
        Args:
            file_path: Path to the data file
            **kwargs: Additional arguments passed to pandas read functions
            
        Returns:
            Loaded DataFrame
        """
        file_path = Path(file_path)
        
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
            
        suffix = file_path.suffix.lower()
        
        try:
            if suffix == '.csv':
                return pd.read_csv(file_path, **kwargs)
            elif suffix in ['.xlsx', '.xls']:
                return pd.read_excel(file_path, **kwargs)
            elif suffix == '.json':
                return pd.read_json(file_path, **kwargs)
            elif suffix == '.parquet':
                return pd.read_parquet(file_path, **kwargs)
            elif suffix == '.pkl':
                return pd.read_pickle(file_path, **kwargs)
            else:
                raise ValueError(f"Unsupported file format: {suffix}")
                
        except Exception as e:
            logger.error(f"Error loading file {file_path}: {str(e)}")
            raise
    
    @staticmethod
    def save_data(df: pd.DataFrame, file_path: Union[str, Path], **kwargs) -> None:
        """
        Save DataFrame to various file formats.
        
        Args:
            df: DataFrame to save
            file_path: Output file path
            **kwargs: Additional arguments passed to pandas save functions
        """
        file_path = Path(file_path)
        file_path.parent.mkdir(parents=True, exist_ok=True)
        
        suffix = file_path.suffix.lower()
        
        try:
            if suffix == '.csv':
                df.to_csv(file_path, index=False, **kwargs)
            elif suffix in ['.xlsx', '.xls']:
                df.to_excel(file_path, index=False, **kwargs)
            elif suffix == '.json':
                df.to_json(file_path, **kwargs)
            elif suffix == '.parquet':
                df.to_parquet(file_path, **kwargs)
            elif suffix == '.pkl':
                df.to_pickle(file_path, **kwargs)
            else:
                raise ValueError(f"Unsupported file format: {suffix}")
                
            logger.info(f"Data saved to {file_path}")
            
        except Exception as e:
            logger.error(f"Error saving file {file_path}: {str(e)}")
            raise


class ConfigManager:
    """Configuration management for Validata components."""
    
    def __init__(self, config_path: Optional[Union[str, Path]] = None):
        self.config_path = Path(config_path) if config_path else None
        self.config = self._load_default_config()
        
        if self.config_path and self.config_path.exists():
            self._load_config_file()
    
    def _load_default_config(self) -> Dict[str, Any]:
        """Load default configuration."""
        return {
            'profiling': {
                'minimal': False,
                'explorative': True,
                'correlations': True,
                'missing_diagrams': True,
                'sample_size': None
            },
            'cleaning': {
                'handle_duplicates': True,
                'handle_missing': True,
                'detect_outliers': True,
                'fix_data_types': True
            },
            'validation': {
                'strict_mode': False,
                'statistical_tests': True,
                'custom_checks': True
            },
            'standardization': {
                'numerical_strategy': 'standard',
                'categorical_strategy': 'onehot',
                'date_format': 'iso'
            },
            'schema_generation': {
                'infer_constraints': True,
                'include_nullability': True,
                'include_relationships': False
            }
        }
    
    def _load_config_file(self) -> None:
        """Load configuration from file."""
        try:
            if self.config_path.suffix.lower() == '.yaml':
                with open(self.config_path, 'r') as f:
                    file_config = yaml.safe_load(f)
            elif self.config_path.suffix.lower() == '.json':
                with open(self.config_path, 'r') as f:
                    file_config = json.load(f)
            else:
                raise ValueError(f"Unsupported config format: {self.config_path.suffix}")
                
            # Deep merge with default config
            self._deep_merge(self.config, file_config)
            logger.info(f"Configuration loaded from {self.config_path}")
            
        except Exception as e:
            logger.warning(f"Error loading config file: {str(e)}")
            logger.info("Using default configuration")
    
    def _deep_merge(self, base_dict: Dict, update_dict: Dict) -> None:
        """Deep merge configuration dictionaries."""
        for key, value in update_dict.items():
            if key in base_dict and isinstance(base_dict[key], dict) and isinstance(value, dict):
                self._deep_merge(base_dict[key], value)
            else:
                base_dict[key] = value
    
    def get(self, section: str, key: str = None, default: Any = None) -> Any:
        """Get configuration value."""
        if key is None:
            return self.config.get(section, default)
        return self.config.get(section, {}).get(key, default)
    
    def set(self, section: str, key: str, value: Any) -> None:
        """Set configuration value."""
        if section not in self.config:
            self.config[section] = {}
        self.config[section][key] = value
    
    def save_config(self, file_path: Union[str, Path]) -> None:
        """Save current configuration to file."""
        file_path = Path(file_path)
        file_path.parent.mkdir(parents=True, exist_ok=True)
        
        if file_path.suffix.lower() == '.yaml':
            with open(file_path, 'w') as f:
                yaml.dump(self.config, f, default_flow_style=False)
        elif file_path.suffix.lower() == '.json':
            with open(file_path, 'w') as f:
                json.dump(self.config, f, indent=2)
        else:
            raise ValueError(f"Unsupported config format: {file_path.suffix}")
            
        logger.info(f"Configuration saved to {file_path}")


class OperationTracker:
    """Track operations and transformations applied to data with detailed performance metrics."""
    
    def __init__(self):
        self.operations = []
        self.start_time = None
        self.end_time = None
        self.start_memory = None
        self.peak_memory = None
    
    def start_operation(self, operation_name: str, **params) -> None:
        """Start tracking an operation with performance monitoring."""
        import time
        import psutil
        from datetime import datetime
        
        self.start_time = time.time()
        
        # Get memory usage
        try:
            process = psutil.Process()
            self.start_memory = process.memory_info().rss / 1024 / 1024  # MB
            self.peak_memory = self.start_memory
        except Exception:
            self.start_memory = None
            self.peak_memory = None
        
        operation = {
            'name': operation_name,
            'parameters': params,
            'start_time': self.start_time,
            'timestamp': datetime.now().isoformat(),
            'status': 'running',
            'performance': {
                'start_memory_mb': self.start_memory
            }
        }
        self.operations.append(operation)
        logger.info(f"Started operation: {operation_name}")
    
    def complete_operation(self, **results) -> None:
        """Complete the current operation with performance metrics."""
        import time
        import psutil
        
        self.end_time = time.time()
        
        if self.operations:
            current_op = self.operations[-1]
            duration = self.end_time - current_op['start_time']
            
            # Get final memory usage
            try:
                process = psutil.Process()
                end_memory = process.memory_info().rss / 1024 / 1024  # MB
                memory_delta = end_memory - (self.start_memory or 0)
            except Exception:
                end_memory = None
                memory_delta = None
            
            # Update operation record
            current_op['end_time'] = self.end_time
            current_op['duration'] = duration
            current_op['status'] = 'completed'
            current_op['results'] = results
            current_op['performance'].update({
                'duration_ms': round(duration * 1000, 2),
                'end_memory_mb': end_memory,
                'memory_delta_mb': round(memory_delta, 2) if memory_delta is not None else None,
                'rows_processed': results.get('rows_processed', 0)
            })
            
            logger.info(f"Completed operation: {current_op['name']} in {duration:.3f}s")
    
    def fail_operation(self, error_message: str) -> None:
        """Mark current operation as failed with performance data."""
        import time
        import psutil
        
        self.end_time = time.time()
        
        if self.operations:
            current_op = self.operations[-1]
            duration = self.end_time - current_op['start_time']
            
            # Get memory usage at failure
            try:
                process = psutil.Process()
                end_memory = process.memory_info().rss / 1024 / 1024  # MB
            except Exception:
                end_memory = None
            
            current_op['end_time'] = self.end_time
            current_op['duration'] = duration
            current_op['status'] = 'failed'
            current_op['error'] = error_message
            current_op['performance'].update({
                'duration_ms': round(duration * 1000, 2),
                'end_memory_mb': end_memory
            })
            
            logger.error(f"Failed operation: {current_op['name']} - {error_message}")
    
    def get_summary(self) -> Dict[str, Any]:
        """Get comprehensive summary of all operations with performance metrics."""
        total_operations = len(self.operations)
        successful_operations = len([op for op in self.operations if op['status'] == 'completed'])
        failed_operations = len([op for op in self.operations if op['status'] == 'failed'])
        total_duration = sum([op.get('duration', 0) for op in self.operations])
        
        # Calculate performance statistics
        completed_ops = [op for op in self.operations if op['status'] == 'completed']
        avg_duration = sum([op.get('duration', 0) for op in completed_ops]) / len(completed_ops) if completed_ops else 0
        total_rows = sum([op.get('results', {}).get('rows_processed', 0) for op in completed_ops])
        
        return {
            'total_operations': total_operations,
            'successful_operations': successful_operations,
            'failed_operations': failed_operations,
            'success_rate': successful_operations / total_operations if total_operations > 0 else 0,
            'total_duration': total_duration,
            'average_duration': avg_duration,
            'total_rows_processed': total_rows,
            'operations': self.operations
        }
    
    def get_current_metadata(self) -> Dict[str, Any]:
        """Get metadata for the current operation."""
        if not self.operations:
            return {}
        
        current_op = self.operations[-1]
        return {
            'operation': current_op['name'],
            'timestamp': current_op['timestamp'],
            'parameters': current_op['parameters'],
            'performance': current_op.get('performance', {})
        }


# Utility functions
def detect_delimiter(file_path: Union[str, Path], sample_size: int = 1024) -> str:
    """
    Detect the delimiter used in a CSV file.
    
    Args:
        file_path: Path to the CSV file
        sample_size: Number of bytes to read for detection
        
    Returns:
        Detected delimiter character
    """
    import csv
    
    with open(file_path, 'r', encoding='utf-8') as f:
        sample = f.read(sample_size)
        
    sniffer = csv.Sniffer()
    delimiter = sniffer.sniff(sample).delimiter
    
    return delimiter


def generate_sample_data(rows: int = 1000, columns: int = 5, data_types: Optional[List[str]] = None) -> pd.DataFrame:
    """
    Generate sample data for testing and demonstrations.
    
    Args:
        rows: Number of rows to generate
        columns: Number of columns to generate
        data_types: List of data types to include ('int', 'float', 'string', 'datetime', 'bool', 'categorical')
        
    Returns:
        Generated DataFrame
    """
    np.random.seed(42)  # For reproducibility
    
    if data_types is None:
        data_types = ['int', 'float', 'string', 'datetime', 'bool']
    
    data = {}
    
    for i in range(columns):
        col_name = f'column_{i+1}'
        data_type = data_types[i % len(data_types)]
        
        if data_type == 'int':
            data[col_name] = np.random.randint(0, 100, rows)
        elif data_type == 'float':
            data[col_name] = np.random.normal(50, 15, rows)
        elif data_type == 'string':
            data[col_name] = [f'text_{np.random.randint(1, 1000)}' for _ in range(rows)]
        elif data_type == 'datetime':
            start_date = pd.Timestamp('2020-01-01')
            data[col_name] = pd.date_range(start_date, periods=rows, freq='D')
        elif data_type == 'bool':
            data[col_name] = np.random.choice([True, False], rows)
        elif data_type == 'categorical':
            categories = ['A', 'B', 'C', 'D', 'E']
            data[col_name] = np.random.choice(categories, rows)
    
    return pd.DataFrame(data)