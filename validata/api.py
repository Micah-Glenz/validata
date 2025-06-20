"""
Main API Module for Validata

Provides the primary LLM-friendly interface that unifies all data validation capabilities.
This module serves as the main entry point for users, offering simple method wrappers
and workflow orchestration across all specialized modules.

Key Features:
- Unified interface for all data validation operations
- Method chaining support for workflow creation
- Configuration management and operation history
- Auto-save/load functionality for workflows
- Simple wrapper methods for complex operations
- Intelligent defaults and recommendations
- Comprehensive reporting and analytics
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Union, Tuple
from pathlib import Path
import logging
import json
from datetime import datetime

from .profiler import DataProfiler
from .cleaner import DataCleaner
from .standardizer import DataStandardizer
from .validator import DataValidator
from .schema_generator import SchemaGenerator
from .utils import ConfigManager, OperationTracker, FileOperations

logger = logging.getLogger(__name__)


class ValidataAPI:
    """
    LLM-friendly unified interface for all data wrangling operations.
    
    This is the main class that provides simple, chainable methods for:
    - Data profiling and quality assessment
    - Data cleaning and preprocessing
    - Data validation and quality checks
    - Data standardization and normalization
    - Schema generation and inference
    
    Example:
        >>> from wangler import WranglerAPI
        >>> wrangler = WranglerAPI()
        >>> 
        >>> # Simple usage
        >>> result = wrangler.profile_data("data.csv")
        >>> cleaned = wrangler.clean_data("data.csv")
        >>> 
        >>> # Method chaining
        >>> pipeline = (wrangler
        ...     .load_data("data.csv")
        ...     .clean_data()
        ...     .standardize_data()
        ...     .validate_data()
        ...     .generate_schema())
    """
    
    def __init__(self, config_path: Optional[Union[str, Path]] = None):
        """
        Initialize the WranglerAPI.
        
        Args:
            config_path: Optional path to configuration file
        """
        # Initialize configuration manager
        self.config_manager = ConfigManager(config_path)
        
        # Initialize specialized modules
        self.profiler = DataProfiler(self.config_manager)
        self.cleaner = DataCleaner(self.config_manager)
        self.standardizer = DataStandardizer(self.config_manager)
        self.validator = DataValidator(self.config_manager)
        self.schema_generator = SchemaGenerator(self.config_manager)
        
        # Initialize API-level tracking
        self.tracker = OperationTracker()
        self._current_data = None
        self._workflow_history = []
        self._auto_save = False
        self._workflow_name = None
        
        logger.info("WranglerAPI initialized successfully")
    
    def load_data(
        self,
        data_source: Union[pd.DataFrame, str, Path],
        **kwargs
    ) -> 'WranglerAPI':
        """
        Load data for processing. Supports method chaining.
        
        Args:
            data_source: DataFrame, file path, or data source
            **kwargs: Additional arguments for data loading
            
        Returns:
            Self for method chaining
            
        Example:
            >>> wrangler.load_data("data.csv").profile_data()
        """
        self.tracker.start_operation("load_data", source=str(data_source))
        
        try:
            if isinstance(data_source, pd.DataFrame):
                self._current_data = data_source.copy()
            else:
                self._current_data = FileOperations.load_data(data_source, **kwargs)
            
            self._record_workflow_step('load_data', {
                'source': str(data_source),
                'shape': self._current_data.shape,
                'columns': list(self._current_data.columns)
            })
            
            self.tracker.complete_operation(
                rows_loaded=len(self._current_data),
                columns_loaded=len(self._current_data.columns)
            )
            
            logger.info(f"Data loaded successfully: {self._current_data.shape}")
            return self
            
        except Exception as e:
            self.tracker.fail_operation(str(e))
            logger.error(f"Error loading data: {str(e)}")
            raise
    
    def profile_data(
        self,
        data: Optional[Union[pd.DataFrame, str, Path]] = None,
        title: str = "Data Profile",
        minimal: Optional[bool] = None,
        **kwargs
    ) -> Union[Dict[str, Any], 'WranglerAPI']:
        """
        Generate comprehensive data profile.
        
        Args:
            data: Data to profile (uses current data if None)
            title: Title for the profiling report
            minimal: Use minimal profiling for faster execution
            **kwargs: Additional arguments for profiling
            
        Returns:
            Profile results dict or self for chaining
            
        Example:
            >>> # Direct usage
            >>> profile = wrangler.profile_data("data.csv")
            >>> 
            >>> # Method chaining
            >>> wrangler.load_data("data.csv").profile_data(minimal=True)
        """
        data_to_use = self._get_data_to_use(data)
        
        result = self.profiler.profile_data(
            data_to_use,
            title=title,
            minimal=minimal,
            **kwargs
        )
        
        self._record_workflow_step('profile_data', {
            'title': title,
            'minimal': minimal,
            'summary': result['summary']
        })
        
        # Return result if called directly, self if chaining
        if data is not None:
            return result
        else:
            self._last_profile_result = result
            return self
    
    def clean_data(
        self,
        data: Optional[Union[pd.DataFrame, str, Path]] = None,
        strategy: str = 'auto',
        **kwargs
    ) -> Union[Dict[str, Any], 'WranglerAPI']:
        """
        Clean data using comprehensive cleaning methods.
        
        Args:
            data: Data to clean (uses current data if None)
            strategy: Cleaning strategy ('auto', 'conservative', 'aggressive')
            **kwargs: Additional arguments for cleaning
            
        Returns:
            Cleaning results dict or self for chaining
            
        Example:
            >>> # Direct usage
            >>> cleaned = wrangler.clean_data("data.csv", strategy='auto')
            >>> 
            >>> # Method chaining
            >>> wrangler.load_data("data.csv").clean_data().standardize_data()
        """
        data_to_use = self._get_data_to_use(data)
        
        # Configure cleaning based on strategy
        if strategy == 'conservative':
            config = {
                'handle_missing': False,
                'handle_duplicates': True,
                'fix_data_types': False,
                'detect_outliers': False,
                'clean_text': True
            }
        elif strategy == 'aggressive':
            config = {
                'handle_missing': True,
                'handle_duplicates': True,
                'fix_data_types': True,
                'detect_outliers': True,
                'clean_text': True
            }
        else:  # auto
            config = None
        
        result = self.cleaner.clean_all(data_to_use, config=config)
        
        self._record_workflow_step('clean_data', {
            'strategy': strategy,
            'summary': result['summary']
        })
        
        # Update current data if chaining
        if data is None:
            self._current_data = result['cleaned_data']
            return self
        else:
            return result
    
    def standardize_data(
        self,
        data: Optional[Union[pd.DataFrame, str, Path]] = None,
        method: str = 'auto',
        **kwargs
    ) -> Union[Dict[str, Any], 'WranglerAPI']:
        """
        Standardize data using normalization and encoding methods.
        
        Args:
            data: Data to standardize (uses current data if None)
            method: Standardization method ('auto', 'minimal', 'full')
            **kwargs: Additional arguments for standardization
            
        Returns:
            Standardization results dict or self for chaining
            
        Example:
            >>> # Direct usage
            >>> standardized = wrangler.standardize_data("data.csv")
            >>> 
            >>> # Method chaining
            >>> wrangler.load_data("data.csv").clean_data().standardize_data()
        """
        data_to_use = self._get_data_to_use(data)
        
        # Configure standardization based on method
        if method == 'minimal':
            config = {
                'standardize_numerical': False,
                'encode_categorical': True,
                'standardize_dates': True,
                'extract_date_features': False
            }
        elif method == 'full':
            config = {
                'standardize_numerical': True,
                'encode_categorical': True,
                'standardize_dates': True,
                'extract_date_features': True
            }
        else:  # auto
            config = None
        
        result = self.standardizer.standardize_all(data_to_use, config=config)
        
        self._record_workflow_step('standardize_data', {
            'method': method,
            'summary': result['summary']
        })
        
        # Update current data if chaining
        if data is None:
            self._current_data = result['standardized_data']
            return self
        else:
            return result
    
    def validate_data(
        self,
        data: Optional[Union[pd.DataFrame, str, Path]] = None,
        validation_type: str = 'comprehensive',
        **kwargs
    ) -> Union[Dict[str, Any], 'WranglerAPI']:
        """
        Validate data using schema and business rule validation.
        
        Args:
            data: Data to validate (uses current data if None)
            validation_type: Type of validation ('schema', 'business', 'statistical', 'comprehensive')
            **kwargs: Additional arguments for validation
            
        Returns:
            Validation results dict or self for chaining
            
        Example:
            >>> # Direct usage
            >>> validation = wrangler.validate_data("data.csv", validation_type='schema')
            >>> 
            >>> # Method chaining
            >>> wrangler.load_data("data.csv").clean_data().validate_data()
        """
        data_to_use = self._get_data_to_use(data)
        
        if validation_type == 'schema':
            # Use lazy validation to allow data type changes during processing
            result = self.validator.validate_schema(data_to_use, infer_schema=True, strict=False)
        elif validation_type == 'statistical':
            result = self.validator.statistical_validation(data_to_use)
        elif validation_type == 'comprehensive':
            # Perform multiple validation types
            schema_result = self.validator.validate_schema(data_to_use, infer_schema=True, strict=False)
            statistical_result = self.validator.statistical_validation(data_to_use)
            
            result = {
                'schema_validation': schema_result,
                'statistical_validation': statistical_result,
                'overall_passed': schema_result['validation_passed'],
                'combined_summary': {
                    'schema_errors': len(schema_result['errors']),
                    'statistical_issues': len(statistical_result.get('test_results', {}))
                }
            }
        else:
            raise ValueError(f"Unknown validation type: {validation_type}")
        
        self._record_workflow_step('validate_data', {
            'validation_type': validation_type,
            'passed': result.get('validation_passed', result.get('overall_passed', True))
        })
        
        # Return result if called directly, self if chaining
        if data is not None:
            return result
        else:
            self._last_validation_result = result
            return self
    
    def generate_schema(
        self,
        data: Optional[Union[pd.DataFrame, str, Path]] = None,
        table_name: str = "generated_table",
        schema_type: str = 'comprehensive',
        **kwargs
    ) -> Union[Dict[str, Any], 'WranglerAPI']:
        """
        Generate schema from data analysis.
        
        Args:
            data: Data to analyze (uses current data if None)
            table_name: Name for the generated table/model
            schema_type: Type of schema ('basic', 'comprehensive', 'with_models')
            **kwargs: Additional arguments for schema generation
            
        Returns:
            Schema results dict or self for chaining
            
        Example:
            >>> # Direct usage
            >>> schema = wrangler.generate_schema("data.csv", table_name="users")
            >>> 
            >>> # Method chaining
            >>> wrangler.load_data("data.csv").clean_data().generate_schema("users")
        """
        data_to_use = self._get_data_to_use(data)
        
        # Generate base schema
        result = self.schema_generator.infer_schema(
            data_to_use,
            table_name=table_name,
            infer_constraints=(schema_type in ['comprehensive', 'with_models']),
            include_relationships=(schema_type in ['comprehensive', 'with_models']),
            **kwargs
        )
        
        # Generate additional models if requested
        if schema_type == 'with_models':
            sqlmodel_result = self.schema_generator.generate_sqlmodel(
                result['schema'], table_name
            )
            pydantic_result = self.schema_generator.generate_pydantic_model(
                result['schema'], table_name
            )
            ddl_result = self.schema_generator.generate_database_ddl(
                result['schema'], table_name
            )
            
            result['generated_models'] = {
                'sqlmodel': sqlmodel_result,
                'pydantic': pydantic_result,
                'ddl': ddl_result
            }
        
        self._record_workflow_step('generate_schema', {
            'table_name': table_name,
            'schema_type': schema_type,
            'columns_generated': len(result['schema']['columns'])
        })
        
        # Return result if called directly, self if chaining
        if data is not None:
            return result
        else:
            self._last_schema_result = result
            return self
    
    def analyze_all(
        self,
        data: Union[pd.DataFrame, str, Path],
        table_name: str = "analysis_table"
    ) -> Dict[str, Any]:
        """
        Perform comprehensive analysis including profiling, cleaning, validation, and schema generation.
        
        Args:
            data: Data to analyze
            table_name: Name for schema generation
            
        Returns:
            Comprehensive analysis results
            
        Example:
            >>> analysis = wrangler.analyze_all("data.csv", table_name="customers")
            >>> print(analysis['summary'])
        """
        self.tracker.start_operation("analyze_all", table_name=table_name)
        
        try:
            # Load data
            if not isinstance(data, pd.DataFrame):
                data = FileOperations.load_data(data)
            
            analysis_results = {
                'original_data': data,
                'data_info': {
                    'shape': data.shape,
                    'columns': list(data.columns),
                    'dtypes': data.dtypes.to_dict()
                }
            }
            
            # 1. Profile data
            logger.info("Profiling data...")
            profile_result = self.profile_data(data, title=f"Analysis of {table_name}")
            analysis_results['profiling'] = profile_result
            
            # 2. Clean data
            logger.info("Cleaning data...")
            clean_result = self.clean_data(data, strategy='auto')
            analysis_results['cleaning'] = clean_result
            cleaned_data = clean_result['cleaned_data']
            
            # 3. Standardize data
            logger.info("Standardizing data...")
            standardize_result = self.standardize_data(cleaned_data, method='auto')
            analysis_results['standardization'] = standardize_result
            standardized_data = standardize_result['standardized_data']
            
            # 4. Validate data
            logger.info("Validating data...")
            validation_result = self.validate_data(standardized_data, validation_type='comprehensive')
            analysis_results['validation'] = validation_result
            
            # 5. Generate schema
            logger.info("Generating schema...")
            schema_result = self.generate_schema(standardized_data, table_name, schema_type='with_models')
            analysis_results['schema_generation'] = schema_result
            
            # Generate comprehensive summary
            analysis_results['summary'] = self._generate_analysis_summary(analysis_results)
            
            # Generate recommendations
            analysis_results['recommendations'] = self._generate_recommendations(analysis_results)
            
            self.tracker.complete_operation(
                phases_completed=5,
                final_data_shape=standardized_data.shape
            )
            
            logger.info(f"Comprehensive analysis completed for {table_name}")
            return analysis_results
            
        except Exception as e:
            self.tracker.fail_operation(str(e))
            logger.error(f"Error in comprehensive analysis: {str(e)}")
            raise
    
    def create_workflow(self, name: str) -> 'WranglerAPI':
        """
        Start a named workflow for tracking and potential replay.
        
        Args:
            name: Name for the workflow
            
        Returns:
            Self for method chaining
        """
        self._workflow_name = name
        self._workflow_history = []
        logger.info(f"Started workflow: {name}")
        return self
    
    def save_workflow(self, file_path: Union[str, Path]) -> str:
        """
        Save current workflow to file.
        
        Args:
            file_path: Path to save workflow
            
        Returns:
            Path to saved workflow file
        """
        workflow_data = {
            'name': self._workflow_name,
            'steps': self._workflow_history,
            'created_at': pd.Timestamp.now().isoformat(),
            'final_data_shape': self._current_data.shape if self._current_data is not None else None
        }
        
        file_path = Path(file_path)
        file_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(file_path, 'w') as f:
            json.dump(workflow_data, f, indent=2, default=str)
        
        logger.info(f"Workflow saved to {file_path}")
        return str(file_path)
    
    def get_current_data(self) -> Optional[pd.DataFrame]:
        """Get the current data being processed."""
        return self._current_data.copy() if self._current_data is not None else None
    
    def get_data_summary(self) -> Dict[str, Any]:
        """Get summary of current data."""
        if self._current_data is None:
            return {'error': 'No data loaded'}
        
        return {
            'shape': self._current_data.shape,
            'columns': list(self._current_data.columns),
            'dtypes': self._current_data.dtypes.to_dict(),
            'missing_values': self._current_data.isnull().sum().to_dict(),
            'memory_usage_mb': self._current_data.memory_usage(deep=True).sum() / 1024 / 1024
        }
    
    def export_results(
        self,
        file_path: Union[str, Path],
        include_data: bool = True,
        include_reports: bool = True
    ) -> str:
        """
        Export comprehensive results and reports.
        
        Args:
            file_path: Base path for exports
            include_data: Whether to export processed data
            include_reports: Whether to export analysis reports
            
        Returns:
            Directory path containing all exports
        """
        export_dir = Path(file_path)
        export_dir.mkdir(parents=True, exist_ok=True)
        
        exported_files = []
        
        # Export current data
        if include_data and self._current_data is not None:
            data_path = export_dir / 'processed_data.csv'
            self._current_data.to_csv(data_path, index=False)
            exported_files.append(str(data_path))
        
        # Export reports
        if include_reports:
            # Profiling report
            if hasattr(self, '_last_profile_result'):
                profile_path = export_dir / 'profiling_report.json'
                with open(profile_path, 'w') as f:
                    json.dump(self._last_profile_result, f, indent=2, default=str)
                exported_files.append(str(profile_path))
            
            # Validation report
            if hasattr(self, '_last_validation_result'):
                validation_path = export_dir / 'validation_report.json'
                with open(validation_path, 'w') as f:
                    json.dump(self._last_validation_result, f, indent=2, default=str)
                exported_files.append(str(validation_path))
            
            # Schema
            if hasattr(self, '_last_schema_result'):
                schema_path = export_dir / 'schema.json'
                with open(schema_path, 'w') as f:
                    json.dump(self._last_schema_result, f, indent=2, default=str)
                exported_files.append(str(schema_path))
            
            # Workflow history
            if self._workflow_history:
                workflow_path = export_dir / 'workflow.json'
                workflow_data = {
                    'name': self._workflow_name,
                    'steps': self._workflow_history,
                    'exported_at': pd.Timestamp.now().isoformat()
                }
                with open(workflow_path, 'w') as f:
                    json.dump(workflow_data, f, indent=2, default=str)
                exported_files.append(str(workflow_path))
        
        logger.info(f"Exported {len(exported_files)} files to {export_dir}")
        return str(export_dir)
    
    def _get_data_to_use(self, data: Optional[Union[pd.DataFrame, str, Path]]) -> pd.DataFrame:
        """Get data to use for operations."""
        if data is not None:
            if isinstance(data, pd.DataFrame):
                return data
            else:
                return FileOperations.load_data(data)
        elif self._current_data is not None:
            return self._current_data
        else:
            raise ValueError("No data provided and no current data available. Use load_data() first.")
    
    def _record_workflow_step(self, operation: str, details: Dict[str, Any]) -> None:
        """Record a workflow step."""
        step = {
            'operation': operation,
            'timestamp': pd.Timestamp.now().isoformat(),
            'details': details
        }
        self._workflow_history.append(step)
        
        if self._auto_save and self._workflow_name:
            auto_save_path = f"workflows/{self._workflow_name}_autosave.json"
            self.save_workflow(auto_save_path)
    
    def _generate_analysis_summary(self, analysis_results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate comprehensive analysis summary."""
        original_shape = analysis_results['data_info']['shape']
        
        # Get final data shape
        final_shape = None
        if 'standardization' in analysis_results:
            final_shape = analysis_results['standardization']['summary']['final_shape']
        elif 'cleaning' in analysis_results:
            final_shape = analysis_results['cleaning']['summary']['final_shape']
        else:
            final_shape = original_shape
        
        # Calculate data quality scores
        quality_scores = {}
        if 'profiling' in analysis_results:
            profile_data = analysis_results['profiling']
            missing_pct = profile_data['summary'].get('missing_cells_percent', 0)
            duplicate_pct = profile_data['summary'].get('duplicate_rows_percent', 0)
            
            quality_scores = {
                'completeness': max(0, 100 - missing_pct),
                'uniqueness': max(0, 100 - duplicate_pct),
                'overall': max(0, (200 - missing_pct - duplicate_pct) / 2)
            }
        
        # Validation summary
        validation_summary = {'passed': True, 'issues': []}
        if 'validation' in analysis_results:
            val_result = analysis_results['validation']
            if 'overall_passed' in val_result:
                validation_summary['passed'] = val_result['overall_passed']
            elif 'validation_passed' in val_result:
                validation_summary['passed'] = val_result['validation_passed']
        
        return {
            'data_transformation': {
                'original_shape': original_shape,
                'final_shape': final_shape,
                'rows_changed': final_shape[0] - original_shape[0] if final_shape else 0,
                'columns_changed': final_shape[1] - original_shape[1] if final_shape else 0
            },
            'data_quality': quality_scores,
            'validation_status': validation_summary,
            'operations_completed': [
                'profiling', 'cleaning', 'standardization', 'validation', 'schema_generation'
            ],
            'analysis_timestamp': pd.Timestamp.now().isoformat()
        }
    
    def _generate_recommendations(self, analysis_results: Dict[str, Any]) -> List[str]:
        """Generate recommendations based on analysis results."""
        recommendations = []
        
        # Data quality recommendations
        if 'profiling' in analysis_results:
            profile = analysis_results['profiling']['summary']
            
            if profile.get('missing_cells_percent', 0) > 10:
                recommendations.append("Consider investigating missing data patterns and implementing appropriate handling strategies")
            
            if profile.get('duplicate_rows_percent', 0) > 5:
                recommendations.append("Review duplicate removal strategy - high duplication rate detected")
            
            if profile.get('numeric_variables', 0) == 0:
                recommendations.append("No numeric variables detected - consider data type conversion if numeric analysis is needed")
        
        # Cleaning recommendations
        if 'cleaning' in analysis_results:
            clean_summary = analysis_results['cleaning']['summary']
            
            if clean_summary.get('total_rows_removed', 0) > len(analysis_results['original_data']) * 0.1:
                recommendations.append("Significant data loss during cleaning - review cleaning strategy")
        
        # Validation recommendations
        if 'validation' in analysis_results:
            val_result = analysis_results['validation']
            
            if not val_result.get('overall_passed', val_result.get('validation_passed', True)):
                recommendations.append("Data validation issues detected - review data quality before proceeding")
        
        # Schema recommendations
        if 'schema_generation' in analysis_results:
            schema = analysis_results['schema_generation']['schema']
            
            if not schema.get('constraints', {}).get('primary_key_candidates'):
                recommendations.append("No primary key candidates found - consider adding a unique identifier column")
            
            relationships = schema.get('relationships', {})
            if relationships.get('potential_foreign_keys'):
                recommendations.append("Potential foreign key relationships detected - consider implementing referential integrity")
        
        if not recommendations:
            recommendations.append("Data analysis completed successfully with no major issues detected")
        
        return recommendations
    
    def get_operation_summary(self) -> Dict[str, Any]:
        """Get summary of all operations performed across all modules."""
        return {
            'api_operations': self.tracker.get_summary(),
            'profiler_operations': self.profiler.get_operation_summary(),
            'cleaner_operations': self.cleaner.get_operation_summary(),
            'standardizer_operations': self.standardizer.get_operation_summary(),
            'validator_operations': self.validator.get_operation_summary(),
            'schema_generator_operations': self.schema_generator.get_operation_summary(),
            'workflow_history': self._workflow_history
        }


# Convenience functions for direct usage
def quick_profile(data: Union[pd.DataFrame, str, Path]) -> Dict[str, Any]:
    """Quick data profiling function."""
    api = WranglerAPI()
    return api.profile_data(data)


def quick_clean(data: Union[pd.DataFrame, str, Path]) -> pd.DataFrame:
    """Quick data cleaning function."""
    api = WranglerAPI()
    result = api.clean_data(data)
    return result['cleaned_data']


def quick_analyze(data: Union[pd.DataFrame, str, Path], table_name: str = "quick_analysis") -> Dict[str, Any]:
    """Quick comprehensive analysis function."""
    api = WranglerAPI()
    return api.analyze_all(data, table_name)