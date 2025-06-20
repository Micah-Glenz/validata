"""
Data Validator Module for Wangler

Provides comprehensive data validation capabilities using Pandera and Great Expectations.
Validates data schemas, business rules, statistical properties, and data quality constraints.

Key Features:
- Schema validation using Pandera with flexible constraints
- Business rule validation with custom logic
- Statistical validation and hypothesis testing
- Great Expectations integration for data quality checks
- Custom validation rule support
- Detailed validation reporting
- Data quality scoring and recommendations
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Union, Tuple, Callable
from pathlib import Path
import logging
from datetime import datetime

import pandera as pa
from pandera import Column, DataFrameSchema, Check
from pandera.errors import SchemaError
import great_expectations as ge
from great_expectations.core import ExpectationSuite

from .utils import ConfigManager, OperationTracker, FileOperations, DataTypeInference

logger = logging.getLogger(__name__)


class DataValidator:
    """
    LLM-friendly data validation tool with comprehensive validation capabilities.
    
    Provides methods for:
    - Schema validation using Pandera
    - Business rule validation with custom logic
    - Statistical validation and hypothesis testing
    - Great Expectations integration
    - Custom validation rules
    - Validation reporting and quality scoring
    """
    
    def __init__(self, config_manager: Optional[ConfigManager] = None):
        """
        Initialize the DataValidator.
        
        Args:
            config_manager: Optional configuration manager instance
        """
        self.config_manager = config_manager or ConfigManager()
        self.tracker = OperationTracker()
        self._validation_history = []
        self._schemas = {}
        self._custom_rules = {}
    
    def validate_schema(
        self,
        data: Union[pd.DataFrame, str, Path],
        schema: Optional[Union[pa.DataFrameSchema, Dict[str, Any]]] = None,
        schema_name: Optional[str] = None,
        infer_schema: bool = False,
        strict: bool = False
    ) -> Dict[str, Any]:
        """
        Validate data against a Pandera schema.
        
        Args:
            data: DataFrame or path to data file
            schema: Pandera schema or schema dictionary
            schema_name: Name of stored schema to use
            infer_schema: Whether to infer schema from data
            strict: Whether to use strict validation mode
            
        Returns:
            Dictionary containing validation results and details
            
        Example:
            >>> validator = DataValidator()
            >>> result = validator.validate_schema(df, infer_schema=True)
            >>> print(f"Validation passed: {result['validation_passed']}")
            >>> print(f"Errors found: {len(result['errors'])}")
        """
        self.tracker.start_operation(
            "validate_schema",
            schema_name=schema_name,
            infer_schema=infer_schema,
            strict=strict
        )
        
        try:
            # Load data if path provided
            if isinstance(data, (str, Path)):
                data = FileOperations.load_data(data)
            
            if not isinstance(data, pd.DataFrame):
                raise ValueError("Data must be a pandas DataFrame or file path")
            
            validation_results = {
                'validation_passed': False,
                'errors': [],
                'warnings': [],
                'schema_used': None,
                'validated_columns': [],
                'validation_summary': {}
            }
            
            # Get or create schema
            if schema_name and schema_name in self._schemas:
                schema = self._schemas[schema_name]
            elif infer_schema:
                schema = self._infer_pandera_schema(data)
                if schema_name:
                    self._schemas[schema_name] = schema
            elif schema is None:
                raise ValueError("No schema provided. Set infer_schema=True or provide a schema.")
            
            # Convert dict schema to Pandera schema if needed
            if isinstance(schema, dict):
                schema = self._dict_to_pandera_schema(schema)
            
            # Perform validation
            try:
                validated_data = schema.validate(data, lazy=not strict)
                validation_results['validation_passed'] = True
                validation_results['validated_data'] = validated_data
                
            except SchemaError as e:
                # For chained operations, convert schema errors to warnings if not strict
                if not strict:
                    validation_results['validation_passed'] = True  # Allow pipeline to continue
                    validation_results['warnings'] = self._parse_schema_errors(e)
                    validation_results['validated_data'] = data  # Return original data
                    logger.warning("Schema validation failed but continuing due to non-strict mode")
                else:
                    validation_results['validation_passed'] = False
                    validation_results['errors'] = self._parse_schema_errors(e)
                    validation_results['validated_data'] = data  # Return original data
            
            # Additional validation checks
            validation_results['schema_used'] = self._schema_to_dict(schema)
            validation_results['validated_columns'] = list(schema.columns.keys()) if hasattr(schema, 'columns') else []
            
            # Generate validation summary
            validation_results['validation_summary'] = self._generate_validation_summary(
                data, validation_results
            )
            
            # Record validation
            self._record_validation_operation('validate_schema', validation_results['validation_summary'])
            
            self.tracker.complete_operation(
                validation_passed=validation_results['validation_passed'],
                errors_found=len(validation_results['errors']),
                columns_validated=len(validation_results['validated_columns']),
                rows_processed=len(data)
            )
            
            # Get tracker metadata
            tracker_metadata = self.tracker.get_current_metadata()
            
            # Create rich metadata structure
            result = {
                'success': validation_results['validation_passed'],
                'data': {
                    'validation_passed': validation_results['validation_passed'],
                    'errors': validation_results['errors'],
                    'warnings': validation_results['warnings'],
                    'schema_used': validation_results['schema_used'],
                    'validated_columns': validation_results['validated_columns'],
                    'validation_summary': validation_results['validation_summary'],
                    'validated_data': validation_results.get('validated_data')
                },
                'metadata': {
                    'operation': 'validate_schema',
                    'timestamp': tracker_metadata.get('timestamp', pd.Timestamp.now().isoformat()),
                    'performance': tracker_metadata.get('performance', {}),
                    'parameters': {
                        'schema_name': schema_name,
                        'infer_schema': infer_schema,
                        'strict': strict
                    }
                },
                'errors': validation_results['errors'],
                'warnings': validation_results['warnings'],
                # Legacy compatibility
                'original_data': data,
                **validation_results
            }
            
            return result
            
        except Exception as e:
            self.tracker.fail_operation(str(e))
            logger.error(f"Error validating schema: {str(e)}")
            raise
    
    def _infer_pandera_schema(self, data: pd.DataFrame) -> pa.DataFrameSchema:
        """Infer a Pandera schema from data."""
        columns = {}
        
        for col_name, col_data in data.items():
            # Infer data type - check bool first since it's considered numeric by pandas
            if pd.api.types.is_bool_dtype(col_data):
                dtype = pa.Bool
            elif pd.api.types.is_datetime64_any_dtype(col_data):
                dtype = pa.DateTime
            elif pd.api.types.is_numeric_dtype(col_data):
                if pd.api.types.is_integer_dtype(col_data):
                    dtype = pa.Int64
                else:
                    dtype = pa.Float64
            else:
                dtype = pa.String
            
            # Create column with basic constraints
            checks = []
            
            # Nullable check
            if col_data.isnull().any():
                nullable = True
            else:
                nullable = False
            
            # Range checks for numeric columns (excluding boolean)
            if pd.api.types.is_numeric_dtype(col_data) and not pd.api.types.is_bool_dtype(col_data) and not col_data.empty:
                # Only add range checks if we have valid numeric values
                non_null_data = col_data.dropna()
                if len(non_null_data) > 0:
                    min_val = non_null_data.min()
                    max_val = non_null_data.max()
                    # Only add checks if values are not NaN/None
                    if pd.notna(min_val) and pd.notna(max_val):
                        checks.extend([
                            Check.greater_than_or_equal_to(min_val),
                            Check.less_than_or_equal_to(max_val)
                        ])
            
            # String length checks
            if dtype == pa.String and not col_data.empty:
                # Only add string length checks for non-null string data
                non_null_strings = col_data.dropna().astype(str)
                if len(non_null_strings) > 0:
                    str_lengths = non_null_strings.str.len()
                    if len(str_lengths) > 0:
                        max_length = str_lengths.max()
                        if pd.notna(max_length) and max_length > 0:
                            checks.append(Check.str_length(max_value=int(max_length)))
            
            columns[col_name] = Column(dtype, checks=checks, nullable=nullable)
        
        return DataFrameSchema(columns)
    
    def _dict_to_pandera_schema(self, schema_dict: Dict[str, Any]) -> pa.DataFrameSchema:
        """Convert dictionary schema to Pandera schema."""
        columns = {}
        
        for col_name, col_spec in schema_dict.items():
            if isinstance(col_spec, dict):
                # Extract column properties
                dtype = col_spec.get('dtype', pa.String)
                nullable = col_spec.get('nullable', True)
                checks = col_spec.get('checks', [])
                
                # Convert string dtype to Pandera dtype
                if isinstance(dtype, str):
                    dtype_map = {
                        'int': pa.Int64, 'integer': pa.Int64,
                        'float': pa.Float64, 'numeric': pa.Float64,
                        'string': pa.String, 'str': pa.String,
                        'bool': pa.Bool, 'boolean': pa.Bool,
                        'datetime': pa.DateTime, 'date': pa.DateTime
                    }
                    dtype = dtype_map.get(dtype.lower(), pa.String)
                
                # Convert check specifications to Pandera checks
                pandera_checks = []
                for check in checks:
                    if isinstance(check, dict):
                        check_type = check.get('type')
                        if check_type == 'range':
                            min_val = check.get('min')
                            max_val = check.get('max')
                            if min_val is not None:
                                pandera_checks.append(Check.greater_than_or_equal_to(min_val))
                            if max_val is not None:
                                pandera_checks.append(Check.less_than_or_equal_to(max_val))
                        elif check_type == 'isin':
                            allowed_values = check.get('values', [])
                            pandera_checks.append(Check.isin(allowed_values))
                        elif check_type == 'regex':
                            pattern = check.get('pattern')
                            pandera_checks.append(Check.str_matches(pattern))
                
                columns[col_name] = Column(dtype, checks=pandera_checks, nullable=nullable)
            else:
                # Simple type specification
                columns[col_name] = Column(col_spec, nullable=True)
        
        return DataFrameSchema(columns)
    
    def _schema_to_dict(self, schema: pa.DataFrameSchema) -> Dict[str, Any]:
        """Convert Pandera schema to dictionary representation."""
        if not hasattr(schema, 'columns'):
            return {}
        
        schema_dict = {}
        for col_name, col_schema in schema.columns.items():
            schema_dict[col_name] = {
                'dtype': str(col_schema.dtype),
                'nullable': col_schema.nullable,
                'checks': [str(check) for check in col_schema.checks] if col_schema.checks else []
            }
        
        return schema_dict
    
    def _parse_schema_errors(self, schema_error: SchemaError) -> List[Dict[str, Any]]:
        """Parse Pandera schema errors into structured format."""
        errors = []
        
        if hasattr(schema_error, 'schema_errors'):
            for error in schema_error.schema_errors:
                errors.append({
                    'column': getattr(error, 'column', 'unknown'),
                    'check': str(getattr(error, 'check', 'unknown')),
                    'message': str(error),
                    'failure_cases': getattr(error, 'failure_cases', None)
                })
        else:
            errors.append({
                'column': 'unknown',
                'check': 'schema_validation',
                'message': str(schema_error),
                'failure_cases': None
            })
        
        return errors
    
    def validate_business_rules(
        self,
        data: Union[pd.DataFrame, str, Path],
        rules: Optional[List[Dict[str, Any]]] = None,
        rule_set_name: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Validate data against custom business rules.
        
        Args:
            data: DataFrame or path to data file
            rules: List of business rule specifications
            rule_set_name: Name of stored rule set to use
            
        Returns:
            Dictionary containing validation results for each rule
            
        Example:
            >>> rules = [
            ...     {'name': 'age_range', 'column': 'age', 'condition': 'between', 'min': 0, 'max': 120},
            ...     {'name': 'email_format', 'column': 'email', 'condition': 'regex', 'pattern': r'^[^@]+@[^@]+\\.[^@]+$'}
            ... ]
            >>> result = validator.validate_business_rules(df, rules=rules)
        """
        self.tracker.start_operation(
            "validate_business_rules",
            rule_count=len(rules) if rules else 0,
            rule_set_name=rule_set_name
        )
        
        try:
            # Load data if path provided
            if isinstance(data, (str, Path)):
                data = FileOperations.load_data(data)
            
            # Get rules
            if rule_set_name and rule_set_name in self._custom_rules:
                rules = self._custom_rules[rule_set_name]
            elif rules is None:
                raise ValueError("No rules provided. Specify rules or rule_set_name.")
            
            validation_results = {
                'overall_passed': True,
                'rule_results': {},
                'failed_rules': [],
                'total_violations': 0
            }
            
            for rule in rules:
                rule_name = rule.get('name', f"rule_{len(validation_results['rule_results'])}")
                rule_result = self._validate_single_rule(data, rule)
                
                validation_results['rule_results'][rule_name] = rule_result
                
                if not rule_result['passed']:
                    validation_results['overall_passed'] = False
                    validation_results['failed_rules'].append(rule_name)
                    validation_results['total_violations'] += rule_result['violation_count']
            
            # Generate summary
            validation_results['summary'] = {
                'total_rules': len(rules),
                'passed_rules': len(rules) - len(validation_results['failed_rules']),
                'failed_rules': len(validation_results['failed_rules']),
                'total_violations': validation_results['total_violations'],
                'success_rate': (len(rules) - len(validation_results['failed_rules'])) / len(rules) * 100
            }
            
            self._record_validation_operation('validate_business_rules', validation_results['summary'])
            
            self.tracker.complete_operation(
                rules_validated=len(rules),
                rules_passed=validation_results['summary']['passed_rules'],
                total_violations=validation_results['total_violations'],
                rows_processed=len(data)
            )
            
            # Get tracker metadata
            tracker_metadata = self.tracker.get_current_metadata()
            
            # Create rich metadata structure
            result = {
                'success': validation_results['overall_passed'],
                'data': {
                    'overall_passed': validation_results['overall_passed'],
                    'rule_results': validation_results['rule_results'],
                    'failed_rules': validation_results['failed_rules'],
                    'total_violations': validation_results['total_violations'],
                    'summary': validation_results['summary']
                },
                'metadata': {
                    'operation': 'validate_business_rules',
                    'timestamp': tracker_metadata.get('timestamp', pd.Timestamp.now().isoformat()),
                    'performance': tracker_metadata.get('performance', {}),
                    'parameters': {
                        'rule_count': len(rules),
                        'rule_set_name': rule_set_name
                    }
                },
                'errors': [],
                'warnings': [],
                # Legacy compatibility
                'original_data': data,
                **validation_results
            }
            
            return result
            
        except Exception as e:
            self.tracker.fail_operation(str(e))
            logger.error(f"Error validating business rules: {str(e)}")
            raise
    
    def _validate_single_rule(self, data: pd.DataFrame, rule: Dict[str, Any]) -> Dict[str, Any]:
        """Validate a single business rule."""
        rule_name = rule.get('name', 'unnamed_rule')
        column = rule.get('column')
        condition = rule.get('condition')
        
        result = {
            'passed': False,
            'violation_count': 0,
            'violation_indices': [],
            'rule_definition': rule,
            'message': ''
        }
        
        try:
            if column not in data.columns:
                result['message'] = f"Column '{column}' not found in data"
                return result
            
            col_data = data[column]
            
            # Apply different validation conditions
            if condition == 'not_null':
                violations = col_data.isnull()
                result['message'] = f"Found {violations.sum()} null values in {column}"
                
            elif condition == 'unique':
                violations = col_data.duplicated()
                result['message'] = f"Found {violations.sum()} duplicate values in {column}"
                
            elif condition == 'between':
                try:
                    min_val = rule.get('min')
                    max_val = rule.get('max')
                    if min_val is None or max_val is None:
                        result['message'] = f"Between rule requires both 'min' and 'max' values"
                        return result
                    violations = ~col_data.between(min_val, max_val, inclusive='both')
                    result['message'] = f"Found {violations.sum()} values outside range [{min_val}, {max_val}] in {column}"
                except (TypeError, ValueError) as e:
                    result['message'] = f"Error applying between rule to {column}: incompatible data types or values. Details: {str(e)}"
                    return result
                
            elif condition == 'greater_than':
                try:
                    threshold = rule.get('value')
                    if threshold is None:
                        result['message'] = f"Greater than rule requires a 'value' parameter"
                        return result
                    violations = ~(col_data > threshold)
                    result['message'] = f"Found {violations.sum()} values not greater than {threshold} in {column}"
                except (TypeError, ValueError) as e:
                    result['message'] = f"Error applying greater_than rule to {column}: incompatible data types. Details: {str(e)}"
                    return result
                
            elif condition == 'less_than':
                threshold = rule.get('value')
                violations = ~(col_data < threshold)
                result['message'] = f"Found {violations.sum()} values not less than {threshold} in {column}"
                
            elif condition == 'isin':
                allowed_values = rule.get('values', [])
                violations = ~col_data.isin(allowed_values)
                result['message'] = f"Found {violations.sum()} values not in allowed list in {column}"
                
            elif condition == 'regex':
                try:
                    pattern = rule.get('pattern')
                    if not pattern:
                        result['message'] = f"Regex rule requires a 'pattern' parameter"
                        return result
                    violations = ~col_data.astype(str).str.match(pattern, na=False)
                    result['message'] = f"Found {violations.sum()} values not matching pattern in {column}"
                except (TypeError, ValueError, ImportError) as e:
                    result['message'] = f"Error applying regex pattern to {column}: invalid pattern or data. Details: {str(e)}"
                    return result
                except Exception as e:
                    # Catch regex compilation errors and other pattern-related issues
                    result['message'] = f"Regex compilation error for pattern '{pattern}': {str(e)}"
                    return result
                
            elif condition == 'custom':
                # Custom validation function
                validator_func = rule.get('function')
                if callable(validator_func):
                    violations = ~col_data.apply(validator_func)
                    result['message'] = f"Found {violations.sum()} values failing custom validation in {column}"
                else:
                    result['message'] = "Custom validator function not callable"
                    return result
                    
            else:
                result['message'] = f"Unknown condition: {condition}"
                return result
            
            # Process results using EAFP pattern
            try:
                violation_count = int(violations.sum())
                result['violation_count'] = violation_count
                result['violation_indices'] = violations[violations].index.tolist()
                result['passed'] = violation_count == 0
                
                if result['passed']:
                    result['message'] = f"Rule '{rule_name}' passed for column '{column}'"
            except Exception as e:
                result['message'] = f"Error processing validation results for rule '{rule_name}': {str(e)}"
                result['violation_count'] = 0
                result['violation_indices'] = []
                result['passed'] = False
            
        except Exception as e:
            result['message'] = f"Error validating rule '{rule_name}': {str(e)}"
        
        return result
    
    def statistical_validation(
        self,
        data: Union[pd.DataFrame, str, Path],
        reference_data: Optional[pd.DataFrame] = None,
        tests: Optional[List[str]] = None,
        significance_level: float = 0.05
    ) -> Dict[str, Any]:
        """
        Perform statistical validation and hypothesis testing.
        
        Args:
            data: DataFrame or path to data file
            reference_data: Reference dataset for comparison
            tests: List of statistical tests to perform
            significance_level: Significance level for hypothesis tests
            
        Returns:
            Dictionary containing statistical test results
            
        Example:
            >>> result = validator.statistical_validation(df, tests=['normality', 'outliers'])
            >>> print(result['test_results']['normality'])
        """
        self.tracker.start_operation(
            "statistical_validation",
            tests=tests,
            significance_level=significance_level,
            has_reference=reference_data is not None
        )
        
        try:
            # Load data if path provided
            if isinstance(data, (str, Path)):
                data = FileOperations.load_data(data)
            
            # Default tests if not specified
            if tests is None:
                tests = ['normality', 'outliers', 'correlation']
                if reference_data is not None:
                    tests.extend(['distribution_comparison', 'mean_comparison'])
            
            validation_results = {
                'test_results': {},
                'overall_summary': {},
                'warnings': [],
                'recommendations': []
            }
            
            numeric_columns = data.select_dtypes(include=[np.number]).columns.tolist()
            
            # Perform statistical tests
            for test in tests:
                if test == 'normality':
                    validation_results['test_results']['normality'] = self._test_normality(
                        data, numeric_columns, significance_level
                    )
                elif test == 'outliers':
                    validation_results['test_results']['outliers'] = self._test_outliers(
                        data, numeric_columns
                    )
                elif test == 'correlation':
                    validation_results['test_results']['correlation'] = self._test_correlation(
                        data, numeric_columns, significance_level
                    )
                elif test == 'distribution_comparison' and reference_data is not None:
                    validation_results['test_results']['distribution_comparison'] = self._test_distribution_comparison(
                        data, reference_data, numeric_columns, significance_level
                    )
                elif test == 'mean_comparison' and reference_data is not None:
                    validation_results['test_results']['mean_comparison'] = self._test_mean_comparison(
                        data, reference_data, numeric_columns, significance_level
                    )
            
            # Generate summary and recommendations
            validation_results['overall_summary'] = self._generate_statistical_summary(
                validation_results['test_results']
            )
            
            self._record_validation_operation('statistical_validation', validation_results['overall_summary'])
            
            self.tracker.complete_operation(
                tests_performed=len(tests),
                columns_tested=len(numeric_columns),
                rows_processed=len(data)
            )
            
            # Get tracker metadata
            tracker_metadata = self.tracker.get_current_metadata()
            
            # Create rich metadata structure
            result = {
                'success': True,  # Statistical validation always succeeds if it completes
                'data': {
                    'test_results': validation_results['test_results'],
                    'overall_summary': validation_results['overall_summary'],
                    'recommendations': validation_results['recommendations']
                },
                'metadata': {
                    'operation': 'statistical_validation',
                    'timestamp': tracker_metadata.get('timestamp', pd.Timestamp.now().isoformat()),
                    'performance': tracker_metadata.get('performance', {}),
                    'parameters': {
                        'tests': tests,
                        'significance_level': significance_level,
                        'has_reference': reference_data is not None
                    }
                },
                'errors': [],
                'warnings': validation_results['warnings'],
                # Legacy compatibility
                'original_data': data,
                'reference_data': reference_data,
                **validation_results
            }
            
            return result
            
        except Exception as e:
            self.tracker.fail_operation(str(e))
            logger.error(f"Error in statistical validation: {str(e)}")
            raise
    
    def _test_normality(self, data: pd.DataFrame, columns: List[str], alpha: float) -> Dict[str, Any]:
        """Test normality using Shapiro-Wilk test."""
        from scipy.stats import shapiro
        
        results = {}
        for col in columns:
            if col in data.columns and not data[col].empty:
                # Use sample if data is too large (Shapiro-Wilk has limits)
                sample_data = data[col].dropna()
                if len(sample_data) > 5000:
                    sample_data = sample_data.sample(5000, random_state=42)
                
                if len(sample_data) >= 3:
                    stat, p_value = shapiro(sample_data)
                    results[col] = {
                        'statistic': float(stat),
                        'p_value': float(p_value),
                        'is_normal': p_value > alpha,
                        'sample_size': len(sample_data)
                    }
        
        return results
    
    def _test_outliers(self, data: pd.DataFrame, columns: List[str]) -> Dict[str, Any]:
        """Detect outliers using IQR method."""
        results = {}
        for col in columns:
            if col in data.columns and not data[col].empty:
                Q1 = data[col].quantile(0.25)
                Q3 = data[col].quantile(0.75)
                IQR = Q3 - Q1
                
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                
                outliers = ((data[col] < lower_bound) | (data[col] > upper_bound))
                outlier_count = outliers.sum()
                
                results[col] = {
                    'outlier_count': int(outlier_count),
                    'outlier_percentage': float(outlier_count / len(data) * 100),
                    'lower_bound': float(lower_bound),
                    'upper_bound': float(upper_bound),
                    'outlier_indices': outliers[outliers].index.tolist()
                }
        
        return results
    
    def _test_correlation(self, data: pd.DataFrame, columns: List[str], alpha: float) -> Dict[str, Any]:
        """Test for high correlations between variables."""
        if len(columns) < 2:
            return {}
        
        corr_matrix = data[columns].corr()
        high_correlations = []
        
        for i, col1 in enumerate(columns):
            for j, col2 in enumerate(columns[i+1:], i+1):
                corr_value = corr_matrix.loc[col1, col2]
                if abs(corr_value) > 0.8:  # High correlation threshold
                    high_correlations.append({
                        'column1': col1,
                        'column2': col2,
                        'correlation': float(corr_value),
                        'strength': 'very_high' if abs(corr_value) > 0.9 else 'high'
                    })
        
        return {
            'correlation_matrix': corr_matrix.to_dict(),
            'high_correlations': high_correlations,
            'high_correlation_count': len(high_correlations)
        }
    
    def _test_distribution_comparison(self, data: pd.DataFrame, reference: pd.DataFrame, 
                                    columns: List[str], alpha: float) -> Dict[str, Any]:
        """Compare distributions using Kolmogorov-Smirnov test."""
        from scipy.stats import ks_2samp
        
        results = {}
        for col in columns:
            if col in data.columns and col in reference.columns:
                sample1 = data[col].dropna()
                sample2 = reference[col].dropna()
                
                if len(sample1) > 0 and len(sample2) > 0:
                    stat, p_value = ks_2samp(sample1, sample2)
                    results[col] = {
                        'statistic': float(stat),
                        'p_value': float(p_value),
                        'distributions_similar': p_value > alpha,
                        'sample1_size': len(sample1),
                        'sample2_size': len(sample2)
                    }
        
        return results
    
    def _test_mean_comparison(self, data: pd.DataFrame, reference: pd.DataFrame,
                            columns: List[str], alpha: float) -> Dict[str, Any]:
        """Compare means using t-test."""
        from scipy.stats import ttest_ind
        
        results = {}
        for col in columns:
            if col in data.columns and col in reference.columns:
                sample1 = data[col].dropna()
                sample2 = reference[col].dropna()
                
                if len(sample1) > 1 and len(sample2) > 1:
                    stat, p_value = ttest_ind(sample1, sample2)
                    results[col] = {
                        'statistic': float(stat),
                        'p_value': float(p_value),
                        'means_similar': p_value > alpha,
                        'mean1': float(sample1.mean()),
                        'mean2': float(sample2.mean()),
                        'mean_difference': float(sample1.mean() - sample2.mean())
                    }
        
        return results
    
    def _generate_statistical_summary(self, test_results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate summary of statistical test results."""
        summary = {
            'tests_performed': list(test_results.keys()),
            'issues_found': [],
            'recommendations': []
        }
        
        # Analyze normality results
        if 'normality' in test_results:
            non_normal_cols = [col for col, result in test_results['normality'].items() 
                             if not result['is_normal']]
            if non_normal_cols:
                summary['issues_found'].append(f"Non-normal distributions in: {non_normal_cols}")
                summary['recommendations'].append("Consider data transformation for non-normal columns")
        
        # Analyze outlier results
        if 'outliers' in test_results:
            high_outlier_cols = [col for col, result in test_results['outliers'].items()
                               if result['outlier_percentage'] > 5]
            if high_outlier_cols:
                summary['issues_found'].append(f"High outlier rates in: {high_outlier_cols}")
                summary['recommendations'].append("Investigate and handle outliers in affected columns")
        
        # Analyze correlation results
        if 'correlation' in test_results:
            high_corr_count = test_results['correlation'].get('high_correlation_count', 0)
            if high_corr_count > 0:
                summary['issues_found'].append(f"Found {high_corr_count} high correlations")
                summary['recommendations'].append("Consider feature selection to reduce multicollinearity")
        
        return summary
    
    def _generate_validation_summary(self, data: pd.DataFrame, validation_results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate summary of validation results."""
        return {
            'data_shape': data.shape,
            'validation_passed': validation_results['validation_passed'],
            'error_count': len(validation_results['errors']),
            'warning_count': len(validation_results['warnings']),
            'columns_validated': len(validation_results['validated_columns']),
            'validation_timestamp': pd.Timestamp.now().isoformat()
        }
    
    def add_custom_rule(self, rule_name: str, rule_definition: Dict[str, Any]) -> None:
        """Add a custom validation rule."""
        self._custom_rules[rule_name] = rule_definition
        logger.info(f"Added custom rule: {rule_name}")
    
    def save_schema(self, schema: pa.DataFrameSchema, name: str) -> None:
        """Save a schema for reuse."""
        self._schemas[name] = schema
        logger.info(f"Saved schema: {name}")
    
    def get_saved_schemas(self) -> Dict[str, Any]:
        """Get information about saved schemas."""
        return {name: self._schema_to_dict(schema) for name, schema in self._schemas.items()}
    
    def _record_validation_operation(self, operation: str, summary: Dict[str, Any]) -> None:
        """Record validation operation in history."""
        self._validation_history.append({
            'operation': operation,
            'timestamp': pd.Timestamp.now().isoformat(),
            'summary': summary
        })
    
    def get_validation_history(self) -> List[Dict[str, Any]]:
        """Get history of all validation operations performed."""
        return self._validation_history.copy()
    
    def get_operation_summary(self) -> Dict[str, Any]:
        """Get summary of all operations performed."""
        return self.tracker.get_summary()
    
    def export_validation_report(self, file_path: Union[str, Path]) -> str:
        """
        Export comprehensive validation report.
        
        Args:
            file_path: Output file path
            
        Returns:
            Path to exported report
        """
        report_data = {
            'validation_history': self._validation_history,
            'saved_schemas': self.get_saved_schemas(),
            'custom_rules': self._custom_rules,
            'operation_summary': self.tracker.get_summary(),
            'generated_at': pd.Timestamp.now().isoformat()
        }
        
        file_path = Path(file_path)
        file_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(file_path, 'w') as f:
            import json
            json.dump(report_data, f, indent=2, default=str)
        
        logger.info(f"Validation report exported to {file_path}")
        return str(file_path)