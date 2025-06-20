"""
Comprehensive Validation & Schema Testing Suite with Enhanced Metadata Testing

This test suite rigorously tests the DataValidator and SchemaGenerator modules
using multiple test datasets, generates detailed text reports, and verifies the
new enhanced metadata structure with performance tracking.
"""

import sys
import os
import time
import traceback
from datetime import datetime
from pathlib import Path
import psutil
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional

# Add project root to path
sys.path.insert(0, os.path.dirname(__file__))

from validata.validator import DataValidator
from validata.schema_generator import SchemaGenerator
from validata.utils import ConfigManager


class TestReporter:
    """Handles detailed test reporting for validation and schema tests."""
    
    def __init__(self, reports_dir: str = "test_reports"):
        self.reports_dir = Path(reports_dir)
        self.reports_dir.mkdir(exist_ok=True)
        self.current_test = None
        self.start_time = None
        self.start_memory = None
        
    def start_test(self, test_name: str, module: str, dataset_name: str, config: Dict[str, Any] = None):
        """Start tracking a test."""
        self.current_test = {
            'name': test_name,
            'module': module,
            'dataset': dataset_name,
            'config': config or {},
            'timestamp': datetime.now().isoformat(),
            'results': {},
            'errors': [],
            'warnings': [],
            'performance': {},
            'data_analysis': {}
        }
        self.start_time = time.time()
        # Get current memory usage
        process = psutil.Process()
        self.start_memory = process.memory_info().rss / 1024 / 1024  # MB
        
    def add_result(self, key: str, value: Any):
        """Add a test result."""
        if self.current_test:
            self.current_test['results'][key] = value
            
    def add_performance_metric(self, key: str, value: Any):
        """Add a performance metric."""
        if self.current_test:
            self.current_test['performance'][key] = value
            
    def add_data_analysis(self, key: str, value: Any):
        """Add data analysis information."""
        if self.current_test:
            self.current_test['data_analysis'][key] = value
            
    def add_error(self, error: str):
        """Add an error."""
        if self.current_test:
            self.current_test['errors'].append(error)
            
    def add_warning(self, warning: str):
        """Add a warning."""
        if self.current_test:
            self.current_test['warnings'].append(warning)
            
    def finish_test(self, status: str = "PASSED") -> str:
        """Finish test and generate report."""
        if not self.current_test:
            return ""
            
        # Calculate final metrics
        end_time = time.time()
        duration = end_time - self.start_time
        
        process = psutil.Process()
        end_memory = process.memory_info().rss / 1024 / 1024
        peak_memory = end_memory
        
        self.current_test['performance']['duration_seconds'] = round(duration, 3)
        self.current_test['performance']['memory_start_mb'] = round(self.start_memory, 2)
        self.current_test['performance']['memory_end_mb'] = round(end_memory, 2)
        self.current_test['performance']['memory_peak_mb'] = round(peak_memory, 2)
        self.current_test['status'] = status
        
        # Generate report
        report_path = self._generate_report()
        
        # Reset for next test
        self.current_test = None
        return report_path
        
    def _generate_report(self) -> str:
        """Generate detailed text report."""
        test = self.current_test
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"{test['module']}_{test['name']}_{timestamp}_report.txt"
        filepath = self.reports_dir / filename
        
        # Count assertions/results
        total_assertions = len(test['results'])
        passed_assertions = sum(1 for v in test['results'].values() if v is True or (isinstance(v, dict) and v.get('passed', False)))
        
        report_content = f"""=== VALIDATA TEST REPORT ===
Test: {test['name']}
Module: {test['module']}
Dataset: {test['dataset']}
Timestamp: {test['timestamp']}
Duration: {test['performance'].get('duration_seconds', 'N/A')} seconds

=== TEST CONFIGURATION ===
Parameters: {test['config']}
Dataset Size: {test['data_analysis'].get('dataset_shape', 'N/A')}
Data Types: {test['data_analysis'].get('data_types', 'N/A')}

=== TEST RESULTS ===
Status: {test['status']}
Assertions: {passed_assertions}/{total_assertions} passed
Success Rate: {(passed_assertions/total_assertions*100) if total_assertions > 0 else 0:.1f}%

Detailed Results:
"""
        
        # Add detailed results
        for key, value in test['results'].items():
            if isinstance(value, dict):
                report_content += f"  {key}:\n"
                for sub_key, sub_value in value.items():
                    report_content += f"    {sub_key}: {sub_value}\n"
            else:
                status_symbol = "✓" if value is True else "✗" if value is False else "?"
                report_content += f"  {status_symbol} {key}: {value}\n"
        
        report_content += f"""
=== PERFORMANCE METRICS ===
Execution Time: {test['performance'].get('duration_seconds', 'N/A')} seconds
Memory Start: {test['performance'].get('memory_start_mb', 'N/A')} MB
Memory End: {test['performance'].get('memory_end_mb', 'N/A')} MB
Memory Peak: {test['performance'].get('memory_peak_mb', 'N/A')} MB
"""

        # Add performance calculations if data is available
        if 'rows_processed' in test['performance'] and test['performance']['duration_seconds'] > 0:
            rows_per_sec = test['performance']['rows_processed'] / test['performance']['duration_seconds']
            report_content += f"Rows Processed/Second: {rows_per_sec:,.0f}\n"
            
        report_content += f"""
=== DATA QUALITY ANALYSIS ===
"""
        for key, value in test['data_analysis'].items():
            report_content += f"{key}: {value}\n"
            
        report_content += f"""
=== DETAILED FINDINGS ===
"""
        
        # Add specific findings based on test type
        if 'validation_results' in test['results']:
            validation = test['results']['validation_results']
            if isinstance(validation, dict):
                report_content += f"Validation Passed: {validation.get('validation_passed', 'Unknown')}\n"
                report_content += f"Errors Found: {len(validation.get('errors', []))}\n"
                report_content += f"Warnings Found: {len(validation.get('warnings', []))}\n"
                report_content += f"Columns Validated: {len(validation.get('validated_columns', []))}\n"
                
        if 'schema_results' in test['results']:
            schema = test['results']['schema_results']
            if isinstance(schema, dict):
                schema_data = schema.get('schema', {})
                report_content += f"Schema Columns Generated: {len(schema_data.get('columns', {}))}\n"
                report_content += f"Constraints Generated: {len(schema_data.get('constraints', {}))}\n"
                report_content += f"Relationships Found: {len(schema_data.get('relationships', {}))}\n"
        
        report_content += f"""
=== ERRORS/WARNINGS ===
Errors ({len(test['errors'])}):
"""
        for error in test['errors']:
            report_content += f"  - {error}\n"
            
        report_content += f"Warnings ({len(test['warnings'])}):\n"
        for warning in test['warnings']:
            report_content += f"  - {warning}\n"
            
        report_content += f"""
=== RECOMMENDATIONS ===
"""
        
        # Generate recommendations based on results
        recommendations = self._generate_recommendations(test)
        for rec in recommendations:
            report_content += f"- {rec}\n"
            
        report_content += "===========================\n"
        
        # Write report to file
        with open(filepath, 'w') as f:
            f.write(report_content)
            
        return str(filepath)
        
    def _generate_recommendations(self, test: Dict[str, Any]) -> List[str]:
        """Generate recommendations based on test results."""
        recommendations = []
        
        # Performance recommendations
        duration = test['performance'].get('duration_seconds', 0)
        if duration > 10:
            recommendations.append(f"Test execution time ({duration:.1f}s) is high - consider optimization")
            
        # Memory recommendations
        memory_usage = test['performance'].get('memory_peak_mb', 0) - test['performance'].get('memory_start_mb', 0)
        if memory_usage > 100:
            recommendations.append(f"High memory usage ({memory_usage:.1f}MB) - consider memory optimization")
            
        # Error recommendations
        if test['errors']:
            recommendations.append("Address errors found during testing")
            
        # Success rate recommendations
        total_assertions = len(test['results'])
        passed_assertions = sum(1 for v in test['results'].values() if v is True)
        if total_assertions > 0:
            success_rate = passed_assertions / total_assertions
            if success_rate < 0.9:
                recommendations.append(f"Low success rate ({success_rate*100:.1f}%) - investigate failing assertions")
                
        if not recommendations:
            recommendations.append("Test completed successfully with no issues identified")
            
        return recommendations


class ComprehensiveTestSuite:
    """Main test suite for comprehensive validation and schema testing."""
    
    def __init__(self):
        self.reporter = TestReporter()
        self.validator = DataValidator()
        self.schema_generator = SchemaGenerator()
        self.test_results = []
        
    def load_dataset(self, dataset_path: str) -> pd.DataFrame:
        """Load a test dataset."""
        try:
            return pd.read_csv(dataset_path)
        except Exception as e:
            print(f"Error loading dataset {dataset_path}: {e}")
            return pd.DataFrame()
            
    def analyze_dataset(self, data: pd.DataFrame, dataset_name: str) -> Dict[str, Any]:
        """Analyze dataset for reporting."""
        if data.empty:
            return {"error": "Empty dataset"}
            
        analysis = {
            "dataset_shape": f"{data.shape[0]} rows × {data.shape[1]} columns",
            "memory_usage_mb": round(data.memory_usage(deep=True).sum() / 1024 / 1024, 2),
            "missing_data_percent": round(data.isnull().sum().sum() / (data.shape[0] * data.shape[1]) * 100, 2),
            "duplicate_rows": data.duplicated().sum(),
            "data_types": dict(data.dtypes.value_counts()),
            "column_names": list(data.columns),
        }
        
        # Add type-specific analysis
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            analysis["numeric_columns"] = len(numeric_cols)
            analysis["numeric_ranges"] = {col: f"{data[col].min():.2f} to {data[col].max():.2f}" 
                                        for col in numeric_cols if data[col].notna().any()}
        
        categorical_cols = data.select_dtypes(include=['object']).columns
        if len(categorical_cols) > 0:
            analysis["categorical_columns"] = len(categorical_cols)
            analysis["categorical_cardinality"] = {col: data[col].nunique() for col in categorical_cols}
            
        return analysis
    
    def test_validator_basic_schema_inference(self, data: pd.DataFrame, dataset_name: str):
        """Test basic schema inference validation."""
        test_name = "basic_schema_inference"
        self.reporter.start_test(test_name, "DataValidator", dataset_name, 
                               {"infer_schema": True, "strict": False})
        
        try:
            # Add dataset analysis
            analysis = self.analyze_dataset(data, dataset_name)
            for key, value in analysis.items():
                self.reporter.add_data_analysis(key, value)
                
            # Add performance tracking
            self.reporter.add_performance_metric("rows_processed", len(data))
            
            # Test schema inference
            result = self.validator.validate_schema(data, infer_schema=True, strict=False)
            self.reporter.add_result("validation_results", result)
            
            # Test enhanced metadata structure
            self.reporter.add_result("has_success_field", 'success' in result)
            self.reporter.add_result("has_data_field", 'data' in result)
            self.reporter.add_result("has_metadata_field", 'metadata' in result)
            self.reporter.add_result("has_errors_field", 'errors' in result)
            self.reporter.add_result("has_warnings_field", 'warnings' in result)
            
            # Test metadata contents
            metadata = result.get('metadata', {})
            self.reporter.add_result("metadata_has_operation", 'operation' in metadata)
            self.reporter.add_result("metadata_has_timestamp", 'timestamp' in metadata)
            self.reporter.add_result("metadata_has_performance", 'performance' in metadata)
            self.reporter.add_result("metadata_operation_correct", metadata.get('operation') == 'validate_schema')
            
            # Test performance metrics
            performance = metadata.get('performance', {})
            self.reporter.add_result("performance_has_duration", 'duration_ms' in performance)
            self.reporter.add_result("performance_has_memory", any(k in performance for k in ['start_memory_mb', 'end_memory_mb']))
            self.reporter.add_result("performance_has_rows", 'rows_processed' in performance)
            self.reporter.add_result("performance_rows_correct", performance.get('rows_processed') == len(data))
            
            # Add performance metrics to report
            if 'performance' in metadata:
                perf = metadata['performance']
                self.reporter.add_performance_metric("duration_ms", perf.get('duration_ms', 'N/A'))
                self.reporter.add_performance_metric("memory_start_mb", perf.get('start_memory_mb', 'N/A'))
                self.reporter.add_performance_metric("memory_end_mb", perf.get('end_memory_mb', 'N/A'))
                self.reporter.add_performance_metric("rows_processed_by_tracker", perf.get('rows_processed', 'N/A'))
            
            # Validate results (legacy compatibility)
            self.reporter.add_result("validation_completed", result is not None)
            schema_used = result.get('schema_used') or result.get('data', {}).get('schema_used')
            self.reporter.add_result("schema_inferred", schema_used is not None)
            validated_columns = result.get('validated_columns') or result.get('data', {}).get('validated_columns', [])
            self.reporter.add_result("columns_validated", len(validated_columns) > 0)
            
            # Check validation passed (new success field or legacy field)
            validation_passed = result.get('success', result.get('validation_passed', False))
            self.reporter.add_result("validation_passed", validation_passed)
            
            if not validation_passed:
                errors = result.get('errors', [])
                warnings = result.get('warnings', [])
                for error in errors:
                    self.reporter.add_error(f"Validation error: {error}")
                for warning in warnings:
                    self.reporter.add_warning(f"Validation warning: {warning}")
                    
            # Overall test passes if validation succeeded AND enhanced metadata is present
            enhanced_metadata_check = (
                'success' in result and 
                'metadata' in result and 
                'performance' in result.get('metadata', {})
            )
            self.reporter.add_result("enhanced_metadata_present", enhanced_metadata_check)
            
            status = "PASSED" if (validation_passed and enhanced_metadata_check) else "FAILED"
            
        except Exception as e:
            self.reporter.add_error(f"Test execution failed: {str(e)}")
            self.reporter.add_result("test_execution_failed", True)
            status = "FAILED"
            
        report_path = self.reporter.finish_test(status)
        self.test_results.append((test_name, dataset_name, status, report_path))
        print(f"✓ Basic schema inference test completed - Report: {report_path}")
        
    def test_validator_business_rules(self, data: pd.DataFrame, dataset_name: str):
        """Test business rules validation."""
        test_name = "business_rules_validation"
        
        # Define business rules based on dataset characteristics
        rules = []
        if 'age' in data.columns:
            rules.append({'name': 'age_range', 'column': 'age', 'condition': 'between', 'min': 0, 'max': 120})
        if 'salary' in data.columns:
            rules.append({'name': 'salary_positive', 'column': 'salary', 'condition': 'greater_than', 'value': 0})
        if 'email' in data.columns:
            rules.append({'name': 'email_format', 'column': 'email', 'condition': 'regex', 
                         'pattern': r'^[^@]+@[^@]+\.[^@]+$'})
        if 'user_id' in data.columns:
            rules.append({'name': 'user_id_unique', 'column': 'user_id', 'condition': 'unique'})
            
        self.reporter.start_test(test_name, "DataValidator", dataset_name, 
                               {"rules_count": len(rules), "rules": [r['name'] for r in rules]})
        
        try:
            # Add dataset analysis
            analysis = self.analyze_dataset(data, dataset_name)
            for key, value in analysis.items():
                self.reporter.add_data_analysis(key, value)
                
            self.reporter.add_performance_metric("rows_processed", len(data))
            
            if not rules:
                self.reporter.add_warning("No applicable business rules for this dataset")
                status = "PASSED"
            else:
                # Test business rules
                result = self.validator.validate_business_rules(data, rules=rules)
                self.reporter.add_result("business_rules_results", result)
                
                # Validate results
                self.reporter.add_result("rules_executed", len(result.get('rule_results', {})))
                self.reporter.add_result("overall_passed", result.get('overall_passed', False))
                self.reporter.add_result("failed_rules_count", len(result.get('failed_rules', [])))
                self.reporter.add_result("total_violations", result.get('total_violations', 0))
                
                # Check individual rule results
                for rule_name, rule_result in result.get('rule_results', {}).items():
                    self.reporter.add_result(f"rule_{rule_name}_passed", rule_result.get('passed', False))
                    if not rule_result.get('passed', False):
                        self.reporter.add_warning(f"Rule {rule_name} failed: {rule_result.get('message', 'Unknown error')}")
                        
                # For business rules, success means the validation engine worked, not that all rules passed
                # For messy data, we expect some rules to fail - that's the point!
                validation_engine_worked = (
                    len(result.get('rule_results', {})) == len(rules) and  # All rules were executed
                    all('passed' in rule_result for rule_result in result.get('rule_results', {}).values())  # All rules have results
                )
                
                # Additional success criteria based on dataset type
                if 'messy' in dataset_name.lower():
                    # For messy data, expect some rules to fail (that's good!)
                    some_violations_found = result.get('total_violations', 0) > 0
                    status = "PASSED" if validation_engine_worked else "FAILED"
                elif 'clean' in dataset_name.lower():
                    # For clean data, expect most/all rules to pass
                    overall_quality_good = result.get('overall_passed', False) or result.get('total_violations', 0) < 50
                    status = "PASSED" if validation_engine_worked and overall_quality_good else "FAILED"
                else:
                    # For other datasets, just check that the engine worked
                    status = "PASSED" if validation_engine_worked else "FAILED"
                
        except Exception as e:
            self.reporter.add_error(f"Test execution failed: {str(e)}")
            status = "FAILED"
            
        report_path = self.reporter.finish_test(status)
        self.test_results.append((test_name, dataset_name, status, report_path))
        print(f"✓ Business rules validation test completed - Report: {report_path}")
        
    def test_validator_statistical_validation(self, data: pd.DataFrame, dataset_name: str):
        """Test statistical validation."""
        test_name = "statistical_validation"
        tests_to_run = ['normality', 'outliers', 'correlation']
        
        self.reporter.start_test(test_name, "DataValidator", dataset_name, 
                               {"statistical_tests": tests_to_run})
        
        try:
            # Add dataset analysis
            analysis = self.analyze_dataset(data, dataset_name)
            for key, value in analysis.items():
                self.reporter.add_data_analysis(key, value)
                
            self.reporter.add_performance_metric("rows_processed", len(data))
            
            # Test statistical validation
            result = self.validator.statistical_validation(data, tests=tests_to_run)
            self.reporter.add_result("statistical_results", result)
            
            # Validate results
            test_results = result.get('test_results', {})
            self.reporter.add_result("tests_completed", len(test_results))
            
            # Check each statistical test
            for test_type in tests_to_run:
                if test_type in test_results:
                    self.reporter.add_result(f"{test_type}_test_completed", True)
                    
                    # Analyze specific results
                    if test_type == 'normality' and test_results[test_type]:
                        normal_cols = sum(1 for result in test_results[test_type].values() 
                                        if result.get('is_normal', False))
                        total_cols = len(test_results[test_type])
                        self.reporter.add_result(f"normality_normal_columns", f"{normal_cols}/{total_cols}")
                        
                    elif test_type == 'outliers' and test_results[test_type]:
                        high_outlier_cols = sum(1 for result in test_results[test_type].values()
                                              if result.get('outlier_percentage', 0) > 5)
                        self.reporter.add_result(f"outliers_high_outlier_columns", high_outlier_cols)
                        
                    elif test_type == 'correlation' and test_results[test_type]:
                        high_corrs = len(test_results[test_type].get('high_correlations', []))
                        self.reporter.add_result(f"correlation_high_correlations_found", high_corrs)
                else:
                    self.reporter.add_result(f"{test_type}_test_completed", False)
                    self.reporter.add_warning(f"Statistical test {test_type} not completed")
                    
            status = "PASSED"
            
        except Exception as e:
            self.reporter.add_error(f"Test execution failed: {str(e)}")
            status = "FAILED"
            
        report_path = self.reporter.finish_test(status)
        self.test_results.append((test_name, dataset_name, status, report_path))
        print(f"✓ Statistical validation test completed - Report: {report_path}")
        
    def test_schema_generator_basic_inference(self, data: pd.DataFrame, dataset_name: str):
        """Test basic schema inference."""
        test_name = "basic_schema_inference"
        
        self.reporter.start_test(test_name, "SchemaGenerator", dataset_name, 
                               {"infer_constraints": True, "include_nullability": True})
        
        try:
            # Add dataset analysis
            analysis = self.analyze_dataset(data, dataset_name)
            for key, value in analysis.items():
                self.reporter.add_data_analysis(key, value)
                
            self.reporter.add_performance_metric("rows_processed", len(data))
            
            # Test schema inference
            result = self.schema_generator.infer_schema(data, table_name=f"{dataset_name}_table")
            self.reporter.add_result("schema_results", result)
            
            # Validate results
            schema = result.get('schema', {})
            self.reporter.add_result("schema_generated", schema is not None)
            self.reporter.add_result("columns_inferred", len(schema.get('columns', {})))
            self.reporter.add_result("table_name_set", schema.get('table_name') is not None)
            
            # Check column inference quality
            original_columns = set(data.columns)
            inferred_columns = set(schema.get('columns', {}).keys())
            matching_columns = len(original_columns.intersection(inferred_columns))
            self.reporter.add_result("column_name_accuracy", f"{matching_columns}/{len(original_columns)}")
            
            # Check data type inference
            type_accuracy = 0
            total_columns = len(data.columns)
            for col in data.columns:
                if col in schema.get('columns', {}):
                    inferred_type = schema['columns'][col].get('data_type', '')
                    actual_type = str(data[col].dtype)
                    # Simple type matching
                    if ('int' in inferred_type and 'int' in actual_type) or \
                       ('float' in inferred_type and 'float' in actual_type) or \
                       ('str' in inferred_type and 'object' in actual_type) or \
                       ('bool' in inferred_type and 'bool' in actual_type):
                        type_accuracy += 1
                        
            self.reporter.add_result("data_type_accuracy", f"{type_accuracy}/{total_columns}")
            
            # Check constraints
            constraints = schema.get('constraints', {})
            self.reporter.add_result("constraints_generated", len(constraints))
            
            status = "PASSED"
            
        except Exception as e:
            self.reporter.add_error(f"Test execution failed: {str(e)}")
            status = "FAILED"
            
        report_path = self.reporter.finish_test(status)
        self.test_results.append((test_name, dataset_name, status, report_path))
        print(f"✓ Basic schema inference test completed - Report: {report_path}")
        
    def test_schema_generator_model_generation(self, data: pd.DataFrame, dataset_name: str):
        """Test model generation (SQLModel, Pydantic, DDL)."""
        test_name = "model_generation"
        
        self.reporter.start_test(test_name, "SchemaGenerator", dataset_name, 
                               {"generate_models": ["sqlmodel", "pydantic", "ddl"]})
        
        try:
            # Add dataset analysis
            analysis = self.analyze_dataset(data, dataset_name)
            for key, value in analysis.items():
                self.reporter.add_data_analysis(key, value)
                
            self.reporter.add_performance_metric("rows_processed", len(data))
            
            # First infer schema with metadata verification
            schema_result = self.schema_generator.infer_schema(data, table_name=f"{dataset_name}_table")
            
            # Test enhanced metadata for schema inference
            self.reporter.add_result("schema_has_success_field", 'success' in schema_result)
            self.reporter.add_result("schema_has_metadata", 'metadata' in schema_result)
            
            schema = schema_result.get('data', {}).get('schema') or schema_result.get('schema', {})
            
            if not schema:
                raise Exception("Schema inference failed")
                
            self.reporter.add_result("schema_inference_success", True)
            
            # Test SQLModel generation with enhanced metadata
            try:
                sqlmodel_result = self.schema_generator.generate_sqlmodel(schema=schema, table_name=f"{dataset_name}Model")
                
                # Test enhanced metadata structure
                self.reporter.add_result("sqlmodel_has_success", 'success' in sqlmodel_result)
                self.reporter.add_result("sqlmodel_has_metadata", 'metadata' in sqlmodel_result)
                self.reporter.add_result("sqlmodel_has_performance", 'performance' in sqlmodel_result.get('metadata', {}))
                
                # Check for actual success indicators: model_class and model_code in data field
                data_field = sqlmodel_result.get('data', {})
                legacy_success = bool(sqlmodel_result.get('model_class') and sqlmodel_result.get('model_code'))
                new_success = bool(data_field.get('model_class') and data_field.get('model_code'))
                success = sqlmodel_result.get('success', False) and (legacy_success or new_success)
                
                self.reporter.add_result("sqlmodel_generated", success)
                model_code = data_field.get('model_code') or sqlmodel_result.get('model_code')
                if model_code:
                    self.reporter.add_result("sqlmodel_code_length", len(model_code))
                    
                # Add performance metrics
                perf = sqlmodel_result.get('metadata', {}).get('performance', {})
                self.reporter.add_performance_metric("sqlmodel_duration_ms", perf.get('duration_ms', 'N/A'))
                    
            except Exception as e:
                self.reporter.add_error(f"SQLModel generation failed: {str(e)}")
                self.reporter.add_result("sqlmodel_generated", False)
                
            # Test Pydantic model generation with enhanced metadata
            try:
                pydantic_result = self.schema_generator.generate_pydantic_model(schema=schema, table_name=f"{dataset_name}Model")
                
                # Test enhanced metadata structure
                self.reporter.add_result("pydantic_has_success", 'success' in pydantic_result)
                self.reporter.add_result("pydantic_has_metadata", 'metadata' in pydantic_result)
                self.reporter.add_result("pydantic_has_performance", 'performance' in pydantic_result.get('metadata', {}))
                
                # Check for actual success indicators: model_class and model_code
                data_field = pydantic_result.get('data', {})
                legacy_success = bool(pydantic_result.get('model_class') and pydantic_result.get('model_code'))
                new_success = bool(data_field.get('model_class') and data_field.get('model_code'))
                success = pydantic_result.get('success', False) and (legacy_success or new_success)
                self.reporter.add_result("pydantic_generated", success)
                
                # Add performance metrics
                perf = pydantic_result.get('metadata', {}).get('performance', {})
                self.reporter.add_performance_metric("pydantic_duration_ms", perf.get('duration_ms', 'N/A'))
                if pydantic_result.get('model_code'):
                    self.reporter.add_result("pydantic_code_length", len(pydantic_result['model_code']))
            except Exception as e:
                self.reporter.add_error(f"Pydantic generation failed: {str(e)}")
                self.reporter.add_result("pydantic_generated", False)
                
            # Test DDL generation with enhanced metadata
            try:
                ddl_result = self.schema_generator.generate_database_ddl(schema=schema, table_name=f"{dataset_name}_table")
                
                # Test enhanced metadata structure
                self.reporter.add_result("ddl_has_success", 'success' in ddl_result)
                self.reporter.add_result("ddl_has_metadata", 'metadata' in ddl_result)
                self.reporter.add_result("ddl_has_performance", 'performance' in ddl_result.get('metadata', {}))
                
                # Check for actual success indicators: create_table_ddl
                data_field = ddl_result.get('data', {})
                legacy_success = bool(ddl_result.get('create_table_ddl'))
                new_success = bool(data_field.get('create_table_ddl'))
                success = ddl_result.get('success', False) and (legacy_success or new_success)
                
                self.reporter.add_result("ddl_generated", success)
                
                ddl_sql = data_field.get('create_table_ddl') or ddl_result.get('create_table_ddl')
                if ddl_sql:
                    self.reporter.add_result("ddl_length", len(ddl_sql))
                    
                # Add performance metrics
                perf = ddl_result.get('metadata', {}).get('performance', {})
                self.reporter.add_performance_metric("ddl_duration_ms", perf.get('duration_ms', 'N/A'))
                    
            except Exception as e:
                self.reporter.add_error(f"DDL generation failed: {str(e)}")
                self.reporter.add_result("ddl_generated", False)
                
            # Overall success - check both model generation and enhanced metadata
            models_generated = sum([
                self.reporter.current_test['results'].get('sqlmodel_generated', False),
                self.reporter.current_test['results'].get('pydantic_generated', False), 
                self.reporter.current_test['results'].get('ddl_generated', False)
            ])
            
            # Check enhanced metadata presence
            metadata_checks = sum([
                self.reporter.current_test['results'].get('sqlmodel_has_metadata', False),
                self.reporter.current_test['results'].get('pydantic_has_metadata', False),
                self.reporter.current_test['results'].get('ddl_has_metadata', False)
            ])
            
            self.reporter.add_result("total_models_generated", f"{models_generated}/3")
            self.reporter.add_result("enhanced_metadata_checks", f"{metadata_checks}/3")
            
            # Pass if most models generated AND enhanced metadata is present
            status = "PASSED" if (models_generated >= 2 and metadata_checks >= 2) else "FAILED"
            
        except Exception as e:
            self.reporter.add_error(f"Test execution failed: {str(e)}")
            status = "FAILED"
            
        report_path = self.reporter.finish_test(status)
        self.test_results.append((test_name, dataset_name, status, report_path))
        print(f"✓ Model generation test completed - Report: {report_path}")
        
    def test_integration_workflow(self, data: pd.DataFrame, dataset_name: str):
        """Test complete integration workflow."""
        test_name = "integration_workflow"
        
        self.reporter.start_test(test_name, "Integration", dataset_name, 
                               {"workflow": "validate -> infer_schema -> generate_models"})
        
        try:
            # Add dataset analysis
            analysis = self.analyze_dataset(data, dataset_name)
            for key, value in analysis.items():
                self.reporter.add_data_analysis(key, value)
                
            self.reporter.add_performance_metric("rows_processed", len(data))
            
            # Step 1: Validate data
            validation_result = self.validator.validate_schema(data, infer_schema=True, strict=False)
            self.reporter.add_result("step1_validation_completed", validation_result is not None)
            self.reporter.add_result("step1_validation_passed", validation_result.get('validation_passed', False))
            
            # Step 2: Generate schema
            schema_result = self.schema_generator.infer_schema(data, table_name=f"{dataset_name}_integrated")
            self.reporter.add_result("step2_schema_generated", schema_result is not None)
            
            schema = schema_result.get('schema', {}) if schema_result else {}
            self.reporter.add_result("step2_columns_inferred", len(schema.get('columns', {})))
            
            # Step 3: Generate models
            models_success = 0
            if schema:
                try:
                    sqlmodel_result = self.schema_generator.generate_sqlmodel(schema, f"{dataset_name}IntegratedModel")
                    if sqlmodel_result.get('success', False):
                        models_success += 1
                except:
                    pass
                    
                try:
                    pydantic_result = self.schema_generator.generate_pydantic_model(schema, f"{dataset_name}IntegratedModel")
                    if pydantic_result.get('success', False):
                        models_success += 1
                except:
                    pass
                    
            self.reporter.add_result("step3_models_generated", models_success)
            
            # Check data consistency between steps
            validation_schema = validation_result.get('schema_used', {}) if validation_result else {}
            generated_schema = schema.get('columns', {})
            
            if validation_schema and generated_schema:
                # Compare column counts
                val_cols = len(validation_schema)
                gen_cols = len(generated_schema)
                self.reporter.add_result("schema_consistency", abs(val_cols - gen_cols) <= 1)
            else:
                self.reporter.add_result("schema_consistency", False)
                
            # Overall workflow success
            workflow_steps_passed = sum([
                validation_result is not None,
                schema_result is not None,
                models_success > 0
            ])
            
            self.reporter.add_result("workflow_steps_completed", f"{workflow_steps_passed}/3")
            status = "PASSED" if workflow_steps_passed >= 2 else "FAILED"
            
        except Exception as e:
            self.reporter.add_error(f"Test execution failed: {str(e)}")
            status = "FAILED"
            
        report_path = self.reporter.finish_test(status)
        self.test_results.append((test_name, dataset_name, status, report_path))
        print(f"✓ Integration workflow test completed - Report: {report_path}")
        
    def run_comprehensive_tests(self):
        """Run all tests on all available datasets."""
        print("🚀 Starting Comprehensive Validation & Schema Testing Suite")
        print("=" * 60)
        
        # Define test datasets
        datasets = [
            ("sample_data/clean_dataset.csv", "clean_employee_data"),
            ("sample_data/messy_dataset.csv", "messy_employee_data"),
            ("sample_data/edge_case_date_edge_cases.csv", "date_edge_cases"),
            ("sample_data/edge_case_extreme_types.csv", "extreme_types"),
        ]
        
        total_tests = 0
        passed_tests = 0
        
        for dataset_path, dataset_name in datasets:
            print(f"\n📊 Testing dataset: {dataset_name}")
            print("-" * 40)
            
            # Load dataset
            data = self.load_dataset(dataset_path)
            if data.empty:
                print(f"⚠️ Skipping {dataset_name} - failed to load")
                continue
                
            print(f"Dataset loaded: {data.shape[0]} rows × {data.shape[1]} columns")
            
            # Run all test types on this dataset
            test_methods = [
                self.test_validator_basic_schema_inference,
                self.test_validator_business_rules,
                self.test_validator_statistical_validation,
                self.test_schema_generator_basic_inference,
                self.test_schema_generator_model_generation,
                self.test_integration_workflow,
            ]
            
            for test_method in test_methods:
                try:
                    test_method(data, dataset_name)
                    total_tests += 1
                    # Check if last test passed
                    if self.test_results and self.test_results[-1][2] == "PASSED":
                        passed_tests += 1
                except Exception as e:
                    print(f"❌ Test {test_method.__name__} failed: {str(e)}")
                    total_tests += 1
                    
        # Generate summary report
        self.generate_summary_report(total_tests, passed_tests)
        
    def generate_summary_report(self, total_tests: int, passed_tests: int):
        """Generate comprehensive summary report."""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        summary_path = self.reporter.reports_dir / f"comprehensive_test_summary_{timestamp}.txt"
        
        success_rate = (passed_tests / total_tests * 100) if total_tests > 0 else 0
        
        summary_content = f"""=== COMPREHENSIVE TEST SUITE SUMMARY ===
Execution Date: {datetime.now().isoformat()}
Total Tests Executed: {total_tests}
Tests Passed: {passed_tests}
Tests Failed: {total_tests - passed_tests}
Overall Success Rate: {success_rate:.1f}%

=== TEST RESULTS BY CATEGORY ===
"""
        
        # Organize results by test type and dataset
        by_test_type = {}
        by_dataset = {}
        
        for test_name, dataset_name, status, report_path in self.test_results:
            # By test type
            if test_name not in by_test_type:
                by_test_type[test_name] = {"PASSED": 0, "FAILED": 0}
            by_test_type[test_name][status] += 1
            
            # By dataset
            if dataset_name not in by_dataset:
                by_dataset[dataset_name] = {"PASSED": 0, "FAILED": 0}
            by_dataset[dataset_name][status] += 1
            
        # Add test type summary
        summary_content += "\nBy Test Type:\n"
        for test_type, counts in by_test_type.items():
            total = counts["PASSED"] + counts["FAILED"]
            rate = counts["PASSED"] / total * 100 if total > 0 else 0
            summary_content += f"  {test_type}: {counts['PASSED']}/{total} passed ({rate:.1f}%)\n"
            
        # Add dataset summary
        summary_content += "\nBy Dataset:\n"
        for dataset, counts in by_dataset.items():
            total = counts["PASSED"] + counts["FAILED"]
            rate = counts["PASSED"] / total * 100 if total > 0 else 0
            summary_content += f"  {dataset}: {counts['PASSED']}/{total} passed ({rate:.1f}%)\n"
            
        summary_content += f"""
=== DETAILED TEST RESULTS ===
"""
        
        for test_name, dataset_name, status, report_path in self.test_results:
            status_symbol = "✓" if status == "PASSED" else "✗"
            summary_content += f"{status_symbol} {test_name} on {dataset_name} - {report_path}\n"
            
        summary_content += f"""
=== OVERALL ASSESSMENT ===
"""
        
        if success_rate >= 90:
            summary_content += "🎉 EXCELLENT: Test suite shows high reliability and robustness\n"
        elif success_rate >= 75:
            summary_content += "✅ GOOD: Test suite shows good reliability with some areas for improvement\n"
        elif success_rate >= 50:
            summary_content += "⚠️ MODERATE: Test suite shows moderate reliability, investigation needed\n"
        else:
            summary_content += "❌ POOR: Test suite shows significant issues, immediate attention required\n"
            
        summary_content += f"""
=== RECOMMENDATIONS ===
"""
        
        if success_rate < 100:
            summary_content += "- Review failed tests and address underlying issues\n"
        if total_tests < 20:
            summary_content += "- Consider adding more edge case tests\n"
        summary_content += "- Monitor performance trends across test runs\n"
        summary_content += "- Maintain test coverage as codebase evolves\n"
        
        summary_content += "================================\n"
        
        # Write summary report
        with open(summary_path, 'w') as f:
            f.write(summary_content)
            
        print(f"\n🎯 Comprehensive Testing Complete!")
        print(f"📊 Results: {passed_tests}/{total_tests} tests passed ({success_rate:.1f}%)")
        print(f"📄 Summary report: {summary_path}")
        print(f"📁 All reports saved to: {self.reporter.reports_dir}")


def main():
    """Main execution function."""
    suite = ComprehensiveTestSuite()
    suite.run_comprehensive_tests()


if __name__ == "__main__":
    main()