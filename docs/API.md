# Validata API Reference

This document provides detailed API reference for all Validata modules and classes.

## Table of Contents

- [ValidataAPI](#validataapi) - Main unified interface
- [DataProfiler](#dataprofiler) - Data profiling and analysis
- [DataCleaner](#datacleaner) - Data cleaning operations  
- [DataValidator](#datavalidator) - Data validation and quality checks
- [DataStandardizer](#datastandardizer) - Data standardization and preprocessing
- [SchemaGenerator](#schemagenerator) - Schema inference and model generation
- [Utility Classes](#utility-classes) - Support classes and functions

---

## ValidataAPI

Main unified interface for all data validation and processing operations.

### Constructor

```python
ValidataAPI(config_path: Optional[Union[str, Path]] = None)
```

**Parameters:**
- `config_path`: Optional path to configuration file

### Core Methods

#### `load_data(data_source, **kwargs) -> ValidataAPI`

Load data for processing. Supports method chaining.

**Parameters:**
- `data_source`: DataFrame, file path, or data source
- `**kwargs`: Additional arguments for data loading

**Returns:** Self for method chaining

**Example:**
```python
validata.load_data("data.csv").profile_data()
```

#### `profile_data(data=None, title="Data Profile", minimal=None, **kwargs) -> Union[Dict, ValidataAPI]`

Generate comprehensive data profile.

**Parameters:**
- `data`: Data to profile (uses current data if None)
- `title`: Title for the profiling report
- `minimal`: Use minimal profiling for faster execution
- `**kwargs`: Additional arguments for profiling

**Returns:** Profile results dict or self for chaining

#### `clean_data(data=None, strategy='auto', **kwargs) -> Union[Dict, ValidataAPI]`

Clean data using comprehensive cleaning methods.

**Parameters:**
- `data`: Data to clean (uses current data if None)  
- `strategy`: Cleaning strategy ('auto', 'conservative', 'aggressive')
- `**kwargs`: Additional arguments for cleaning

**Returns:** Cleaning results dict or self for chaining

#### `standardize_data(data=None, method='auto', **kwargs) -> Union[Dict, ValidataAPI]`

Standardize data using normalization and encoding methods.

**Parameters:**
- `data`: Data to standardize (uses current data if None)
- `method`: Standardization method ('auto', 'minimal', 'full')
- `**kwargs`: Additional arguments for standardization

#### `validate_data(data=None, validation_type='comprehensive', **kwargs) -> Union[Dict, ValidataAPI]`

Validate data using schema and business rule validation.

**Parameters:**
- `data`: Data to validate (uses current data if None)
- `validation_type`: Type of validation ('schema', 'business', 'statistical', 'comprehensive')
- `**kwargs`: Additional arguments for validation

#### `generate_schema(data=None, table_name="generated_table", schema_type='comprehensive', **kwargs) -> Union[Dict, ValidataAPI]`

Generate schema from data analysis.

**Parameters:**
- `data`: Data to analyze (uses current data if None)
- `table_name`: Name for the generated table/model
- `schema_type`: Type of schema ('basic', 'comprehensive', 'with_models')
- `**kwargs`: Additional arguments for schema generation

#### `analyze_all(data, table_name="analysis_table") -> Dict[str, Any]`

Perform comprehensive analysis including profiling, cleaning, validation, and schema generation.

**Parameters:**
- `data`: Data to analyze
- `table_name`: Name for schema generation

**Returns:** Comprehensive analysis results

### Workflow Methods

#### `create_workflow(name: str) -> ValidataAPI`

Start a named workflow for tracking and potential replay.

#### `save_workflow(file_path: Union[str, Path]) -> str`

Save current workflow to file.

#### `get_current_data() -> Optional[pd.DataFrame]`

Get the current data being processed.

#### `get_data_summary() -> Dict[str, Any]`

Get summary of current data.

#### `export_results(file_path, include_data=True, include_reports=True) -> str`

Export comprehensive results and reports.

---

## DataProfiler

Custom pandas-based profiler for comprehensive data analysis.

### Constructor

```python
DataProfiler(config_manager: Optional[ConfigManager] = None)
```

### Methods

#### `profile_data(data, title=None, minimal=False, include_correlations=True, sample_size=None) -> Dict[str, Any]`

Generate comprehensive data profile.

**Parameters:**
- `data`: DataFrame or path to data file
- `title`: Title for the profile report
- `minimal`: Whether to generate minimal profile (faster)
- `include_correlations`: Whether to compute correlations
- `sample_size`: Sample size for large datasets (None = use all data)

**Returns:** Dictionary containing comprehensive data profile

**Output Structure:**
```python
{
    "title": "Data Profile Title",
    "summary": {
        "shape": [1000, 10],
        "n_records": 1000,
        "n_variables": 10,
        "missing_cells_percent": 2.5,
        "duplicate_rows_percent": 0.1,
        "memory_usage_mb": 0.5,
        "data_types": {
            "numerical": 6,
            "categorical": 3,
            "datetime": 1,
            "boolean": 0
        }
    },
    "columns": {
        "column_name": {
            "type": "numerical",
            "missing_count": 10,
            "missing_percent": 1.0,
            "unique_count": 950,
            "data_type": "float64",
            "stats": {...},  # For numerical columns
            "quality_issues": []
        }
    },
    "correlations": {
        "correlation_matrix": {...},
        "high_correlations": [...]
    },
    "quality_report": {
        "issues_found": 3,
        "issues": [...],
        "recommendations": [...],
        "quality_score": 85
    },
    "metadata": {...}
}
```

#### `quick_profile(data) -> str`

Generate a quick text summary of the dataset.

**Parameters:**
- `data`: DataFrame or path to data file

**Returns:** String with quick summary

#### `export_profile(profile, file_path) -> str`

Export profile to JSON file.

---

## DataCleaner

Comprehensive data cleaning with multiple strategies.

### Constructor

```python
DataCleaner(config_manager: Optional[ConfigManager] = None)
```

### Methods

#### `handle_missing_data(data, strategy='auto', fill_value=None, threshold=None, columns=None) -> Dict[str, Any]`

Handle missing data using various strategies.

**Parameters:**
- `data`: DataFrame or path to data file
- `strategy`: Strategy to use ('drop', 'fill', 'interpolate', 'auto')
- `fill_value`: Value to use for filling (can be dict for column-specific values)
- `threshold`: Threshold for dropping rows/columns (fraction of missing values)
- `columns`: Specific columns to process (all if None)

**Returns:** Dictionary containing cleaned data and operation summary

#### `remove_duplicates(data, subset=None, keep='first', ignore_index=True) -> Dict[str, Any]`

Remove duplicate rows from the dataset.

**Parameters:**
- `data`: DataFrame or path to data file
- `subset`: Columns to consider for duplicate identification
- `keep`: Which duplicates to keep ('first', 'last', False)
- `ignore_index`: Reset index after removing duplicates

#### `handle_outliers(data, method='auto', threshold=None, columns=None, action='flag') -> Dict[str, Any]`

Detect and handle outliers in numeric columns.

**Parameters:**
- `data`: DataFrame or path to data file
- `method`: Outlier detection method ('iqr', 'zscore', 'isolation_forest', 'auto')
- `threshold`: Threshold for outlier detection (method-specific)
- `columns`: Numeric columns to check (auto-detect if None)
- `action`: Action to take ('remove', 'cap', 'flag')

#### `fix_data_types(data, type_mapping=None, auto_infer=True, strict=False) -> Dict[str, Any]`

Fix and convert data types based on content analysis.

**Parameters:**
- `data`: DataFrame or path to data file
- `type_mapping`: Manual type mapping (column -> type)
- `auto_infer`: Automatically infer optimal types
- `strict`: Raise errors on conversion failures

#### `clean_text_data(data, columns=None, operations=None, custom_patterns=None) -> Dict[str, Any]`

Clean and standardize text data.

**Parameters:**
- `data`: DataFrame or path to data file
- `columns`: Text columns to clean (auto-detect if None)
- `operations`: List of cleaning operations to perform
- `custom_patterns`: Custom regex patterns for cleaning

**Available operations:** 'lowercase', 'uppercase', 'strip', 'normalize_whitespace', 'remove_special_chars', 'remove_digits', 'remove_punctuation'

#### `clean_all(data, config=None) -> Dict[str, Any]`

Perform comprehensive data cleaning using all available methods.

**Parameters:**
- `data`: DataFrame or path to data file
- `config`: Configuration for cleaning operations

---

## DataValidator

Comprehensive validation with Pandera and Great Expectations.

### Constructor

```python
DataValidator(config_manager: Optional[ConfigManager] = None)
```

### Methods

#### `validate_schema(data, schema=None, schema_name=None, infer_schema=False, strict=False) -> Dict[str, Any]`

Validate data against a Pandera schema.

**Parameters:**
- `data`: DataFrame or path to data file
- `schema`: Pandera schema or schema dictionary
- `schema_name`: Name of stored schema to use
- `infer_schema`: Whether to infer schema from data
- `strict`: Whether to use strict validation mode

#### `validate_business_rules(data, rules=None, rule_set_name=None) -> Dict[str, Any]`

Validate data against custom business rules.

**Parameters:**
- `data`: DataFrame or path to data file
- `rules`: List of business rule specifications
- `rule_set_name`: Name of stored rule set to use

**Rule Specification Format:**
```python
{
    'name': 'rule_name',
    'column': 'column_name',
    'condition': 'validation_condition',
    # Additional parameters based on condition
}
```

**Available Conditions:**
- `'not_null'`: Check for non-null values
- `'unique'`: Check for unique values
- `'between'`: Check values are in range (requires 'min', 'max')
- `'greater_than'`: Check values > threshold (requires 'value')
- `'less_than'`: Check values < threshold (requires 'value')
- `'isin'`: Check values in allowed list (requires 'values')
- `'regex'`: Check values match pattern (requires 'pattern')
- `'custom'`: Custom validation function (requires 'function')

#### `statistical_validation(data, reference_data=None, tests=None, significance_level=0.05) -> Dict[str, Any]`

Perform statistical validation and hypothesis testing.

**Parameters:**
- `data`: DataFrame or path to data file
- `reference_data`: Reference dataset for comparison
- `tests`: List of statistical tests to perform
- `significance_level`: Significance level for hypothesis tests

**Available Tests:**
- `'normality'`: Shapiro-Wilk normality test
- `'outliers'`: IQR-based outlier detection
- `'correlation'`: High correlation detection
- `'distribution_comparison'`: Kolmogorov-Smirnov test (requires reference_data)
- `'mean_comparison'`: T-test for mean comparison (requires reference_data)

### Schema Management

#### `save_schema(schema: pa.DataFrameSchema, name: str) -> None`

Save a schema for reuse.

#### `get_saved_schemas() -> Dict[str, Any]`

Get information about saved schemas.

#### `add_custom_rule(rule_name: str, rule_definition: Dict[str, Any]) -> None`

Add a custom validation rule.

---

## DataStandardizer

Data standardization and preprocessing operations.

### Constructor

```python
DataStandardizer(config_manager: Optional[ConfigManager] = None)
```

### Methods

#### `standardize_numerical(data, columns=None, method='standard', **kwargs) -> Dict[str, Any]`

Standardize numerical columns using various scaling methods.

**Parameters:**
- `data`: DataFrame or path to data file
- `columns`: Numerical columns to standardize (auto-detect if None)
- `method`: Scaling method ('standard', 'minmax', 'robust', 'maxabs', 'quantile')

**Scaling Methods:**
- `'standard'`: StandardScaler (mean=0, std=1)
- `'minmax'`: MinMaxScaler (range 0-1)
- `'robust'`: RobustScaler (median and IQR)
- `'maxabs'`: MaxAbsScaler (maximum absolute value)
- `'quantile'`: QuantileTransformer

#### `encode_categorical(data, columns=None, method='onehot', **kwargs) -> Dict[str, Any]`

Encode categorical variables using various encoding methods.

**Parameters:**
- `data`: DataFrame or path to data file
- `columns`: Categorical columns to encode (auto-detect if None)
- `method`: Encoding method ('onehot', 'label', 'ordinal', 'target', 'binary')

**Encoding Methods:**
- `'onehot'`: One-hot encoding (dummy variables)
- `'label'`: Label encoding (integer mapping)
- `'ordinal'`: Ordinal encoding (custom order)
- `'target'`: Target encoding (mean of target variable)
- `'binary'`: Binary encoding

#### `standardize_dates(data, columns=None, format=None, extract_features=False) -> Dict[str, Any]`

Standardize date/datetime columns.

**Parameters:**
- `data`: DataFrame or path to data file
- `columns`: Date columns to standardize (auto-detect if None)
- `format`: Date format string (auto-infer if None)
- `extract_features`: Whether to extract date features (year, month, day, etc.)

#### `standardize_all(data, config=None) -> Dict[str, Any]`

Perform comprehensive standardization using all methods.

**Parameters:**
- `data`: DataFrame or path to data file
- `config`: Configuration for standardization operations

---

## SchemaGenerator

Automatic schema inference and model generation.

### Constructor

```python
SchemaGenerator(config_manager: Optional[ConfigManager] = None)
```

### Methods

#### `infer_schema(data, table_name="inferred_table", infer_constraints=True, include_relationships=False) -> Dict[str, Any]`

Infer comprehensive schema from data.

**Parameters:**
- `data`: DataFrame or path to data file
- `table_name`: Name for the table/model
- `infer_constraints`: Whether to infer data constraints
- `include_relationships`: Whether to detect potential relationships

**Returns:** Dictionary containing inferred schema information

#### `generate_sqlmodel(schema, table_name, **kwargs) -> Dict[str, Any]`

Generate SQLModel code from schema.

**Parameters:**
- `schema`: Schema dictionary from infer_schema()
- `table_name`: Name for the SQLModel class
- `**kwargs`: Additional generation options

#### `generate_pydantic_model(schema, table_name, **kwargs) -> Dict[str, Any]`

Generate Pydantic model code from schema.

#### `generate_database_ddl(schema, table_name, dialect="postgresql", **kwargs) -> Dict[str, Any]`

Generate database DDL (CREATE TABLE) from schema.

**Parameters:**
- `schema`: Schema dictionary from infer_schema()
- `table_name`: Name for the database table
- `dialect`: SQL dialect ('postgresql', 'mysql', 'sqlite', 'oracle')

---

## Utility Classes

### ConfigManager

Configuration management for Validata operations.

```python
ConfigManager(config_path: Optional[Union[str, Path, Dict]] = None)
```

### OperationTracker

Tracks operations and performance metrics.

```python
OperationTracker()
```

### FileOperations

File I/O utilities for data loading and saving.

#### `load_data(file_path, **kwargs) -> pd.DataFrame`

Load data from various file formats.

#### `save_data(data, file_path, **kwargs) -> str`

Save DataFrame to file.

### DataTypeInference

Intelligent data type inference utilities.

#### `infer_column_types(data) -> Dict[str, str]`

Infer optimal data types for columns.

---

## Quick Functions

Convenience functions for common operations:

```python
# Quick profiling
profile = quick_profile("data.csv")

# Quick cleaning  
cleaned_df = quick_clean("data.csv")

# Quick comprehensive analysis
analysis = quick_analyze("data.csv", table_name="my_table")
```

## Error Handling

All Validata methods include comprehensive error handling:

- **ValidationError**: Raised for data validation failures
- **ConfigurationError**: Raised for configuration issues
- **DataLoadError**: Raised for data loading problems
- **SchemaError**: Raised for schema-related issues

## Return Value Patterns

Validata follows consistent return patterns:

### Method Chaining
Methods return `self` when used in chains:
```python
result = validata.load_data().clean_data().validate_data()
```

### Direct Usage
Methods return detailed dictionaries when called directly:
```python
{
    'success': bool,
    'data': {...},           # Main operation results
    'metadata': {            # Operation metadata
        'operation': str,
        'timestamp': str,
        'performance': {...},
        'parameters': {...}
    },
    'errors': [...],         # Any errors encountered
    'warnings': [...]        # Warnings or issues
}
```

## Performance Considerations

- Use `minimal=True` for faster profiling on large datasets
- Set `sample_size` parameter for very large datasets
- Use `strict=False` for validation in data pipelines
- Configure operations through ConfigManager for consistent behavior

## Configuration

See the main README for configuration options and examples.