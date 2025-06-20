# Validata Examples and Tutorials

This document provides comprehensive examples and tutorials for using Validata in various scenarios.

## Table of Contents

- [Getting Started](#getting-started)
- [Data Profiling Examples](#data-profiling-examples)
- [Data Cleaning Examples](#data-cleaning-examples)
- [Data Validation Examples](#data-validation-examples)
- [Schema Generation Examples](#schema-generation-examples)
- [Complete Workflows](#complete-workflows)
- [Advanced Use Cases](#advanced-use-cases)
- [Integration Examples](#integration-examples)

---

## Getting Started

### Basic Setup

```python
import pandas as pd
from validata import ValidataAPI, DataProfiler, DataCleaner, DataValidator

# Initialize the main API
validata = ValidataAPI()

# Or use individual modules
profiler = DataProfiler()
cleaner = DataCleaner()
validator = DataValidator()
```

### Sample Data Creation

```python
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Create sample dataset for examples
np.random.seed(42)
n_records = 1000

sample_data = pd.DataFrame({
    'user_id': range(1, n_records + 1),
    'name': [f'User_{i}' if i % 10 != 0 else None for i in range(n_records)],
    'email': [f'user{i}@example.com' if i % 15 != 0 else 'invalid_email' 
              for i in range(n_records)],
    'age': np.random.normal(35, 12, n_records),
    'income': np.random.lognormal(10, 1, n_records),
    'signup_date': [datetime.now() - timedelta(days=np.random.randint(0, 365)) 
                    for _ in range(n_records)],
    'status': np.random.choice(['active', 'inactive', 'pending'], n_records),
    'premium': np.random.choice([True, False], n_records)
})

# Introduce some data quality issues
sample_data.loc[0:50, 'age'] = None  # Missing ages
sample_data.loc[100:110] = sample_data.loc[100:110]  # Duplicates
sample_data.loc[200, 'age'] = -5  # Invalid age
sample_data.loc[300, 'income'] = 10000000  # Outlier income

print(f"Sample dataset created: {sample_data.shape}")
```

---

## Data Profiling Examples

### Example 1: Basic Data Profiling

```python
from validata import DataProfiler

profiler = DataProfiler()

# Generate comprehensive profile
profile = profiler.profile_data(
    sample_data,
    title="Customer Data Analysis",
    minimal=False,
    include_correlations=True
)

# Display summary information
print("=== DATASET SUMMARY ===")
print(f"Shape: {profile['summary']['shape']}")
print(f"Missing data: {profile['summary']['missing_cells_percent']:.1f}%")
print(f"Duplicates: {profile['summary']['duplicate_rows_percent']:.1f}%")
print(f"Memory usage: {profile['summary']['memory_usage_mb']:.2f} MB")

# Display data types
print("\n=== DATA TYPES ===")
for dtype, count in profile['summary']['data_types'].items():
    print(f"{dtype.title()}: {count} columns")

# Display quality issues
print("\n=== QUALITY ISSUES ===")
if profile['quality_report']['issues']:
    for issue in profile['quality_report']['issues']:
        print(f"• {issue}")
else:
    print("No major quality issues detected")

# Column-level analysis
print("\n=== COLUMN ANALYSIS ===")
for col_name, col_info in profile['columns'].items():
    print(f"\n{col_name}:")
    print(f"  Type: {col_info['type']}")
    print(f"  Missing: {col_info['missing_percent']:.1f}%")
    print(f"  Unique values: {col_info['unique_count']}")
    
    if col_info['quality_issues']:
        print(f"  Issues: {', '.join(col_info['quality_issues'])}")
```

### Example 2: Quick Profiling for Large Datasets

```python
# For large datasets, use sampling and minimal mode
large_profile = profiler.profile_data(
    "large_dataset.csv",
    title="Large Dataset Quick Profile",
    minimal=True,
    sample_size=10000  # Sample 10k rows
)

# Quick text summary
quick_summary = profiler.quick_profile(sample_data)
print(quick_summary)
```

### Example 3: Correlation Analysis

```python
# Profile with focus on correlations
profile = profiler.profile_data(
    sample_data,
    include_correlations=True
)

# Display correlation findings
if 'high_correlations' in profile['correlations']:
    print("=== HIGH CORRELATIONS ===")
    for corr in profile['correlations']['high_correlations']:
        print(f"{corr['column1']} ↔ {corr['column2']}: {corr['correlation']:.3f} ({corr['strength']})")
else:
    print("No high correlations found")

# Export profile for later use
profile_path = profiler.export_profile(profile, "customer_profile.json")
print(f"Profile exported to: {profile_path}")
```

---

## Data Cleaning Examples

### Example 1: Handling Missing Data

```python
from validata import DataCleaner

cleaner = DataCleaner()

# Strategy 1: Automatic handling
auto_result = cleaner.handle_missing_data(
    sample_data,
    strategy='auto'
)

print("=== AUTOMATIC MISSING DATA HANDLING ===")
print(f"Original missing cells: {auto_result['summary']['original_missing_cells']}")
print(f"Final missing cells: {auto_result['summary']['final_missing_cells']}")
print(f"Reduction: {auto_result['summary']['missing_reduction_percent']:.1f}%")

# Strategy 2: Custom fill values
fill_values = {
    'name': 'Unknown User',
    'age': sample_data['age'].median(),
    'email': 'no-email@example.com'
}

fill_result = cleaner.handle_missing_data(
    sample_data,
    strategy='fill',
    fill_value=fill_values
)

# Strategy 3: Drop rows with too much missing data
threshold_result = cleaner.handle_missing_data(
    sample_data,
    strategy='drop',
    threshold=0.3  # Drop rows missing >30% of data
)

print(f"Threshold cleaning removed {threshold_result['summary']['rows_dropped']} rows")
```

### Example 2: Duplicate Removal

```python
# Remove duplicates based on specific columns
dup_result = cleaner.remove_duplicates(
    sample_data,
    subset=['name', 'email'],  # Consider these columns for duplicates
    keep='first'  # Keep first occurrence
)

print("=== DUPLICATE REMOVAL ===")
print(f"Duplicates found: {dup_result['summary']['duplicates_found']}")
print(f"Duplicates removed: {dup_result['summary']['duplicates_removed']}")
print(f"Duplicate rate: {dup_result['summary']['duplicate_rate_percent']:.1f}%")

# Show duplicate rows that were found
if len(dup_result['duplicate_rows']) > 0:
    print("\nSample duplicate rows:")
    print(dup_result['duplicate_rows'].head())
```

### Example 3: Outlier Detection and Handling

```python
# Method 1: IQR-based outlier detection
iqr_result = cleaner.handle_outliers(
    sample_data,
    method='iqr',
    columns=['age', 'income'],
    action='flag'  # Add outlier flags instead of removing
)

print("=== IQR OUTLIER DETECTION ===")
for col, info in iqr_result['outlier_info'].items():
    print(f"{col}: {info['outlier_count']} outliers ({len(info['outlier_values'])} values)")
    if info['outlier_values']:
        print(f"  Sample outlier values: {info['outlier_values'][:5]}")

# Method 2: Z-score based detection
zscore_result = cleaner.handle_outliers(
    sample_data,
    method='zscore',
    threshold=3.0,
    action='cap'  # Cap outliers to reasonable bounds
)

# Method 3: Isolation Forest (for multivariate outliers)
isolation_result = cleaner.handle_outliers(
    sample_data,
    method='isolation_forest',
    columns=['age', 'income'],
    action='remove'
)
```

### Example 4: Data Type Fixing

```python
# Create dataset with mixed types
messy_data = pd.DataFrame({
    'id': ['1', '2', '3', '4'],
    'score': ['85.5', '92', 'invalid', '78.2'],
    'date': ['2023-01-15', '2023/02/20', '15-03-2023', 'invalid_date'],
    'flag': ['true', 'False', '1', '0']
})

# Fix data types automatically
type_result = cleaner.fix_data_types(
    messy_data,
    auto_infer=True,
    strict=False  # Don't fail on conversion errors
)

print("=== DATA TYPE CONVERSION ===")
print(f"Successful conversions: {type_result['summary']['successful_conversions']}")
print(f"Failed conversions: {type_result['summary']['failed_conversions']}")

# Show conversion details
for col, details in type_result['summary']['types_changed'].items():
    print(f"{col}: {details['original_type']} → {details['new_type']}")

# Manual type mapping
type_mapping = {
    'id': 'integer',
    'score': 'float',
    'date': 'datetime',
    'flag': 'boolean'
}

manual_result = cleaner.fix_data_types(
    messy_data,
    type_mapping=type_mapping,
    auto_infer=False
)
```

### Example 5: Text Data Cleaning

```python
# Create messy text data
text_data = pd.DataFrame({
    'name': ['  John Doe  ', 'JANE SMITH', 'bob@#$johnson', 'Mary   O\'Connor'],
    'description': ['Great Product!!!', 'excellent    quality', 'AMAZING123', 'so-so product'],
    'phone': ['(555) 123-4567', '555.987.6543', '555-111-2222', '5551234567']
})

# Apply text cleaning operations
text_result = cleaner.clean_text_data(
    text_data,
    columns=['name', 'description'],
    operations=[
        'lowercase',
        'strip',
        'normalize_whitespace',
        'remove_special_chars'
    ]
)

print("=== TEXT CLEANING ===")
print(f"Columns processed: {text_result['summary']['columns_processed']}")
print(f"Values changed: {text_result['summary']['total_values_changed']}")

# Before and after comparison
print("\nBefore:")
print(text_data[['name', 'description']])
print("\nAfter:")
print(text_result['cleaned_data'][['name', 'description']])

# Custom cleaning patterns
custom_patterns = {
    'phone_cleanup': r'[^\d]',  # Remove non-digits from phone
    'remove_repeated_chars': r'(.)\1{2,}'  # Remove 3+ repeated chars
}

custom_result = cleaner.clean_text_data(
    text_data,
    columns=['phone'],
    custom_patterns=custom_patterns
)
```

### Example 6: Comprehensive Cleaning

```python
# Clean all issues at once
comprehensive_result = cleaner.clean_all(sample_data)

print("=== COMPREHENSIVE CLEANING SUMMARY ===")
summary = comprehensive_result['summary']
print(f"Original shape: {summary['original_shape']}")
print(f"Final shape: {summary['final_shape']}")
print(f"Operations performed: {', '.join(summary['operations_performed'])}")
print(f"Rows removed: {summary['total_rows_removed']}")

# Quality improvement metrics
quality_improvement = summary['data_quality_improvement']
print(f"\nQuality Improvements:")
print(f"Missing data reduction: {quality_improvement['missing_data_reduction']:.1f}%")
print(f"Duplicate reduction: {quality_improvement['duplicate_reduction']:.1f}%")
print(f"Completeness improvement: {quality_improvement['completeness_improvement']:.1f}%")

# Get the cleaned dataset
cleaned_data = comprehensive_result['cleaned_data']
```

---

## Data Validation Examples

### Example 1: Schema Validation

```python
from validata import DataValidator

validator = DataValidator()

# Method 1: Infer schema from data
schema_result = validator.validate_schema(
    sample_data,
    infer_schema=True,
    strict=False
)

print("=== SCHEMA VALIDATION ===")
print(f"Validation passed: {schema_result['validation_passed']}")
print(f"Columns validated: {len(schema_result['validated_columns'])}")

if schema_result['errors']:
    print("Validation errors:")
    for error in schema_result['errors']:
        print(f"  • {error['column']}: {error['message']}")

# Method 2: Define custom schema
import pandera as pa

custom_schema = pa.DataFrameSchema({
    'user_id': pa.Column(pa.Int64, checks=pa.Check.greater_than(0)),
    'name': pa.Column(pa.String, nullable=True),
    'email': pa.Column(pa.String, checks=pa.Check.str_matches(r'^[^@]+@[^@]+\.[^@]+$')),
    'age': pa.Column(pa.Float64, checks=[
        pa.Check.greater_than_or_equal_to(0),
        pa.Check.less_than_or_equal_to(120)
    ]),
    'income': pa.Column(pa.Float64, checks=pa.Check.greater_than(0)),
    'status': pa.Column(pa.String, checks=pa.Check.isin(['active', 'inactive', 'pending']))
})

custom_result = validator.validate_schema(
    sample_data,
    schema=custom_schema,
    strict=True
)
```

### Example 2: Business Rules Validation

```python
# Define comprehensive business rules
business_rules = [
    {
        'name': 'valid_user_id',
        'column': 'user_id',
        'condition': 'not_null'
    },
    {
        'name': 'unique_user_id',
        'column': 'user_id',
        'condition': 'unique'
    },
    {
        'name': 'reasonable_age',
        'column': 'age',
        'condition': 'between',
        'min': 13,
        'max': 120
    },
    {
        'name': 'valid_email_format',
        'column': 'email',
        'condition': 'regex',
        'pattern': r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    },
    {
        'name': 'valid_status',
        'column': 'status',
        'condition': 'isin',
        'values': ['active', 'inactive', 'pending', 'suspended']
    },
    {
        'name': 'positive_income',
        'column': 'income',
        'condition': 'greater_than',
        'value': 0
    }
]

# Apply business rules validation
business_result = validator.validate_business_rules(
    sample_data,
    rules=business_rules
)

print("=== BUSINESS RULES VALIDATION ===")
print(f"Overall passed: {business_result['overall_passed']}")
print(f"Rules passed: {business_result['summary']['passed_rules']}/{business_result['summary']['total_rules']}")
print(f"Total violations: {business_result['total_violations']}")

# Show failed rules
if business_result['failed_rules']:
    print("\nFailed rules:")
    for rule_name in business_result['failed_rules']:
        rule_result = business_result['rule_results'][rule_name]
        print(f"  • {rule_name}: {rule_result['violation_count']} violations")
        print(f"    Message: {rule_result['message']}")
```

### Example 3: Custom Validation Functions

```python
# Define custom validation function
def validate_email_domain(email):
    """Custom validator for allowed email domains."""
    if pd.isna(email):
        return True  # Allow null values
    allowed_domains = ['example.com', 'company.com', 'organization.org']
    try:
        domain = email.split('@')[1]
        return domain in allowed_domains
    except:
        return False

def validate_income_age_ratio(row):
    """Custom validator for income-age relationship."""
    if pd.isna(row['income']) or pd.isna(row['age']):
        return True
    # Simple rule: income should be reasonable for age
    expected_min_income = max(0, (row['age'] - 18) * 1000)
    return row['income'] >= expected_min_income

# Custom validation rules
custom_rules = [
    {
        'name': 'allowed_email_domains',
        'column': 'email',
        'condition': 'custom',
        'function': validate_email_domain
    }
]

custom_validation_result = validator.validate_business_rules(
    sample_data,
    rules=custom_rules
)

# Row-level custom validation (more complex)
sample_data['income_age_valid'] = sample_data.apply(validate_income_age_ratio, axis=1)
```

### Example 4: Statistical Validation

```python
# Perform statistical validation
stats_result = validator.statistical_validation(
    sample_data,
    tests=['normality', 'outliers', 'correlation'],
    significance_level=0.05
)

print("=== STATISTICAL VALIDATION ===")

# Normality tests
if 'normality' in stats_result['test_results']:
    print("\nNormality Tests (Shapiro-Wilk):")
    for col, result in stats_result['test_results']['normality'].items():
        status = "Normal" if result['is_normal'] else "Non-normal"
        print(f"  {col}: {status} (p={result['p_value']:.4f})")

# Outlier detection
if 'outliers' in stats_result['test_results']:
    print("\nOutlier Detection (IQR method):")
    for col, result in stats_result['test_results']['outliers'].items():
        if result['outlier_count'] > 0:
            print(f"  {col}: {result['outlier_count']} outliers ({result['outlier_percentage']:.1f}%)")

# Correlation analysis
if 'correlation' in stats_result['test_results']:
    high_corrs = stats_result['test_results']['correlation']['high_correlations']
    if high_corrs:
        print("\nHigh Correlations (>0.8):")
        for corr in high_corrs:
            print(f"  {corr['column1']} ↔ {corr['column2']}: {corr['correlation']:.3f}")

# Overall summary
print(f"\nStatistical Summary:")
for issue in stats_result['overall_summary']['issues_found']:
    print(f"  • {issue}")
```

### Example 5: Comparative Validation

```python
# Create reference dataset for comparison
reference_data = sample_data.sample(500).copy()

# Compare distributions
comparison_result = validator.statistical_validation(
    sample_data,
    reference_data=reference_data,
    tests=['distribution_comparison', 'mean_comparison'],
    significance_level=0.05
)

print("=== COMPARATIVE VALIDATION ===")

# Distribution comparison (Kolmogorov-Smirnov test)
if 'distribution_comparison' in comparison_result['test_results']:
    print("\nDistribution Comparison:")
    for col, result in comparison_result['test_results']['distribution_comparison'].items():
        similar = "Similar" if result['distributions_similar'] else "Different"
        print(f"  {col}: {similar} (p={result['p_value']:.4f})")

# Mean comparison (t-test)
if 'mean_comparison' in comparison_result['test_results']:
    print("\nMean Comparison:")
    for col, result in comparison_result['test_results']['mean_comparison'].items():
        similar = "Similar" if result['means_similar'] else "Different"
        print(f"  {col}: {similar} (diff={result['mean_difference']:.2f}, p={result['p_value']:.4f})")
```

---

## Schema Generation Examples

### Example 1: Basic Schema Inference

```python
from validata import SchemaGenerator

generator = SchemaGenerator()

# Infer schema from data
schema_result = generator.infer_schema(
    sample_data,
    table_name="customers",
    infer_constraints=True,
    include_relationships=False
)

print("=== SCHEMA INFERENCE ===")
print(f"Table name: {schema_result['schema']['table_name']}")
print(f"Columns: {len(schema_result['schema']['columns'])}")

# Display column information
for col_name, col_info in schema_result['schema']['columns'].items():
    print(f"\n{col_name}:")
    print(f"  Type: {col_info['data_type']}")
    print(f"  Nullable: {col_info['nullable']}")
    if col_info['constraints']:
        print(f"  Constraints: {col_info['constraints']}")
```

### Example 2: Generate SQLModel

```python
# Generate SQLModel code
sqlmodel_result = generator.generate_sqlmodel(
    schema_result['schema'],
    table_name="Customer"
)

print("=== GENERATED SQLMODEL ===")
print(sqlmodel_result['model_code'])

# Save to file
with open('customer_model.py', 'w') as f:
    f.write(sqlmodel_result['model_code'])
```

### Example 3: Generate Pydantic Model

```python
# Generate Pydantic model
pydantic_result = generator.generate_pydantic_model(
    schema_result['schema'],
    table_name="Customer"
)

print("=== GENERATED PYDANTIC MODEL ===")
print(pydantic_result['model_code'])
```

### Example 4: Generate Database DDL

```python
# Generate PostgreSQL DDL
postgres_ddl = generator.generate_database_ddl(
    schema_result['schema'],
    table_name="customers",
    dialect="postgresql"
)

print("=== POSTGRESQL DDL ===")
print(postgres_ddl['ddl_code'])

# Generate MySQL DDL
mysql_ddl = generator.generate_database_ddl(
    schema_result['schema'],
    table_name="customers",
    dialect="mysql"
)

print("\n=== MYSQL DDL ===")
print(mysql_ddl['ddl_code'])
```

### Example 5: Advanced Schema with Relationships

```python
# Create related datasets
orders_data = pd.DataFrame({
    'order_id': range(1, 501),
    'customer_id': np.random.choice(sample_data['user_id'], 500),
    'order_date': pd.date_range('2023-01-01', periods=500, freq='D'),
    'amount': np.random.uniform(10, 1000, 500),
    'status': np.random.choice(['pending', 'completed', 'cancelled'], 500)
})

# Infer schema with relationships
orders_schema = generator.infer_schema(
    orders_data,
    table_name="orders",
    infer_constraints=True,
    include_relationships=True
)

# Check for potential foreign keys
if orders_schema['schema']['relationships']['potential_foreign_keys']:
    print("=== POTENTIAL RELATIONSHIPS ===")
    for fk in orders_schema['schema']['relationships']['potential_foreign_keys']:
        print(f"  {fk['column']} → {fk['referenced_table']}.{fk['referenced_column']}")
```

---

## Complete Workflows

### Example 1: End-to-End Data Pipeline

```python
from validata import ValidataAPI

# Initialize API
validata = ValidataAPI()

# Create named workflow
validata.create_workflow("customer_data_pipeline")

print("=== STARTING COMPLETE DATA PIPELINE ===")

# Step 1: Load and profile data
print("\n1. Loading and profiling data...")
validata.load_data(sample_data)
profile_result = validata.profile_data(title="Raw Customer Data", minimal=False)

print(f"   Original shape: {profile_result['summary']['shape']}")
print(f"   Quality score: {profile_result['quality_report']['quality_score']}/100")
print(f"   Issues found: {profile_result['quality_report']['issues_found']}")

# Step 2: Clean data
print("\n2. Cleaning data...")
clean_result = validata.clean_data(strategy='auto')
print(f"   Rows removed: {clean_result['summary']['total_rows_removed']}")

# Step 3: Standardize data
print("\n3. Standardizing data...")
std_result = validata.standardize_data(method='auto')
print(f"   Columns standardized: {len(std_result['summary']['operations_performed'])}")

# Step 4: Validate data
print("\n4. Validating data...")
val_result = validata.validate_data(validation_type='comprehensive')
print(f"   Validation passed: {val_result['validation_passed']}")

# Step 5: Generate schema
print("\n5. Generating schema...")
schema_result = validata.generate_schema("customers", schema_type='with_models')
print(f"   Schema generated with {len(schema_result['schema']['columns'])} columns")

# Get final results
final_data = validata.get_current_data()
summary = validata.get_data_summary()

print(f"\n=== PIPELINE COMPLETED ===")
print(f"Final shape: {summary['shape']}")
print(f"Missing values: {sum(summary['missing_values'].values())}")

# Export everything
export_dir = validata.export_results("outputs/customer_pipeline")
workflow_file = validata.save_workflow("outputs/customer_pipeline.json")

print(f"\nResults exported to: {export_dir}")
print(f"Workflow saved to: {workflow_file}")
```

### Example 2: Data Quality Assessment Workflow

```python
def assess_data_quality(data, title="Data Quality Assessment"):
    """Complete data quality assessment workflow."""
    
    print(f"=== {title.upper()} ===")
    
    # Initialize tools
    profiler = DataProfiler()
    cleaner = DataCleaner()
    validator = DataValidator()
    
    # 1. Profile data
    print("\n1. Profiling data...")
    profile = profiler.profile_data(data, title=title)
    
    initial_quality = profile['quality_report']['quality_score']
    print(f"   Initial quality score: {initial_quality}/100")
    
    # 2. Analyze specific issues
    print("\n2. Analyzing data quality issues...")
    
    # Missing data analysis
    missing_pct = profile['summary']['missing_cells_percent']
    if missing_pct > 5:
        print(f"   ⚠️ High missing data: {missing_pct:.1f}%")
    
    # Duplicate analysis
    dup_pct = profile['summary']['duplicate_rows_percent']
    if dup_pct > 1:
        print(f"   ⚠️ Duplicate rows: {dup_pct:.1f}%")
    
    # Column-level issues
    problem_columns = []
    for col, info in profile['columns'].items():
        if info['quality_issues']:
            problem_columns.append(col)
            print(f"   ⚠️ {col}: {', '.join(info['quality_issues'])}")
    
    # 3. Statistical validation
    print("\n3. Statistical validation...")
    stats_result = validator.statistical_validation(
        data,
        tests=['normality', 'outliers', 'correlation']
    )
    
    # Report statistical issues
    if 'normality' in stats_result['test_results']:
        non_normal = sum(1 for result in stats_result['test_results']['normality'].values() 
                        if not result['is_normal'])
        if non_normal > 0:
            print(f"   ⚠️ {non_normal} columns fail normality test")
    
    # 4. Generate recommendations
    print("\n4. Quality improvement recommendations:")
    recommendations = []
    
    if missing_pct > 5:
        recommendations.append("Consider imputation strategies for missing data")
    if dup_pct > 1:
        recommendations.append("Remove duplicate records")
    if problem_columns:
        recommendations.append(f"Address quality issues in: {', '.join(problem_columns[:3])}")
    
    if not recommendations:
        recommendations.append("Data quality is good - proceed with analysis")
    
    for i, rec in enumerate(recommendations, 1):
        print(f"   {i}. {rec}")
    
    # 5. Quality score and summary
    quality_grade = "A" if initial_quality >= 90 else "B" if initial_quality >= 75 else "C" if initial_quality >= 60 else "D"
    
    print(f"\n=== QUALITY ASSESSMENT SUMMARY ===")
    print(f"Overall Quality Score: {initial_quality}/100 (Grade: {quality_grade})")
    print(f"Issues Found: {profile['quality_report']['issues_found']}")
    print(f"Problem Columns: {len(problem_columns)}")
    
    return {
        'quality_score': initial_quality,
        'grade': quality_grade,
        'issues_found': profile['quality_report']['issues_found'],
        'problem_columns': problem_columns,
        'recommendations': recommendations,
        'profile': profile,
        'statistical_results': stats_result
    }

# Run quality assessment
quality_report = assess_data_quality(sample_data, "Customer Database")
```

### Example 3: Data Preparation for ML Pipeline

```python
def prepare_data_for_ml(data, target_column=None):
    """Prepare data for machine learning pipeline."""
    
    print("=== ML DATA PREPARATION PIPELINE ===")
    
    # Initialize tools
    cleaner = DataCleaner()
    standardizer = DataStandardizer()
    validator = DataValidator()
    
    original_shape = data.shape
    current_data = data.copy()
    
    # 1. Clean data
    print("\n1. Data cleaning...")
    clean_result = cleaner.clean_all(current_data)
    current_data = clean_result['cleaned_data']
    print(f"   Shape after cleaning: {current_data.shape}")
    
    # 2. Handle categorical variables
    print("\n2. Encoding categorical variables...")
    cat_columns = current_data.select_dtypes(include=['object', 'category']).columns.tolist()
    if target_column and target_column in cat_columns:
        cat_columns.remove(target_column)  # Don't encode target
    
    if cat_columns:
        encode_result = standardizer.encode_categorical(
            current_data,
            columns=cat_columns,
            method='onehot'
        )
        current_data = encode_result['standardized_data']
        print(f"   Encoded {len(cat_columns)} categorical columns")
    
    # 3. Standardize numerical features
    print("\n3. Standardizing numerical features...")
    num_columns = current_data.select_dtypes(include=[np.number]).columns.tolist()
    if target_column and target_column in num_columns:
        num_columns.remove(target_column)  # Don't standardize target
    
    if num_columns:
        std_result = standardizer.standardize_numerical(
            current_data,
            columns=num_columns,
            method='standard'
        )
        current_data = std_result['standardized_data']
        print(f"   Standardized {len(num_columns)} numerical columns")
    
    # 4. Final validation
    print("\n4. Final validation...")
    val_result = validator.validate_schema(current_data, infer_schema=True)
    print(f"   Validation passed: {val_result['validation_passed']}")
    
    # 5. ML readiness checks
    print("\n5. ML readiness checks...")
    
    # Check for remaining missing values
    missing_count = current_data.isnull().sum().sum()
    if missing_count > 0:
        print(f"   ⚠️ {missing_count} missing values remain")
    else:
        print("   ✅ No missing values")
    
    # Check for infinite values
    numeric_data = current_data.select_dtypes(include=[np.number])
    infinite_count = np.isinf(numeric_data).sum().sum()
    if infinite_count > 0:
        print(f"   ⚠️ {infinite_count} infinite values found")
    else:
        print("   ✅ No infinite values")
    
    # Check data types
    non_numeric_cols = current_data.select_dtypes(exclude=[np.number]).columns.tolist()
    if target_column and target_column in non_numeric_cols:
        non_numeric_cols.remove(target_column)
    
    if non_numeric_cols:
        print(f"   ⚠️ Non-numeric columns remain: {non_numeric_cols}")
    else:
        print("   ✅ All features are numeric")
    
    print(f"\n=== ML PREPARATION SUMMARY ===")
    print(f"Original shape: {original_shape}")
    print(f"Final shape: {current_data.shape}")
    print(f"Features added: {current_data.shape[1] - original_shape[1]}")
    print(f"Ready for ML: {'Yes' if missing_count == 0 and infinite_count == 0 else 'No'}")
    
    return {
        'data': current_data,
        'original_shape': original_shape,
        'final_shape': current_data.shape,
        'missing_values': missing_count,
        'infinite_values': infinite_count,
        'ml_ready': missing_count == 0 and infinite_count == 0,
        'cleaning_summary': clean_result['summary']
    }

# Prepare data for ML (excluding target column if exists)
ml_data = prepare_data_for_ml(sample_data, target_column='premium')
```

---

## Advanced Use Cases

### Example 1: Custom Data Quality Rules

```python
class CustomDataQualityRules:
    """Custom data quality rules for specific business domain."""
    
    @staticmethod
    def validate_customer_data(data):
        """Validate customer data with business-specific rules."""
        
        validator = DataValidator()
        
        # Define business rules
        customer_rules = [
            # Basic data integrity
            {
                'name': 'user_id_required',
                'column': 'user_id',
                'condition': 'not_null'
            },
            {
                'name': 'user_id_unique',
                'column': 'user_id',
                'condition': 'unique'
            },
            
            # Email validation
            {
                'name': 'email_format',
                'column': 'email',
                'condition': 'regex',
                'pattern': r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
            },
            
            # Age validation
            {
                'name': 'legal_age',
                'column': 'age',
                'condition': 'greater_than',
                'value': 13
            },
            {
                'name': 'reasonable_age',
                'column': 'age',
                'condition': 'less_than',
                'value': 120
            },
            
            # Income validation
            {
                'name': 'positive_income',
                'column': 'income',
                'condition': 'greater_than',
                'value': 0
            },
            {
                'name': 'reasonable_income',
                'column': 'income',
                'condition': 'less_than',
                'value': 10000000  # 10M max
            },
            
            # Status validation
            {
                'name': 'valid_status',
                'column': 'status',
                'condition': 'isin',
                'values': ['active', 'inactive', 'pending', 'suspended']
            }
        ]
        
        # Apply validation
        result = validator.validate_business_rules(data, rules=customer_rules)
        
        # Custom scoring
        total_rules = len(customer_rules)
        passed_rules = result['summary']['passed_rules']
        quality_score = (passed_rules / total_rules) * 100
        
        # Generate detailed report
        report = {
            'overall_score': quality_score,
            'grade': 'A' if quality_score >= 95 else 'B' if quality_score >= 85 else 'C' if quality_score >= 70 else 'F',
            'rules_passed': f"{passed_rules}/{total_rules}",
            'critical_failures': [],
            'warnings': [],
            'recommendations': []
        }
        
        # Analyze failures
        for rule_name in result['failed_rules']:
            rule_result = result['rule_results'][rule_name]
            
            # Critical failures
            if rule_name in ['user_id_required', 'user_id_unique']:
                report['critical_failures'].append({
                    'rule': rule_name,
                    'violations': rule_result['violation_count'],
                    'message': rule_result['message']
                })
            
            # Warnings for other failures
            else:
                report['warnings'].append({
                    'rule': rule_name,
                    'violations': rule_result['violation_count'],
                    'message': rule_result['message']
                })
        
        # Generate recommendations
        if report['critical_failures']:
            report['recommendations'].append("🚨 Fix critical data integrity issues before proceeding")
        
        if len(report['warnings']) > 3:
            report['recommendations'].append("📋 Consider data cleaning to address multiple quality issues")
        
        if quality_score < 85:
            report['recommendations'].append("🔍 Investigate data sources for systematic quality issues")
        
        return report

    @staticmethod
    def validate_financial_data(data):
        """Validate financial data with specific rules."""
        # Implementation for financial data validation
        pass

# Apply custom validation
custom_report = CustomDataQualityRules.validate_customer_data(sample_data)

print("=== CUSTOM BUSINESS VALIDATION ===")
print(f"Quality Score: {custom_report['overall_score']:.1f}/100 (Grade: {custom_report['grade']})")
print(f"Rules Passed: {custom_report['rules_passed']}")

if custom_report['critical_failures']:
    print("\n🚨 CRITICAL FAILURES:")
    for failure in custom_report['critical_failures']:
        print(f"  • {failure['rule']}: {failure['violations']} violations")

if custom_report['warnings']:
    print("\n⚠️ WARNINGS:")
    for warning in custom_report['warnings']:
        print(f"  • {warning['rule']}: {warning['violations']} violations")

print("\n📋 RECOMMENDATIONS:")
for rec in custom_report['recommendations']:
    print(f"  • {rec}")
```

### Example 2: Data Pipeline Monitoring

```python
class DataPipelineMonitor:
    """Monitor data pipeline quality over time."""
    
    def __init__(self):
        self.history = []
        self.thresholds = {
            'quality_score': 80,
            'missing_data_pct': 5,
            'duplicate_pct': 1,
            'outlier_pct': 10
        }
    
    def monitor_batch(self, data, batch_id, timestamp=None):
        """Monitor a batch of data."""
        
        if timestamp is None:
            timestamp = datetime.now()
        
        # Profile data
        profiler = DataProfiler()
        profile = profiler.profile_data(data, title=f"Batch {batch_id}")
        
        # Extract metrics
        metrics = {
            'batch_id': batch_id,
            'timestamp': timestamp,
            'shape': profile['summary']['shape'],
            'quality_score': profile['quality_report']['quality_score'],
            'missing_data_pct': profile['summary']['missing_cells_percent'],
            'duplicate_pct': profile['summary']['duplicate_rows_percent'],
            'issues_count': profile['quality_report']['issues_found']
        }
        
        # Calculate outlier percentage
        validator = DataValidator()
        stats_result = validator.statistical_validation(data, tests=['outliers'])
        if 'outliers' in stats_result['test_results']:
            total_outliers = sum(result['outlier_count'] 
                               for result in stats_result['test_results']['outliers'].values())
            total_values = data.select_dtypes(include=[np.number]).size
            metrics['outlier_pct'] = (total_outliers / max(1, total_values)) * 100
        else:
            metrics['outlier_pct'] = 0
        
        # Check thresholds
        alerts = []
        if metrics['quality_score'] < self.thresholds['quality_score']:
            alerts.append(f"Quality score below threshold: {metrics['quality_score']:.1f} < {self.thresholds['quality_score']}")
        
        if metrics['missing_data_pct'] > self.thresholds['missing_data_pct']:
            alerts.append(f"Missing data above threshold: {metrics['missing_data_pct']:.1f}% > {self.thresholds['missing_data_pct']}%")
        
        if metrics['duplicate_pct'] > self.thresholds['duplicate_pct']:
            alerts.append(f"Duplicate rate above threshold: {metrics['duplicate_pct']:.1f}% > {self.thresholds['duplicate_pct']}%")
        
        if metrics['outlier_pct'] > self.thresholds['outlier_pct']:
            alerts.append(f"Outlier rate above threshold: {metrics['outlier_pct']:.1f}% > {self.thresholds['outlier_pct']}%")
        
        metrics['alerts'] = alerts
        metrics['alert_count'] = len(alerts)
        
        # Store history
        self.history.append(metrics)
        
        # Print monitoring report
        print(f"=== BATCH {batch_id} MONITORING ===")
        print(f"Timestamp: {timestamp}")
        print(f"Shape: {metrics['shape']}")
        print(f"Quality Score: {metrics['quality_score']:.1f}/100")
        print(f"Missing Data: {metrics['missing_data_pct']:.1f}%")
        print(f"Duplicates: {metrics['duplicate_pct']:.1f}%")
        print(f"Outliers: {metrics['outlier_pct']:.1f}%")
        
        if alerts:
            print(f"\n🚨 ALERTS ({len(alerts)}):")
            for alert in alerts:
                print(f"  • {alert}")
        else:
            print(f"\n✅ No alerts - quality within thresholds")
        
        return metrics
    
    def get_trend_analysis(self):
        """Analyze trends across batches."""
        
        if len(self.history) < 2:
            return "Insufficient data for trend analysis"
        
        df = pd.DataFrame(self.history)
        
        # Calculate trends
        trends = {}
        for metric in ['quality_score', 'missing_data_pct', 'duplicate_pct', 'outlier_pct']:
            if len(df) >= 3:
                recent_avg = df[metric].tail(3).mean()
                overall_avg = df[metric].mean()
                trends[metric] = {
                    'recent_avg': recent_avg,
                    'overall_avg': overall_avg,
                    'trend': 'improving' if recent_avg > overall_avg else 'declining' if recent_avg < overall_avg else 'stable'
                }
        
        print("=== PIPELINE TREND ANALYSIS ===")
        for metric, trend_data in trends.items():
            print(f"{metric}: {trend_data['trend']} (recent: {trend_data['recent_avg']:.1f}, overall: {trend_data['overall_avg']:.1f})")
        
        return trends

# Initialize monitor
monitor = DataPipelineMonitor()

# Simulate monitoring multiple batches
for i in range(5):
    # Simulate data changes over time
    batch_data = sample_data.copy()
    
    # Gradually introduce more issues
    if i > 2:
        # Add more missing data
        missing_indices = np.random.choice(batch_data.index, size=50*i, replace=False)
        batch_data.loc[missing_indices, 'name'] = None
    
    if i > 3:
        # Add duplicates
        duplicate_data = batch_data.sample(20).copy()
        batch_data = pd.concat([batch_data, duplicate_data], ignore_index=True)
    
    # Monitor batch
    batch_metrics = monitor.monitor_batch(batch_data, f"BATCH_{i+1}")
    print()

# Analyze trends
trends = monitor.get_trend_analysis()
```

---

## Integration Examples

### Example 1: Integration with Apache Airflow

```python
from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime, timedelta
from validata import ValidataAPI

def validate_daily_data(**context):
    """Airflow task to validate daily data."""
    
    # Get execution date
    execution_date = context['execution_date']
    date_str = execution_date.strftime('%Y-%m-%d')
    
    # Initialize Validata
    validata = ValidataAPI()
    
    # Load daily data
    data_path = f"/data/daily/customers_{date_str}.csv"
    
    try:
        # Create workflow
        validata.create_workflow(f"daily_validation_{date_str}")
        
        # Load and validate
        validata.load_data(data_path)
        profile_result = validata.profile_data(title=f"Daily Data {date_str}")
        validation_result = validata.validate_data(validation_type='comprehensive')
        
        # Check quality thresholds
        quality_score = profile_result['quality_report']['quality_score']
        validation_passed = validation_result['validation_passed']
        
        if quality_score < 80 or not validation_passed:
            raise ValueError(f"Data quality below threshold: score={quality_score}, validation={validation_passed}")
        
        # Export results
        export_dir = validata.export_results(f"/data/validation_reports/daily_{date_str}")
        
        # Log success
        print(f"Daily validation completed successfully for {date_str}")
        print(f"Quality score: {quality_score}/100")
        print(f"Validation passed: {validation_passed}")
        print(f"Report exported to: {export_dir}")
        
        return {
            'date': date_str,
            'quality_score': quality_score,
            'validation_passed': validation_passed,
            'export_path': export_dir
        }
        
    except Exception as e:
        # Log failure and raise for Airflow to handle
        print(f"Daily validation failed for {date_str}: {str(e)}")
        raise

# Define DAG
default_args = {
    'owner': 'data-team',
    'depends_on_past': False,
    'start_date': datetime(2024, 1, 1),
    'email_on_failure': True,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5)
}

dag = DAG(
    'daily_data_validation',
    default_args=default_args,
    description='Daily data validation pipeline',
    schedule_interval='@daily',
    catchup=False
)

# Define tasks
validation_task = PythonOperator(
    task_id='validate_daily_data',
    python_callable=validate_daily_data,
    dag=dag
)
```

### Example 2: Integration with FastAPI

```python
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
import pandas as pd
import io
from validata import ValidataAPI, DataProfiler, DataValidator

app = FastAPI(title="Validata API Service", version="1.0.0")

@app.post("/profile")
async def profile_data(file: UploadFile = File(...)):
    """Profile uploaded data file."""
    
    try:
        # Read uploaded file
        contents = await file.read()
        
        # Determine file type and read data
        if file.filename.endswith('.csv'):
            data = pd.read_csv(io.StringIO(contents.decode('utf-8')))
        elif file.filename.endswith('.json'):
            data = pd.read_json(io.StringIO(contents.decode('utf-8')))
        else:
            raise HTTPException(status_code=400, detail="Unsupported file type")
        
        # Profile data
        profiler = DataProfiler()
        profile_result = profiler.profile_data(
            data,
            title=f"Profile of {file.filename}",
            minimal=False
        )
        
        return JSONResponse(content=profile_result)
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Profiling failed: {str(e)}")

@app.post("/validate")
async def validate_data(file: UploadFile = File(...)):
    """Validate uploaded data file."""
    
    try:
        # Read uploaded file
        contents = await file.read()
        data = pd.read_csv(io.StringIO(contents.decode('utf-8')))
        
        # Validate data
        validator = DataValidator()
        validation_result = validator.validate_schema(
            data,
            infer_schema=True,
            strict=False
        )
        
        return JSONResponse(content=validation_result)
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Validation failed: {str(e)}")

@app.post("/pipeline")
async def run_pipeline(file: UploadFile = File(...)):
    """Run complete data pipeline."""
    
    try:
        # Read uploaded file
        contents = await file.read()
        data = pd.read_csv(io.StringIO(contents.decode('utf-8')))
        
        # Run pipeline
        validata = ValidataAPI()
        
        # Create workflow
        validata.create_workflow(f"api_pipeline_{file.filename}")
        
        # Execute pipeline
        validata.load_data(data)
        profile_result = validata.profile_data(title=f"Analysis of {file.filename}")
        clean_result = validata.clean_data(strategy='auto')
        validation_result = validata.validate_data(validation_type='comprehensive')
        schema_result = validata.generate_schema("api_table")
        
        # Get final summary
        final_summary = validata.get_data_summary()
        
        return JSONResponse(content={
            'pipeline_completed': True,
            'profile': profile_result,
            'cleaning': clean_result,
            'validation': validation_result,
            'schema': schema_result,
            'final_summary': final_summary
        })
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Pipeline failed: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
```

### Example 3: Integration with Jupyter Notebooks

```python
# Jupyter notebook integration utilities

class ValidataNotebookHelper:
    """Helper class for using Validata in Jupyter notebooks."""
    
    @staticmethod
    def display_profile_summary(profile_result):
        """Display profile summary with rich formatting."""
        
        from IPython.display import display, HTML
        import matplotlib.pyplot as plt
        
        summary = profile_result['summary']
        
        # Create HTML summary
        html = f"""
        <div style="background-color: #f8f9fa; padding: 15px; border-radius: 5px; margin: 10px 0;">
            <h3>📊 Data Profile Summary</h3>
            <div style="display: flex; flex-wrap: wrap; gap: 20px;">
                <div><strong>Shape:</strong> {summary['shape'][0]:,} rows × {summary['shape'][1]} columns</div>
                <div><strong>Missing Data:</strong> {summary['missing_cells_percent']:.1f}%</div>
                <div><strong>Duplicates:</strong> {summary['duplicate_rows_percent']:.1f}%</div>
                <div><strong>Quality Score:</strong> {profile_result['quality_report']['quality_score']}/100</div>
            </div>
        </div>
        """
        
        display(HTML(html))
        
        # Create visualization
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        
        # Data types distribution
        data_types = summary['data_types']
        axes[0].pie(data_types.values(), labels=data_types.keys(), autopct='%1.1f%%')
        axes[0].set_title('Data Types Distribution')
        
        # Missing data by column
        missing_data = {col: info['missing_percent'] 
                       for col, info in profile_result['columns'].items() 
                       if info['missing_percent'] > 0}
        
        if missing_data:
            axes[1].bar(range(len(missing_data)), list(missing_data.values()))
            axes[1].set_xticks(range(len(missing_data)))
            axes[1].set_xticklabels(list(missing_data.keys()), rotation=45)
            axes[1].set_ylabel('Missing %')
            axes[1].set_title('Missing Data by Column')
        else:
            axes[1].text(0.5, 0.5, 'No Missing Data', ha='center', va='center', transform=axes[1].transAxes)
            axes[1].set_title('Missing Data by Column')
        
        # Quality score gauge
        score = profile_result['quality_report']['quality_score']
        color = 'green' if score >= 80 else 'orange' if score >= 60 else 'red'
        axes[2].bar(['Quality Score'], [score], color=color, alpha=0.7)
        axes[2].set_ylim(0, 100)
        axes[2].set_ylabel('Score')
        axes[2].set_title('Overall Quality Score')
        
        plt.tight_layout()
        plt.show()
    
    @staticmethod
    def display_validation_results(validation_result):
        """Display validation results with formatting."""
        
        from IPython.display import display, HTML
        
        status_color = "green" if validation_result['validation_passed'] else "red"
        status_text = "✅ PASSED" if validation_result['validation_passed'] else "❌ FAILED"
        
        html = f"""
        <div style="background-color: #f8f9fa; padding: 15px; border-radius: 5px; margin: 10px 0;">
            <h3>🔍 Validation Results</h3>
            <div style="font-size: 18px; color: {status_color}; font-weight: bold; margin-bottom: 10px;">
                {status_text}
            </div>
        """
        
        if validation_result['errors']:
            html += "<h4>❌ Errors:</h4><ul>"
            for error in validation_result['errors'][:5]:  # Show first 5 errors
                html += f"<li>{error.get('column', 'Unknown')}: {error.get('message', str(error))}</li>"
            html += "</ul>"
        
        if validation_result['warnings']:
            html += "<h4>⚠️ Warnings:</h4><ul>"
            for warning in validation_result['warnings'][:5]:  # Show first 5 warnings
                html += f"<li>{warning.get('column', 'Unknown')}: {warning.get('message', str(warning))}</li>"
            html += "</ul>"
        
        html += "</div>"
        display(HTML(html))

# Example notebook usage
def notebook_example():
    """Example of using Validata in a Jupyter notebook."""
    
    # Load data
    data = pd.read_csv('data.csv')
    
    # Initialize Validata
    validata = ValidataAPI()
    helper = ValidataNotebookHelper()
    
    print("🚀 Starting Validata Analysis in Jupyter")
    
    # Profile data
    profile_result = validata.profile_data(data, title="Notebook Analysis")
    helper.display_profile_summary(profile_result)
    
    # Validate data
    validation_result = validata.validate_data(data, validation_type='comprehensive')
    helper.display_validation_results(validation_result)
    
    # Clean data if needed
    if not validation_result['validation_passed']:
        print("🧹 Cleaning data due to validation issues...")
        clean_result = validata.clean_data(data, strategy='auto')
        print(f"Cleaning completed. Rows removed: {clean_result['summary']['total_rows_removed']}")
    
    print("✨ Analysis completed!")

# Run in notebook
# notebook_example()
```

This comprehensive examples document covers all major use cases and integration patterns for Validata. Each example includes detailed explanations and can be adapted for specific requirements.