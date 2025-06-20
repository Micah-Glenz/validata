# Validata - Data Validation and Processing Toolkit

A comprehensive, LLM-friendly data validation, cleaning, and schema generation toolkit built for modern data workflows. Validata provides a unified API for data profiling, cleaning, standardization, validation, and schema generation using proven libraries like Pandera, SQLModel, and scikit-learn.

## 🚀 Features

### Core Capabilities
- **🔍 Data Profiling**: Comprehensive data analysis with custom pandas-based profiler
- **🧹 Data Cleaning**: Advanced missing data handling, duplicate removal, outlier detection
- **📊 Data Standardization**: Numerical scaling, categorical encoding, date normalization
- **✅ Data Validation**: Schema validation, business rules, statistical testing
- **🏗️ Schema Generation**: Automatic inference and model generation (SQLModel, Pydantic, DDL)
- **🔗 Method Chaining**: Fluent API for building data processing pipelines
- **🤖 LLM-Friendly**: Structured outputs optimized for AI/ML workflows

### Advanced Features
- **Custom Validation Rules**: Define and apply business-specific validation logic
- **Statistical Testing**: Normality tests, outlier detection, correlation analysis
- **Great Expectations Integration**: Professional data quality checks
- **Operation Tracking**: Detailed logging and history of all data operations
- **Export Capabilities**: Comprehensive reporting and data export functionality
- **Workflow Management**: Save and replay data processing workflows

## 📋 Table of Contents

- [Quick Start](#quick-start)
- [Installation](#installation)
- [Core Modules](#core-modules)
- [Usage Examples](#usage-examples)
- [API Reference](#api-reference)
- [Configuration](#configuration)
- [Contributing](#contributing)
- [License](#license)

## 🏃 Quick Start

### Simple Functions
```python
from validata import quick_profile, quick_clean, quick_analyze

# Quick data profiling
profile = quick_profile("data.csv")
print(f"Dataset: {profile['summary']['shape']}")
print(f"Missing data: {profile['summary']['missing_cells_percent']:.1f}%")

# Quick data cleaning
cleaned_df = quick_clean("data.csv")
print(f"Cleaned shape: {cleaned_df.shape}")

# Comprehensive analysis
analysis = quick_analyze("data.csv", table_name="users")
print(f"Quality score: {analysis['summary']['data_quality']['overall']:.1f}/100")
```

### Method Chaining Workflow
```python
from validata import ValidataAPI

# Initialize the API
validata = ValidataAPI()

# Build a complete data processing pipeline
result = (validata
    .load_data("data.csv")
    .profile_data(minimal=False)
    .clean_data(strategy='auto')
    .standardize_data(method='auto')
    .validate_data(validation_type='comprehensive')
    .generate_schema("my_table", schema_type='with_models'))

# Get processed data
processed_data = validata.get_current_data()
summary = validata.get_data_summary()
```

### Individual Module Usage
```python
from validata import DataProfiler, DataCleaner, DataValidator

# Data profiling
profiler = DataProfiler()
profile = profiler.profile_data("data.csv", title="My Analysis")

# Data cleaning
cleaner = DataCleaner()
clean_result = cleaner.clean_all("data.csv")
cleaned_data = clean_result['cleaned_data']

# Data validation
validator = DataValidator()
validation = validator.validate_schema(cleaned_data, infer_schema=True)
```

## 📦 Installation

### From Source
```bash
git clone <repository-url>
cd validata
pip install -e .
```

### Dependencies
```bash
pip install -r requirements.txt
```

### Requirements
- **Python**: 3.8+
- **Core**: pandas >= 2.0.0, numpy >= 1.24.0
- **Validation**: pandera >= 0.17.0, great-expectations >= 0.18.0
- **ML**: scikit-learn >= 1.3.0
- **Schema**: sqlmodel >= 0.0.14, pydantic >= 2.0.0

See [`requirements.txt`](requirements.txt) for the complete dependency list.

## 🏗️ Core Modules

### 1. Data Profiler (`validata.profiler`)
**Custom pandas-based profiler for comprehensive data analysis**

```python
from validata import DataProfiler

profiler = DataProfiler()

# Generate comprehensive profile
profile = profiler.profile_data(
    data="data.csv",
    title="Customer Data Analysis",
    minimal=False,
    include_correlations=True
)

# Quick text summary
summary = profiler.quick_profile("data.csv")
print(summary)
```

**Features:**
- Dataset overview and statistics
- Column-by-column analysis with type detection
- Missing data patterns and quality assessment
- Correlation analysis for numerical variables
- Data quality scoring and recommendations
- Export to JSON format

### 2. Data Cleaner (`validata.cleaner`)
**Comprehensive data cleaning with multiple strategies**

```python
from validata import DataCleaner

cleaner = DataCleaner()

# Handle missing data
missing_result = cleaner.handle_missing_data(
    data="data.csv",
    strategy='auto',  # 'auto', 'drop', 'fill', 'interpolate'
    threshold=0.5
)

# Remove duplicates
dup_result = cleaner.remove_duplicates(
    data=df,
    subset=['id', 'email'],
    keep='first'
)

# Handle outliers
outlier_result = cleaner.handle_outliers(
    data=df,
    method='iqr',  # 'iqr', 'zscore', 'isolation_forest', 'auto'
    action='flag'  # 'remove', 'cap', 'flag'
)

# Fix data types
type_result = cleaner.fix_data_types(
    data=df,
    auto_infer=True,
    strict=False
)

# Clean text data
text_result = cleaner.clean_text_data(
    data=df,
    operations=['lowercase', 'strip', 'normalize_whitespace'],
    custom_patterns={'phone': r'\D'}
)

# Comprehensive cleaning
clean_result = cleaner.clean_all(df)
```

### 3. Data Standardizer (`validata.standardizer`)
**Data standardization and preprocessing**

```python
from validata import DataStandardizer

standardizer = DataStandardizer()

# Standardize numerical data
num_result = standardizer.standardize_numerical(
    data=df,
    method='standard',  # 'standard', 'minmax', 'robust'
    columns=['age', 'income']
)

# Encode categorical data
cat_result = standardizer.encode_categorical(
    data=df,
    method='onehot',  # 'onehot', 'label', 'ordinal', 'target'
    columns=['category', 'status']
)

# Standardize dates
date_result = standardizer.standardize_dates(
    data=df,
    columns=['created_date'],
    extract_features=True
)

# Complete standardization
std_result = standardizer.standardize_all(df)
```

### 4. Data Validator (`validata.validator`)
**Comprehensive validation with Pandera and Great Expectations**

```python
from validata import DataValidator

validator = DataValidator()

# Schema validation
schema_result = validator.validate_schema(
    data=df,
    infer_schema=True,
    strict=False
)

# Business rules validation
rules = [
    {'name': 'age_range', 'column': 'age', 'condition': 'between', 'min': 0, 'max': 120},
    {'name': 'email_format', 'column': 'email', 'condition': 'regex', 'pattern': r'^[^@]+@[^@]+\.[^@]+$'},
    {'name': 'required_fields', 'column': 'user_id', 'condition': 'not_null'}
]

business_result = validator.validate_business_rules(df, rules=rules)

# Statistical validation
stats_result = validator.statistical_validation(
    data=df,
    tests=['normality', 'outliers', 'correlation'],
    significance_level=0.05
)
```

### 5. Schema Generator (`validata.schema_generator`)
**Automatic schema inference and model generation**

```python
from validata import SchemaGenerator

generator = SchemaGenerator()

# Infer schema from data
schema_result = generator.infer_schema(
    data=df,
    table_name="users",
    infer_constraints=True,
    include_relationships=True
)

# Generate SQLModel
sqlmodel_code = generator.generate_sqlmodel(
    schema=schema_result['schema'],
    table_name="users"
)

# Generate Pydantic model
pydantic_code = generator.generate_pydantic_model(
    schema=schema_result['schema'],
    table_name="users"
)

# Generate database DDL
ddl_code = generator.generate_database_ddl(
    schema=schema_result['schema'],
    table_name="users",
    dialect="postgresql"
)
```

## 💡 Usage Examples

### Example 1: Data Quality Assessment
```python
from validata import ValidataAPI

# Initialize API
validata = ValidataAPI()

# Load and profile data
profile = validata.profile_data("customer_data.csv", title="Customer Analysis")

print(f"Dataset Overview:")
print(f"- Shape: {profile['summary']['shape']}")
print(f"- Missing data: {profile['summary']['missing_cells_percent']:.1f}%")
print(f"- Duplicates: {profile['summary']['duplicate_rows_percent']:.1f}%")
print(f"- Quality score: {profile['quality_report']['quality_score']}/100")

# Check for quality issues
if profile['quality_report']['issues']:
    print(f"\n⚠️ Quality Issues Found:")
    for issue in profile['quality_report']['issues'][:5]:
        print(f"  • {issue}")
```

### Example 2: Complete Data Pipeline
```python
from validata import ValidataAPI

# Create a comprehensive data processing pipeline
validata = ValidataAPI()

# Create named workflow
validata.create_workflow("customer_data_pipeline")

# Process data
analysis = (validata
    .load_data("raw_customer_data.csv")
    .profile_data(title="Raw Data Analysis")
    .clean_data(strategy='auto')
    .standardize_data(method='auto')
    .validate_data(validation_type='comprehensive')
    .generate_schema("customers", schema_type='with_models'))

# Get results
final_data = validata.get_current_data()
summary = validata.get_data_summary()

# Export everything
export_dir = validata.export_results("outputs/customer_pipeline")
workflow_file = validata.save_workflow("outputs/customer_pipeline.json")

print(f"Pipeline completed successfully!")
print(f"Final data shape: {summary['shape']}")
print(f"Missing values: {sum(summary['missing_values'].values())}")
print(f"Results exported to: {export_dir}")
```

### Example 3: Custom Validation Rules
```python
from validata import DataValidator

validator = DataValidator()

# Define custom business rules
business_rules = [
    {
        'name': 'valid_email',
        'column': 'email',
        'condition': 'regex',
        'pattern': r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    },
    {
        'name': 'reasonable_age',
        'column': 'age',
        'condition': 'between',
        'min': 13,
        'max': 120
    },
    {
        'name': 'valid_status',
        'column': 'status',
        'condition': 'isin',
        'values': ['active', 'inactive', 'pending']
    },
    {
        'name': 'required_id',
        'column': 'user_id',
        'condition': 'not_null'
    }
]

# Apply validation
result = validator.validate_business_rules(df, rules=business_rules)

if result['overall_passed']:
    print("✅ All business rules passed!")
else:
    print(f"❌ {len(result['failed_rules'])} rules failed:")
    for rule_name in result['failed_rules']:
        rule_result = result['rule_results'][rule_name]
        print(f"  • {rule_name}: {rule_result['message']}")
```

### Example 4: Statistical Data Analysis
```python
from validata import DataValidator

validator = DataValidator()

# Perform statistical validation
stats_result = validator.statistical_validation(
    data=df,
    tests=['normality', 'outliers', 'correlation'],
    significance_level=0.05
)

print("Statistical Analysis Results:")

# Normality tests
if 'normality' in stats_result['test_results']:
    print("\n📊 Normality Tests:")
    for col, result in stats_result['test_results']['normality'].items():
        status = "Normal" if result['is_normal'] else "Non-normal"
        print(f"  • {col}: {status} (p={result['p_value']:.4f})")

# Outlier detection
if 'outliers' in stats_result['test_results']:
    print("\n🎯 Outlier Detection:")
    for col, result in stats_result['test_results']['outliers'].items():
        if result['outlier_count'] > 0:
            print(f"  • {col}: {result['outlier_count']} outliers ({result['outlier_percentage']:.1f}%)")

# High correlations
if 'correlation' in stats_result['test_results']:
    high_corrs = stats_result['test_results']['correlation']['high_correlations']
    if high_corrs:
        print("\n🔗 High Correlations:")
        for corr in high_corrs:
            print(f"  • {corr['column1']} ↔ {corr['column2']}: {corr['correlation']:.3f}")
```

## 🔧 Configuration

### Environment Variables
```bash
# Set default configuration path
export VALIDATA_CONFIG_PATH="/path/to/config.yaml"

# Enable debug logging
export VALIDATA_DEBUG=true

# Set default output directory
export VALIDATA_OUTPUT_DIR="/path/to/outputs"
```

### Configuration File (YAML)
```yaml
# validata_config.yaml
profiling:
  default_minimal: false
  include_correlations: true
  sample_size: 10000

cleaning:
  handle_missing: true
  handle_duplicates: true
  fix_data_types: true
  detect_outliers: true
  clean_text: true

standardization:
  numerical_method: "standard"
  categorical_method: "onehot"
  date_features: true

validation:
  default_strict: false
  statistical_tests: ["normality", "outliers", "correlation"]
  significance_level: 0.05

schema_generation:
  infer_constraints: true
  include_relationships: true
  default_nullable: true

logging:
  level: "INFO"
  format: "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
```

### Programmatic Configuration
```python
from validata import ValidataAPI, ConfigManager

# Custom configuration
config = ConfigManager({
    'cleaning': {
        'handle_missing': True,
        'strategy': 'auto'
    },
    'validation': {
        'strict': False,
        'significance_level': 0.01
    }
})

# Initialize with custom config
validata = ValidataAPI(config_manager=config)
```

## 📊 Output Examples

### Profile Report Structure
```json
{
  "title": "Customer Data Analysis",
  "summary": {
    "shape": [1000, 8],
    "n_records": 1000,
    "n_variables": 8,
    "missing_cells_percent": 2.5,
    "duplicate_rows_percent": 0.1,
    "data_types": {
      "numerical": 3,
      "categorical": 4,
      "datetime": 1
    }
  },
  "columns": {
    "user_id": {
      "type": "numerical",
      "missing_percent": 0.0,
      "unique_count": 1000,
      "quality_issues": []
    }
  },
  "quality_report": {
    "quality_score": 85,
    "issues_found": 3,
    "recommendations": [
      "Consider imputation for missing values in email column"
    ]
  }
}
```

### Validation Results Structure
```json
{
  "validation_passed": true,
  "errors": [],
  "warnings": [],
  "summary": {
    "data_shape": [1000, 8],
    "columns_validated": 8,
    "error_count": 0
  }
}
```

## 🛠️ Development

### Running Tests
```bash
# Install development dependencies
pip install -e ".[dev]"

# Run tests
pytest tests/ -v

# Run with coverage
pytest tests/ --cov=validata --cov-report=html
```

### Example Usage
See the [`examples/`](examples/) directory for comprehensive demonstrations:
- [`basic_demo.py`](examples/basic_demo.py) - Basic API usage examples
- [`mvp_demo.py`](mvp_demo.py) - Custom profiler demonstration

### Sample Data
The [`sample_data/`](sample_data/) directory contains test datasets:
- `clean_dataset.csv` - Clean sample data for testing
- `messy_dataset.csv` - Data with quality issues
- Various edge cases for comprehensive testing

## 🤝 Contributing

We welcome contributions! Please see [`CONTRIBUTING.md`](CONTRIBUTING.md) for guidelines.

### Development Setup
```bash
git clone <repository-url>
cd validata
pip install -e ".[dev]"
pre-commit install
```

### Running Tests
```bash
pytest tests/
```

## 📄 License

This project is licensed under the MIT License - see the [`LICENSE`](LICENSE) file for details.

## 🔗 Links

- **Documentation**: [Link to full docs]
- **Examples**: [`examples/`](examples/) directory
- **Issues**: [GitHub Issues]
- **Contributing**: [`CONTRIBUTING.md`](CONTRIBUTING.md)

---

**Validata** - Making data validation and processing simple, reliable, and LLM-friendly. 🚀