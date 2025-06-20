# Wrangler Data Tools Suite

A comprehensive, LLM-friendly data cleaning, validation, and schema generation toolkit built on top of proven libraries like ydata-profiling, Pandera, SQLModel, and scikit-learn.

## Features

- **Data Profiling**: Comprehensive data analysis and quality assessment
- **Data Cleaning**: Handle missing data, duplicates, outliers, and data type issues
- **Data Standardization**: Numerical scaling, categorical encoding, date normalization
- **Data Validation**: Schema validation, business rules, statistical testing
- **Schema Generation**: Automatic inference and model generation (SQLModel, Pydantic, DDL)
- **LLM-Friendly API**: Simple, chainable methods with clear documentation

## Quick Start

```python
from wrangler import WranglerAPI

# Initialize the API
wrangler = WranglerAPI()

# Method chaining approach
result = (wrangler
    .load_data("data.csv")
    .clean_data()
    .standardize_data()
    .validate_data()
    .generate_schema("my_table"))

# Or use convenience functions
from wrangler import quick_profile, quick_clean, quick_analyze

profile = quick_profile("data.csv")
cleaned_data = quick_clean("data.csv")
analysis = quick_analyze("data.csv", table_name="my_table")
```

## Installation

```bash
pip install -e .
```

## Requirements

- Python 3.8+
- pandas >= 2.0.0
- numpy >= 1.24.0
- ydata-profiling >= 4.0.0
- pandera >= 0.17.0
- sqlmodel >= 0.0.14
- scikit-learn >= 1.3.0

See `requirements.txt` for the complete list.

## License

MIT License