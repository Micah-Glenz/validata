"""
Validata: A comprehensive data validation and schema generation toolkit.

An LLM-friendly API for data validation tasks including:
- Data profiling and quality assessment
- Data cleaning and preprocessing
- Data validation and schema enforcement
- Automatic schema generation
- Data standardization and normalization

Built on top of proven libraries:
- ydata-profiling for comprehensive data profiling
- Pandera for flexible data validation
- SQLModel for schema generation
- scikit-learn for data preprocessing
- Great Expectations for data quality checks
"""

__version__ = "0.1.0"
__author__ = "Validata Development Team"

from .api import ValidataAPI, quick_profile, quick_clean, quick_analyze
from .profiler import DataProfiler
from .cleaner import DataCleaner
from .validator import DataValidator
from .standardizer import DataStandardizer
from .schema_generator import SchemaGenerator

# Main API interface
validata = ValidataAPI()

# Convenience functions for quick access
profile_data = validata.profile_data
clean_data = validata.clean_data
validate_data = validata.validate_data
standardize_data = validata.standardize_data
generate_schema = validata.generate_schema

__all__ = [
    "ValidataAPI",
    "DataProfiler", 
    "DataCleaner",
    "DataValidator",
    "DataStandardizer", 
    "SchemaGenerator",
    "validata",
    "profile_data",
    "clean_data",
    "validate_data", 
    "standardize_data",
    "generate_schema",
    "quick_profile",
    "quick_clean", 
    "quick_analyze"
]