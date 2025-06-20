"""
Schema Generator Module for Wangler

Provides comprehensive schema generation capabilities using SQLModel, Pydantic, and custom
inference algorithms. Automatically generates database schemas, API models, and validation
schemas from data analysis.

Key Features:
- Automatic schema inference from DataFrame analysis
- SQLModel generation for database operations
- Pydantic model generation for API schemas
- Database DDL generation for multiple database engines
- Schema export in multiple formats (JSON, YAML, SQL)
- Constraint inference (nullable, unique, foreign keys)
- Schema validation and optimization
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Union, Tuple, Type
from pathlib import Path
import logging
from datetime import datetime
import json
import yaml

from sqlmodel import SQLModel, Field, create_engine
from sqlalchemy import Column, Integer, String, Float, Boolean, DateTime, Text, ForeignKey
from sqlalchemy.schema import CreateTable
from pydantic import BaseModel, create_model
import pandera as pa

from .utils import ConfigManager, OperationTracker, FileOperations, DataTypeInference

logger = logging.getLogger(__name__)


class SchemaGenerator:
    """
    LLM-friendly schema generation tool with comprehensive schema creation capabilities.
    
    Provides methods for:
    - Automatic schema inference from data analysis
    - SQLModel generation for database operations
    - Pydantic model generation for API schemas
    - Database DDL generation
    - Schema export and validation
    - Constraint inference and optimization
    """
    
    def __init__(self, config_manager: Optional[ConfigManager] = None):
        """
        Initialize the SchemaGenerator.
        
        Args:
            config_manager: Optional configuration manager instance
        """
        self.config_manager = config_manager or ConfigManager()
        self.tracker = OperationTracker()
        self._generated_schemas = {}
        self._schema_history = []
    
    def infer_schema(
        self,
        data: Union[pd.DataFrame, str, Path],
        table_name: str = "generated_table",
        infer_constraints: bool = True,
        include_nullability: bool = True,
        include_relationships: bool = False,
        sample_size: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Infer a comprehensive schema from DataFrame analysis.
        
        Args:
            data: DataFrame or path to data file
            table_name: Name for the generated table/model
            infer_constraints: Whether to infer validation constraints
            include_nullability: Whether to include nullable information
            include_relationships: Whether to attempt relationship inference
            sample_size: Number of rows to sample for analysis
            
        Returns:
            Dictionary containing inferred schema and metadata
            
        Example:
            >>> generator = SchemaGenerator()
            >>> result = generator.infer_schema(df, table_name="users")
            >>> schema = result['schema']
            >>> print(f"Generated schema for {len(schema['columns'])} columns")
        """
        self.tracker.start_operation(
            "infer_schema",
            table_name=table_name,
            infer_constraints=infer_constraints,
            include_nullability=include_nullability,
            include_relationships=include_relationships,
            sample_size=sample_size
        )
        
        try:
            # Load data if path provided
            if isinstance(data, (str, Path)):
                data = FileOperations.load_data(data)
            
            # Sample data if specified
            if sample_size and len(data) > sample_size:
                data_sample = data.sample(sample_size, random_state=42)
            else:
                data_sample = data
            
            schema = {
                'table_name': table_name,
                'columns': {},
                'constraints': {},
                'relationships': {},
                'metadata': {
                    'source_rows': len(data),
                    'sample_rows': len(data_sample),
                    'inference_timestamp': pd.Timestamp.now().isoformat()
                }
            }
            
            # Infer column schemas
            for col_name in data_sample.columns:
                col_schema = self._infer_column_schema(
                    data_sample[col_name], 
                    col_name,
                    infer_constraints,
                    include_nullability
                )
                schema['columns'][col_name] = col_schema
            
            # Infer table-level constraints
            if infer_constraints:
                schema['constraints'] = self._infer_table_constraints(data_sample)
            
            # Infer relationships
            if include_relationships:
                schema['relationships'] = self._infer_relationships(data_sample)
            
            # Generate schema statistics
            schema_stats = {
                'total_columns': len(schema['columns']),
                'nullable_columns': sum(1 for col in schema['columns'].values() if col.get('nullable', True)),
                'constrained_columns': sum(1 for col in schema['columns'].values() if col.get('constraints')),
                'primary_key_candidates': [col for col, info in schema['columns'].items() 
                                         if info.get('could_be_primary_key', False)],
                'data_types_distribution': {}
            }
            
            # Count data types
            for col_info in schema['columns'].values():
                dtype = col_info.get('data_type', 'unknown')
                schema_stats['data_types_distribution'][dtype] = schema_stats['data_types_distribution'].get(dtype, 0) + 1
            
            # Store generated schema
            self._generated_schemas[table_name] = schema
            self._record_schema_operation('infer_schema', {
                'table_name': table_name,
                'columns_inferred': schema_stats['total_columns'],
                'constraints_inferred': len(schema['constraints']),
                'relationships_inferred': len(schema['relationships'])
            })
            
            self.tracker.complete_operation(
                columns_processed=schema_stats['total_columns'],
                constraints_inferred=len(schema['constraints']),
                rows_processed=len(data)
            )
            
            # Get tracker metadata
            tracker_metadata = self.tracker.get_current_metadata()
            
            # Create rich metadata structure
            result = {
                'success': True,
                'data': {
                    'schema': schema,
                    'statistics': schema_stats
                },
                'metadata': {
                    'operation': 'infer_schema',
                    'timestamp': tracker_metadata.get('timestamp', pd.Timestamp.now().isoformat()),
                    'performance': tracker_metadata.get('performance', {}),
                    'parameters': {
                        'table_name': table_name,
                        'infer_constraints': infer_constraints,
                        'include_nullability': include_nullability,
                        'include_relationships': include_relationships,
                        'sample_size': sample_size
                    }
                },
                'errors': [],
                'warnings': [],
                # Legacy compatibility
                'schema': schema,
                'statistics': schema_stats,
                'original_data': data
            }
            
            return result
            
        except Exception as e:
            self.tracker.fail_operation(str(e))
            logger.error(f"Error inferring schema: {str(e)}")
            raise
    
    def _infer_column_schema(
        self, 
        series: pd.Series, 
        col_name: str,
        infer_constraints: bool,
        include_nullability: bool
    ) -> Dict[str, Any]:
        """Infer schema for a single column."""
        col_schema = {
            'name': col_name,
            'original_dtype': str(series.dtype)
        }
        
        # Infer data type
        if pd.api.types.is_integer_dtype(series):
            col_schema['data_type'] = 'integer'
            col_schema['sql_type'] = 'INTEGER'
            col_schema['python_type'] = 'int'
        elif pd.api.types.is_float_dtype(series):
            col_schema['data_type'] = 'float'
            col_schema['sql_type'] = 'FLOAT'
            col_schema['python_type'] = 'float'
        elif pd.api.types.is_bool_dtype(series):
            col_schema['data_type'] = 'boolean'
            col_schema['sql_type'] = 'BOOLEAN'
            col_schema['python_type'] = 'bool'
        elif pd.api.types.is_datetime64_any_dtype(series):
            col_schema['data_type'] = 'datetime'
            col_schema['sql_type'] = 'DATETIME'
            col_schema['python_type'] = 'datetime'
        else:
            # String/text analysis
            if series.dropna().empty:
                col_schema['data_type'] = 'string'
                col_schema['sql_type'] = 'VARCHAR(255)'
                col_schema['python_type'] = 'str'
            else:
                max_length = series.astype(str).str.len().max()
                if max_length > 500:
                    col_schema['data_type'] = 'text'
                    col_schema['sql_type'] = 'TEXT'
                else:
                    col_schema['data_type'] = 'string'
                    col_schema['sql_type'] = f'VARCHAR({min(max_length * 2, 255)})'
                col_schema['python_type'] = 'str'
        
        # Nullability analysis
        if include_nullability:
            null_count = series.isnull().sum()
            col_schema['nullable'] = null_count > 0
            col_schema['null_percentage'] = float(null_count / len(series) * 100)
        
        # Uniqueness analysis
        unique_count = series.nunique()
        col_schema['unique_values'] = int(unique_count)
        col_schema['unique_percentage'] = float(unique_count / len(series) * 100)
        
        # Primary key candidate analysis
        col_schema['could_be_primary_key'] = (
            unique_count == len(series.dropna()) and  # All non-null values are unique
            series.isnull().sum() == 0  # No null values
        )
        
        # Infer constraints
        if infer_constraints:
            col_schema['constraints'] = DataTypeInference.suggest_constraints(
                pd.DataFrame({col_name: series}), col_name
            )
        
        # Pattern analysis for strings
        if col_schema['data_type'] in ['string', 'text'] and not series.dropna().empty:
            col_schema['patterns'] = self._analyze_string_patterns(series)
        
        return col_schema
    
    def _analyze_string_patterns(self, series: pd.Series) -> Dict[str, Any]:
        """Analyze patterns in string data."""
        import re
        
        patterns = {
            'email_like': 0,
            'phone_like': 0,
            'url_like': 0,
            'date_like': 0,
            'numeric_like': 0,
            'contains_spaces': 0,
            'all_uppercase': 0,
            'all_lowercase': 0,
            'mixed_case': 0
        }
        
        valid_values = series.dropna().astype(str)
        if len(valid_values) == 0:
            return patterns
        
        for value in valid_values:
            # Email pattern
            if re.match(r'^[^@]+@[^@]+\.[^@]+$', value):
                patterns['email_like'] += 1
            
            # Phone pattern (basic)
            if re.match(r'^[\+]?[\d\s\-\(\)]{10,}$', value):
                patterns['phone_like'] += 1
            
            # URL pattern (basic)
            if re.match(r'^https?://', value):
                patterns['url_like'] += 1
            
            # Date-like pattern
            if re.match(r'^\d{1,4}[-/]\d{1,2}[-/]\d{1,4}', value):
                patterns['date_like'] += 1
            
            # Numeric-like
            if re.match(r'^[\d\.\,\-\+]+$', value):
                patterns['numeric_like'] += 1
            
            # Case patterns
            if ' ' in value:
                patterns['contains_spaces'] += 1
            if value.isupper():
                patterns['all_uppercase'] += 1
            elif value.islower():
                patterns['all_lowercase'] += 1
            else:
                patterns['mixed_case'] += 1
        
        # Convert to percentages
        total = len(valid_values)
        for key in patterns:
            patterns[key] = round(patterns[key] / total * 100, 2)
        
        return patterns
    
    def _infer_table_constraints(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Infer table-level constraints."""
        constraints = {
            'primary_key_candidates': [],
            'unique_constraints': [],
            'composite_keys': []
        }
        
        # Find primary key candidates
        for col in data.columns:
            if (data[col].nunique() == len(data.dropna(subset=[col])) and 
                data[col].isnull().sum() == 0):
                constraints['primary_key_candidates'].append(col)
        
        # Find unique constraints (columns with all unique values but may have nulls)
        for col in data.columns:
            non_null_data = data[col].dropna()
            if len(non_null_data) > 0 and non_null_data.nunique() == len(non_null_data):
                if col not in constraints['primary_key_candidates']:
                    constraints['unique_constraints'].append(col)
        
        # Find potential composite keys (simplified analysis)
        if len(constraints['primary_key_candidates']) == 0:
            # Try pairs of columns
            for i, col1 in enumerate(data.columns):
                for col2 in data.columns[i+1:]:
                    combined = data[[col1, col2]].dropna()
                    if len(combined) > 0 and len(combined.drop_duplicates()) == len(combined):
                        constraints['composite_keys'].append([col1, col2])
                        break  # Only find first valid composite key
        
        return constraints
    
    def _infer_relationships(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Infer potential relationships between columns."""
        relationships = {
            'potential_foreign_keys': [],
            'hierarchical_relationships': []
        }
        
        # Simple foreign key inference based on naming patterns
        for col in data.columns:
            if col.lower().endswith('_id') or col.lower().endswith('id'):
                # Check if values could reference another table
                unique_ratio = data[col].nunique() / len(data.dropna(subset=[col]))
                if 0.1 < unique_ratio < 0.8:  # Not too unique, not too repetitive
                    relationships['potential_foreign_keys'].append({
                        'column': col,
                        'referenced_table': col.replace('_id', '').replace('id', ''),
                        'confidence': 'medium'
                    })
        
        # Hierarchical relationships (parent-child patterns)
        hierarchy_patterns = ['parent_id', 'manager_id', 'category_id']
        for col in data.columns:
            if any(pattern in col.lower() for pattern in hierarchy_patterns):
                relationships['hierarchical_relationships'].append({
                    'child_column': col,
                    'relationship_type': 'hierarchical',
                    'confidence': 'low'
                })
        
        return relationships
    
    def generate_sqlmodel(
        self,
        schema: Optional[Dict[str, Any]] = None,
        table_name: Optional[str] = None,
        base_class: Optional[Type] = None
    ) -> Dict[str, Any]:
        """
        Generate SQLModel class from schema.
        
        Args:
            schema: Schema dictionary (uses stored schema if None)
            table_name: Table name to use from stored schemas
            base_class: Base class for the model
            
        Returns:
            Dictionary containing generated model and code
            
        Example:
            >>> result = generator.generate_sqlmodel(table_name="users")
            >>> model_class = result['model_class']
            >>> print(result['model_code'])
        """
        self.tracker.start_operation(
            "generate_sqlmodel",
            table_name=table_name,
            has_schema=schema is not None
        )
        
        try:
            # Get schema
            if schema is None:
                if table_name and table_name in self._generated_schemas:
                    schema = self._generated_schemas[table_name]
                else:
                    raise ValueError("No schema provided and no stored schema found")
            
            table_name = schema.get('table_name', table_name or 'GeneratedModel')
            class_name = self._to_class_name(table_name)
            
            # Generate field definitions
            fields = {}
            field_definitions = []
            
            # Find primary key
            primary_key_col = None
            for col_name, col_info in schema['columns'].items():
                if col_info.get('could_be_primary_key', False):
                    primary_key_col = col_name
                    break
            
            for col_name, col_info in schema['columns'].items():
                field_def = self._generate_sqlmodel_field(col_info, col_name == primary_key_col)
                fields[col_name] = field_def['field']
                field_definitions.append(field_def['definition'])
            
            # Generate model code
            imports = [
                "from sqlmodel import SQLModel, Field",
                "from typing import Optional",
                "from datetime import datetime"
            ]
            
            class_definition = f"""
class {class_name}(SQLModel, table=True):
    __tablename__ = "{table_name}"
    
{chr(10).join(f'    {field_def}' for field_def in field_definitions)}
"""
            
            model_code = "\n".join(imports) + "\n" + class_definition
            
            # Create the actual model class dynamically
            namespace = {
                'SQLModel': SQLModel,
                'Field': Field,
                'Optional': Optional,
                'datetime': datetime
            }
            
            exec(model_code, namespace)
            model_class = namespace[class_name]
            
            # Generate summary
            summary = {
                'class_name': class_name,
                'table_name': table_name,
                'fields_generated': len(fields),
                'primary_key': primary_key_col,
                'nullable_fields': sum(1 for field in field_definitions if 'Optional' in field)
            }
            
            self._record_schema_operation('generate_sqlmodel', summary)
            
            self.tracker.complete_operation(
                model_generated=True,
                fields_count=len(fields),
                rows_processed=schema.get('metadata', {}).get('source_rows', 0)
            )
            
            # Get tracker metadata
            tracker_metadata = self.tracker.get_current_metadata()
            
            # Create rich metadata structure
            result = {
                'success': True,
                'data': {
                    'model_class': model_class,
                    'model_code': model_code,
                    'summary': summary,
                    'schema_used': schema
                },
                'metadata': {
                    'operation': 'generate_sqlmodel',
                    'timestamp': tracker_metadata.get('timestamp', pd.Timestamp.now().isoformat()),
                    'performance': tracker_metadata.get('performance', {}),
                    'parameters': {
                        'table_name': table_name,
                        'class_name': class_name
                    }
                },
                'errors': [],
                'warnings': [],
                # Legacy compatibility
                'model_class': model_class,
                'model_code': model_code,
                'summary': summary,
                'schema_used': schema
            }
            
            return result
            
        except Exception as e:
            self.tracker.fail_operation(str(e))
            logger.error(f"Error generating SQLModel: {str(e)}")
            raise
    
    def _generate_sqlmodel_field(self, col_info: Dict[str, Any], is_primary_key: bool = False) -> Dict[str, Any]:
        """Generate SQLModel field definition."""
        col_name = col_info['name']
        data_type = col_info['data_type']
        nullable = col_info.get('nullable', True)
        
        # Map data types to Python types
        type_mapping = {
            'integer': 'int',
            'float': 'float',
            'boolean': 'bool',
            'string': 'str',
            'text': 'str',
            'datetime': 'datetime'
        }
        
        python_type = type_mapping.get(data_type, 'str')
        
        # Build field definition
        field_args = []
        
        if is_primary_key:
            field_args.append('primary_key=True')
            python_type = f"Optional[{python_type}]" if nullable else python_type
        elif nullable:
            python_type = f"Optional[{python_type}]"
            field_args.append('default=None')
        
        # Add constraints
        constraints = col_info.get('constraints', {})
        if 'max_length' in constraints and data_type in ['string', 'text']:
            field_args.append(f"max_length={constraints['max_length']}")
        
        if 'min_value' in constraints and data_type in ['integer', 'float']:
            field_args.append(f"ge={constraints['min_value']}")
        
        if 'max_value' in constraints and data_type in ['integer', 'float']:
            field_args.append(f"le={constraints['max_value']}")
        
        # Generate field definition string
        if field_args:
            field_str = f"Field({', '.join(field_args)})"
        else:
            field_str = "Field()"
        
        definition = f"{col_name}: {python_type} = {field_str}"
        
        return {
            'field': (python_type, field_str),
            'definition': definition
        }
    
    def generate_pydantic_model(
        self,
        schema: Optional[Dict[str, Any]] = None,
        table_name: Optional[str] = None,
        model_type: str = 'base'
    ) -> Dict[str, Any]:
        """
        Generate Pydantic model from schema.
        
        Args:
            schema: Schema dictionary (uses stored schema if None)
            table_name: Table name to use from stored schemas
            model_type: Type of model ('base', 'create', 'update', 'response')
            
        Returns:
            Dictionary containing generated model and code
            
        Example:
            >>> result = generator.generate_pydantic_model(table_name="users", model_type="create")
            >>> model_class = result['model_class']
        """
        self.tracker.start_operation(
            "generate_pydantic_model",
            table_name=table_name,
            model_type=model_type
        )
        
        try:
            # Get schema
            if schema is None:
                if table_name and table_name in self._generated_schemas:
                    schema = self._generated_schemas[table_name]
                else:
                    raise ValueError("No schema provided and no stored schema found")
            
            table_name = schema.get('table_name', table_name or 'GeneratedModel')
            class_name = f"{self._to_class_name(table_name)}{model_type.title()}"
            
            # Generate field definitions based on model type
            fields = {}
            field_definitions = []
            
            for col_name, col_info in schema['columns'].items():
                # Skip primary key for create models
                if model_type == 'create' and col_info.get('could_be_primary_key', False):
                    continue
                
                field_def = self._generate_pydantic_field(col_info, model_type)
                fields[col_name] = field_def['field']
                field_definitions.append(field_def['definition'])
            
            # Generate model code
            imports = [
                "from pydantic import BaseModel, Field",
                "from typing import Optional",
                "from datetime import datetime"
            ]
            
            class_definition = f"""
class {class_name}(BaseModel):
{chr(10).join(f'    {field_def}' for field_def in field_definitions)}
    
    class Config:
        from_attributes = True
"""
            
            model_code = "\n".join(imports) + "\n" + class_definition
            
            # Create the actual model class dynamically
            namespace = {
                'BaseModel': BaseModel,
                'Field': Field,
                'Optional': Optional,
                'datetime': datetime
            }
            
            exec(model_code, namespace)
            model_class = namespace[class_name]
            
            summary = {
                'class_name': class_name,
                'model_type': model_type,
                'fields_generated': len(fields),
                'optional_fields': sum(1 for field in field_definitions if 'Optional' in field)
            }
            
            self._record_schema_operation('generate_pydantic_model', summary)
            
            self.tracker.complete_operation(
                model_generated=True,
                fields_count=len(fields),
                rows_processed=schema.get('metadata', {}).get('source_rows', 0)
            )
            
            # Get tracker metadata
            tracker_metadata = self.tracker.get_current_metadata()
            
            # Create rich metadata structure
            result = {
                'success': True,
                'data': {
                    'model_class': model_class,
                    'model_code': model_code,
                    'summary': summary,
                    'schema_used': schema
                },
                'metadata': {
                    'operation': 'generate_pydantic_model',
                    'timestamp': tracker_metadata.get('timestamp', pd.Timestamp.now().isoformat()),
                    'performance': tracker_metadata.get('performance', {}),
                    'parameters': {
                        'table_name': table_name,
                        'class_name': class_name,
                        'model_type': model_type
                    }
                },
                'errors': [],
                'warnings': [],
                # Legacy compatibility
                'model_class': model_class,
                'model_code': model_code,
                'summary': summary,
                'schema_used': schema
            }
            
            return result
            
        except Exception as e:
            self.tracker.fail_operation(str(e))
            logger.error(f"Error generating Pydantic model: {str(e)}")
            raise
    
    def _generate_pydantic_field(self, col_info: Dict[str, Any], model_type: str) -> Dict[str, Any]:
        """Generate Pydantic field definition."""
        col_name = col_info['name']
        data_type = col_info['data_type']
        nullable = col_info.get('nullable', True)
        
        # Map data types to Python types
        type_mapping = {
            'integer': 'int',
            'float': 'float',
            'boolean': 'bool',
            'string': 'str',
            'text': 'str',
            'datetime': 'datetime'
        }
        
        python_type = type_mapping.get(data_type, 'str')
        
        # Adjust nullability based on model type
        if model_type == 'update':
            nullable = True  # All fields optional in update models
        elif model_type == 'create' and col_info.get('could_be_primary_key', False):
            nullable = True  # Primary keys often auto-generated
        
        if nullable:
            python_type = f"Optional[{python_type}]"
            default = " = None"
        else:
            default = ""
        
        # Add validation using Field
        field_args = []
        constraints = col_info.get('constraints', {})
        
        if 'max_length' in constraints and data_type in ['string', 'text']:
            field_args.append(f"max_length={constraints['max_length']}")
        
        if 'min_value' in constraints and data_type in ['integer', 'float']:
            field_args.append(f"ge={constraints['min_value']}")
        
        if 'max_value' in constraints and data_type in ['integer', 'float']:
            field_args.append(f"le={constraints['max_value']}")
        
        # Generate description from patterns
        patterns = col_info.get('patterns', {})
        if patterns and data_type in ['string', 'text']:
            if patterns.get('email_like', 0) > 50:
                field_args.append('description="Email address"')
            elif patterns.get('phone_like', 0) > 50:
                field_args.append('description="Phone number"')
            elif patterns.get('url_like', 0) > 50:
                field_args.append('description="URL"')
        
        if field_args:
            field_str = f" = Field({', '.join(field_args)})"
        else:
            field_str = default
        
        definition = f"{col_name}: {python_type}{field_str}"
        
        return {
            'field': python_type,
            'definition': definition
        }
    
    def generate_database_ddl(
        self,
        schema: Optional[Dict[str, Any]] = None,
        table_name: Optional[str] = None,
        database_type: str = 'postgresql'
    ) -> Dict[str, Any]:
        """
        Generate database DDL statements from schema.
        
        Args:
            schema: Schema dictionary (uses stored schema if None)
            table_name: Table name to use from stored schemas
            database_type: Target database type ('postgresql', 'mysql', 'sqlite')
            
        Returns:
            Dictionary containing DDL statements and metadata
            
        Example:
            >>> result = generator.generate_database_ddl(table_name="users", database_type="postgresql")
            >>> print(result['create_table_ddl'])
        """
        self.tracker.start_operation(
            "generate_database_ddl",
            table_name=table_name,
            database_type=database_type
        )
        
        try:
            # Get schema
            if schema is None:
                if table_name and table_name in self._generated_schemas:
                    schema = self._generated_schemas[table_name]
                else:
                    raise ValueError("No schema provided and no stored schema found")
            
            table_name = schema.get('table_name', table_name or 'generated_table')
            
            # Generate column definitions
            column_definitions = []
            primary_keys = []
            unique_constraints = []
            foreign_keys = []
            
            for col_name, col_info in schema['columns'].items():
                col_def = self._generate_column_ddl(col_info, database_type)
                column_definitions.append(col_def)
                
                if col_info.get('could_be_primary_key', False):
                    primary_keys.append(col_name)
            
            # Add table constraints
            constraints = schema.get('constraints', {})
            
            # Primary key constraint
            if primary_keys:
                column_definitions.append(f"PRIMARY KEY ({', '.join(primary_keys)})")
            
            # Unique constraints
            for unique_col in constraints.get('unique_constraints', []):
                column_definitions.append(f"UNIQUE ({unique_col})")
            
            # Generate CREATE TABLE statement
            create_table_ddl = f"""CREATE TABLE {table_name} (
{chr(10).join(f'    {col_def},' for col_def in column_definitions[:-1])}
    {column_definitions[-1]}
);"""
            
            # Generate additional statements
            ddl_statements = {
                'create_table': create_table_ddl,
                'indexes': self._generate_index_ddl(schema, table_name, database_type),
                'foreign_keys': self._generate_foreign_key_ddl(schema, table_name, database_type)
            }
            
            summary = {
                'table_name': table_name,
                'database_type': database_type,
                'columns_generated': len(schema['columns']),
                'constraints_generated': len(constraints),
                'indexes_generated': len(ddl_statements['indexes']),
                'foreign_keys_generated': len(ddl_statements['foreign_keys'])
            }
            
            self._record_schema_operation('generate_database_ddl', summary)
            
            self.tracker.complete_operation(
                ddl_generated=True,
                statements_count=len(ddl_statements),
                rows_processed=schema.get('metadata', {}).get('source_rows', 0)
            )
            
            # Get tracker metadata
            tracker_metadata = self.tracker.get_current_metadata()
            
            # Create rich metadata structure
            result = {
                'success': True,
                'data': {
                    'create_table_ddl': create_table_ddl,
                    'all_statements': ddl_statements,
                    'summary': summary,
                    'schema_used': schema
                },
                'metadata': {
                    'operation': 'generate_database_ddl',
                    'timestamp': tracker_metadata.get('timestamp', pd.Timestamp.now().isoformat()),
                    'performance': tracker_metadata.get('performance', {}),
                    'parameters': {
                        'table_name': table_name,
                        'database_type': database_type
                    }
                },
                'errors': [],
                'warnings': [],
                # Legacy compatibility
                'create_table_ddl': create_table_ddl,
                'all_statements': ddl_statements,
                'summary': summary,
                'schema_used': schema
            }
            
            return result
            
        except Exception as e:
            self.tracker.fail_operation(str(e))
            logger.error(f"Error generating database DDL: {str(e)}")
            raise
    
    def _generate_column_ddl(self, col_info: Dict[str, Any], database_type: str) -> str:
        """Generate DDL for a single column."""
        col_name = col_info['name']
        data_type = col_info['data_type']
        nullable = col_info.get('nullable', True)
        
        # Map data types to SQL types by database
        type_mappings = {
            'postgresql': {
                'integer': 'INTEGER',
                'float': 'REAL',
                'boolean': 'BOOLEAN',
                'string': 'VARCHAR(255)',
                'text': 'TEXT',
                'datetime': 'TIMESTAMP'
            },
            'mysql': {
                'integer': 'INT',
                'float': 'FLOAT',
                'boolean': 'BOOLEAN',
                'string': 'VARCHAR(255)',
                'text': 'TEXT',
                'datetime': 'DATETIME'
            },
            'sqlite': {
                'integer': 'INTEGER',
                'float': 'REAL',
                'boolean': 'BOOLEAN',
                'string': 'TEXT',
                'text': 'TEXT',
                'datetime': 'DATETIME'
            }
        }
        
        sql_type = type_mappings.get(database_type, type_mappings['postgresql']).get(data_type, 'TEXT')
        
        # Apply constraints from schema
        constraints = col_info.get('constraints', {})
        if 'max_length' in constraints and data_type == 'string':
            sql_type = f"VARCHAR({constraints['max_length']})"
        
        # Build column definition
        col_def = f"{col_name} {sql_type}"
        
        if not nullable:
            col_def += " NOT NULL"
        
        if col_info.get('could_be_primary_key', False) and database_type != 'sqlite':
            # SQLite handles AUTO INCREMENT differently
            if database_type == 'postgresql':
                col_def += " GENERATED ALWAYS AS IDENTITY"
            elif database_type == 'mysql':
                col_def += " AUTO_INCREMENT"
        
        return col_def
    
    def _generate_index_ddl(self, schema: Dict[str, Any], table_name: str, database_type: str) -> List[str]:
        """Generate index DDL statements."""
        indexes = []
        
        # Create indexes for columns that could benefit
        for col_name, col_info in schema['columns'].items():
            # Index foreign key candidates
            if col_name.lower().endswith('_id') or col_name.lower().endswith('id'):
                if not col_info.get('could_be_primary_key', False):
                    indexes.append(f"CREATE INDEX idx_{table_name}_{col_name} ON {table_name} ({col_name});")
            
            # Index columns with high selectivity
            unique_percentage = col_info.get('unique_percentage', 0)
            if 50 < unique_percentage < 95:  # Good selectivity but not unique
                indexes.append(f"CREATE INDEX idx_{table_name}_{col_name} ON {table_name} ({col_name});")
        
        return indexes
    
    def _generate_foreign_key_ddl(self, schema: Dict[str, Any], table_name: str, database_type: str) -> List[str]:
        """Generate foreign key DDL statements."""
        foreign_keys = []
        
        relationships = schema.get('relationships', {})
        for fk_info in relationships.get('potential_foreign_keys', []):
            col_name = fk_info['column']
            referenced_table = fk_info['referenced_table']
            
            fk_ddl = f"""ALTER TABLE {table_name} 
ADD CONSTRAINT fk_{table_name}_{col_name} 
FOREIGN KEY ({col_name}) REFERENCES {referenced_table}(id);"""
            
            foreign_keys.append(fk_ddl)
        
        return foreign_keys
    
    def export_schema(
        self,
        schema: Optional[Dict[str, Any]] = None,
        table_name: Optional[str] = None,
        file_path: Union[str, Path] = None,
        format: str = 'json'
    ) -> str:
        """
        Export schema to file in various formats.
        
        Args:
            schema: Schema dictionary (uses stored schema if None)
            table_name: Table name to use from stored schemas
            file_path: Output file path
            format: Export format ('json', 'yaml', 'sql')
            
        Returns:
            Path to exported file
        """
        # Get schema
        if schema is None:
            if table_name and table_name in self._generated_schemas:
                schema = self._generated_schemas[table_name]
            else:
                raise ValueError("No schema provided and no stored schema found")
        
        table_name = schema.get('table_name', table_name or 'generated_schema')
        
        if file_path is None:
            file_path = f"{table_name}_schema.{format}"
        
        file_path = Path(file_path)
        file_path.parent.mkdir(parents=True, exist_ok=True)
        
        if format == 'json':
            with open(file_path, 'w') as f:
                json.dump(schema, f, indent=2, default=str)
        elif format == 'yaml':
            with open(file_path, 'w') as f:
                yaml.dump(schema, f, default_flow_style=False)
        elif format == 'sql':
            ddl_result = self.generate_database_ddl(schema)
            with open(file_path, 'w') as f:
                f.write(ddl_result['create_table_ddl'])
                f.write('\n\n-- Indexes\n')
                for index in ddl_result['all_statements']['indexes']:
                    f.write(index + '\n')
                f.write('\n-- Foreign Keys\n')
                for fk in ddl_result['all_statements']['foreign_keys']:
                    f.write(fk + '\n')
        else:
            raise ValueError(f"Unsupported format: {format}")
        
        logger.info(f"Schema exported to {file_path}")
        return str(file_path)
    
    def _to_class_name(self, table_name: str) -> str:
        """Convert table name to class name."""
        # Remove special characters and convert to PascalCase
        import re
        words = re.findall(r'[a-zA-Z0-9]+', table_name)
        return ''.join(word.capitalize() for word in words)
    
    def get_generated_schemas(self) -> Dict[str, Any]:
        """Get information about all generated schemas."""
        return {
            name: {
                'table_name': schema.get('table_name'),
                'column_count': len(schema.get('columns', {})),
                'constraints_count': len(schema.get('constraints', {})),
                'generated_at': schema.get('metadata', {}).get('inference_timestamp')
            }
            for name, schema in self._generated_schemas.items()
        }
    
    def _record_schema_operation(self, operation: str, summary: Dict[str, Any]) -> None:
        """Record schema operation in history."""
        self._schema_history.append({
            'operation': operation,
            'timestamp': pd.Timestamp.now().isoformat(),
            'summary': summary
        })
    
    def get_schema_history(self) -> List[Dict[str, Any]]:
        """Get history of all schema operations performed."""
        return self._schema_history.copy()
    
    def get_operation_summary(self) -> Dict[str, Any]:
        """Get summary of all operations performed."""
        return self.tracker.get_summary()
    
    def export_schema_report(self, file_path: Union[str, Path]) -> str:
        """
        Export comprehensive schema generation report.
        
        Args:
            file_path: Output file path
            
        Returns:
            Path to exported report
        """
        report_data = {
            'schema_history': self._schema_history,
            'generated_schemas': self.get_generated_schemas(),
            'operation_summary': self.tracker.get_summary(),
            'generated_at': pd.Timestamp.now().isoformat()
        }
        
        file_path = Path(file_path)
        file_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(file_path, 'w') as f:
            json.dump(report_data, f, indent=2, default=str)
        
        logger.info(f"Schema report exported to {file_path}")
        return str(file_path)