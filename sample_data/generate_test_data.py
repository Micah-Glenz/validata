"""
Generate sample test datasets for Wangler testing and demonstrations.

Creates various datasets with different characteristics to test all modules:
- Clean dataset for baseline testing
- Messy dataset with various data quality issues
- Large dataset for performance testing
- Edge case datasets for robustness testing
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random
import string

# Set random seed for reproducibility
np.random.seed(42)
random.seed(42)

def generate_clean_dataset(n_rows=1000):
    """Generate a clean dataset with no data quality issues."""
    
    # Generate user data
    data = {
        'user_id': range(1, n_rows + 1),
        'username': [f'user_{i:05d}' for i in range(1, n_rows + 1)],
        'email': [f'user{i}@example.com' for i in range(1, n_rows + 1)],
        'age': np.random.randint(18, 80, n_rows),
        'salary': np.random.normal(75000, 25000, n_rows).astype(int),
        'department': np.random.choice(['Engineering', 'Sales', 'Marketing', 'HR', 'Finance'], n_rows),
        'hire_date': [
            datetime(2020, 1, 1) + timedelta(days=np.random.randint(0, 1000))
            for _ in range(n_rows)
        ],
        'is_active': np.random.choice([True, False], n_rows, p=[0.85, 0.15]),
        'performance_score': np.random.uniform(1.0, 5.0, n_rows).round(2),
        'years_experience': np.random.randint(0, 20, n_rows)
    }
    
    # Ensure salary is positive
    data['salary'] = np.abs(data['salary'])
    
    return pd.DataFrame(data)

def generate_messy_dataset(n_rows=1000):
    """Generate a messy dataset with various data quality issues."""
    
    # Start with clean data
    data = generate_clean_dataset(n_rows).to_dict('list')
    
    # Introduce missing values (10-15% across different columns)
    missing_indices = {
        'email': np.random.choice(n_rows, size=int(n_rows * 0.1), replace=False),
        'salary': np.random.choice(n_rows, size=int(n_rows * 0.15), replace=False),
        'department': np.random.choice(n_rows, size=int(n_rows * 0.05), replace=False),
        'performance_score': np.random.choice(n_rows, size=int(n_rows * 0.12), replace=False)
    }
    
    for col, indices in missing_indices.items():
        for idx in indices:
            data[col][idx] = None
    
    # Introduce duplicates (5% of records)
    n_duplicates = int(n_rows * 0.05)
    duplicate_indices = np.random.choice(n_rows, size=n_duplicates, replace=False)
    for idx in duplicate_indices:
        if idx < n_rows - 1:
            # Copy some values to create partial duplicates
            data['username'][idx + 1] = data['username'][idx]
            data['email'][idx + 1] = data['email'][idx]
    
    # Introduce data type issues
    inconsistent_indices = np.random.choice(n_rows, size=int(n_rows * 0.08), replace=False)
    for idx in inconsistent_indices:
        # Mixed data types in age column
        if idx % 3 == 0:
            data['age'][idx] = str(data['age'][idx]) + '.0'
        # Negative salaries
        elif idx % 3 == 1:
            data['salary'][idx] = -abs(data['salary'][idx]) if data['salary'][idx] is not None else None
        # Invalid performance scores
        elif idx % 3 == 2:
            data['performance_score'][idx] = 10.5 if data['performance_score'][idx] is not None else None
    
    # Introduce outliers
    outlier_indices = np.random.choice(n_rows, size=int(n_rows * 0.03), replace=False)
    for idx in outlier_indices:
        if idx % 2 == 0 and data['salary'][idx] is not None:
            data['salary'][idx] = 500000  # Extremely high salary
        elif data['age'][idx] is not None:
            data['age'][idx] = 150  # Impossible age
    
    # Introduce text data issues
    text_issue_indices = np.random.choice(n_rows, size=int(n_rows * 0.1), replace=False)
    for idx in text_issue_indices:
        # Inconsistent department names
        if data['department'][idx] == 'Engineering':
            data['department'][idx] = np.random.choice(['engineering', 'Engineering ', ' Engineering', 'ENGINEERING'])
        # Email format issues
        if data['email'][idx] is not None and idx % 2 == 0:
            data['email'][idx] = data['email'][idx].replace('@', '_at_')
    
    # Add some extra problematic columns
    data['comments'] = [
        ''.join(random.choices(string.ascii_letters + string.digits + string.punctuation + '   ', 
                              k=random.randint(0, 200))) if random.random() > 0.3 else None
        for _ in range(n_rows)
    ]
    
    data['phone'] = [
        f"({random.randint(100, 999)}) {random.randint(100, 999)}-{random.randint(1000, 9999)}"
        if random.random() > 0.2 else None
        for _ in range(n_rows)
    ]
    
    return pd.DataFrame(data)

def generate_large_dataset(n_rows=10000):
    """Generate a large dataset for performance testing."""
    
    # Generate large dataset with more complex structure
    data = {
        'id': range(1, n_rows + 1),
        'timestamp': pd.date_range('2020-01-01', periods=n_rows, freq='1H'),
        'user_id': np.random.randint(1, 1000, n_rows),
        'product_id': np.random.randint(1, 500, n_rows),
        'category': np.random.choice(['A', 'B', 'C', 'D', 'E'], n_rows),
        'subcategory': np.random.choice([f'sub_{i}' for i in range(1, 21)], n_rows),
        'value': np.random.exponential(100, n_rows),
        'quantity': np.random.poisson(5, n_rows),
        'price': np.random.gamma(2, 50, n_rows),
        'discount': np.random.beta(2, 5, n_rows),
        'region': np.random.choice(['North', 'South', 'East', 'West', 'Central'], n_rows),
        'country': np.random.choice(['USA', 'Canada', 'Mexico', 'UK', 'Germany', 'France'], n_rows),
        'is_weekend': np.random.choice([True, False], n_rows, p=[0.2, 0.8]),
        'rating': np.random.normal(3.5, 1.2, n_rows).clip(1, 5),
        'description': [
            ' '.join(random.choices(['excellent', 'good', 'average', 'poor', 'product', 'quality', 'service'], 
                                  k=random.randint(3, 10)))
            for _ in range(n_rows)
        ]
    }
    
    return pd.DataFrame(data)

def generate_edge_case_dataset():
    """Generate datasets with edge cases and corner conditions."""
    
    datasets = {}
    
    # 1. Empty dataset
    datasets['empty'] = pd.DataFrame()
    
    # 2. Single column dataset
    datasets['single_column'] = pd.DataFrame({'values': [1, 2, 3, 4, 5]})
    
    # 3. Single row dataset
    datasets['single_row'] = pd.DataFrame({
        'col1': [1], 'col2': ['text'], 'col3': [True], 'col4': [datetime.now()]
    })
    
    # 4. All null dataset
    datasets['all_nulls'] = pd.DataFrame({
        'col1': [None] * 100,
        'col2': [None] * 100,
        'col3': [None] * 100
    })
    
    # 5. All same values
    datasets['all_same'] = pd.DataFrame({
        'constant_int': [42] * 100,
        'constant_str': ['same'] * 100,
        'constant_bool': [True] * 100
    })
    
    # 6. Extreme data types
    datasets['extreme_types'] = pd.DataFrame({
        'tiny_numbers': np.random.uniform(-1e-10, 1e-10, 100),
        'huge_numbers': np.random.uniform(1e10, 1e15, 100),
        'unicode_text': ['🚀' + ''.join(random.choices('αβγδεζηθικλμνξοπρστυφχψω', k=5)) for _ in range(100)],
        'json_like': [f'{{"key": {i}, "value": "data_{i}"}}' for i in range(100)]
    })
    
    # 7. Date edge cases
    datasets['date_edge_cases'] = pd.DataFrame({
        'ancient_dates': pd.date_range('1900-01-01', periods=100, freq='D'),
        'future_dates': pd.date_range('2100-01-01', periods=100, freq='D'),
        'mixed_date_formats': [
            '2023-01-01', '01/02/2023', '2023-03-01T10:30:00', 
            'April 1, 2023', '2023/05/01'
        ] * 20
    })
    
    return datasets

def save_all_datasets():
    """Generate and save all test datasets."""
    
    print("Generating test datasets...")
    
    # Generate datasets
    clean_data = generate_clean_dataset(1000)
    messy_data = generate_messy_dataset(1000)
    large_data = generate_large_dataset(10000)
    edge_cases = generate_edge_case_dataset()
    
    # Save datasets
    clean_data.to_csv('/home/micah/projects/wangler/sample_data/clean_dataset.csv', index=False)
    print("✓ Saved clean_dataset.csv")
    
    messy_data.to_csv('/home/micah/projects/wangler/sample_data/messy_dataset.csv', index=False)
    print("✓ Saved messy_dataset.csv")
    
    large_data.to_csv('/home/micah/projects/wangler/sample_data/large_dataset.csv', index=False)
    print("✓ Saved large_dataset.csv")
    
    # Save edge case datasets
    for name, dataset in edge_cases.items():
        if not dataset.empty:
            dataset.to_csv(f'/home/micah/projects/wangler/sample_data/edge_case_{name}.csv', index=False)
            print(f"✓ Saved edge_case_{name}.csv")
    
    # Generate summary
    summary = {
        'clean_dataset': {
            'rows': len(clean_data),
            'columns': len(clean_data.columns),
            'description': 'Clean dataset with no data quality issues'
        },
        'messy_dataset': {
            'rows': len(messy_data),
            'columns': len(messy_data.columns),
            'description': 'Dataset with missing values, duplicates, outliers, and type issues'
        },
        'large_dataset': {
            'rows': len(large_data),
            'columns': len(large_data.columns),
            'description': 'Large dataset for performance testing'
        },
        'edge_cases': {
            'datasets': len(edge_cases),
            'description': 'Various edge cases and corner conditions'
        }
    }
    
    import json
    with open('/home/micah/projects/wangler/sample_data/dataset_summary.json', 'w') as f:
        json.dump(summary, f, indent=2, default=str)
    print("✓ Saved dataset_summary.json")
    
    print(f"\nGenerated {len(edge_cases) + 3} test datasets successfully!")
    return summary

if __name__ == "__main__":
    summary = save_all_datasets()
    print("\nDataset Summary:")
    for name, info in summary.items():
        print(f"  {name}: {info}")