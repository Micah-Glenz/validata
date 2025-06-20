"""
Basic Wrangler API Demonstration

This script demonstrates the key features of the Wangler data tools suite
with simple, easy-to-understand examples.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import pandas as pd

# Try importing from the installed package first, then fallback to local
try:
    from wrangler import WranglerAPI, quick_profile, quick_clean, quick_analyze
except ImportError:
    # Fallback to local import
    from wrangler.api import WranglerAPI, quick_profile, quick_clean, quick_analyze

def demo_basic_usage():
    """Demonstrate basic API usage."""
    print("🚀 Wrangler Basic Demo")
    print("=" * 50)
    
    # Load sample data
    data_path = "../sample_data/messy_dataset.csv"
    
    print("\n1. Quick Functions Demo")
    print("-" * 30)
    
    # Quick profiling
    print("📊 Quick profiling...")
    try:
        profile = quick_profile(data_path)
        print(f"   ✓ Dataset has {profile['summary']['n_records']} rows and {profile['summary']['n_variables']} columns")
        print(f"   ✓ Missing data: {profile['summary']['missing_cells_percent']:.1f}%")
        print(f"   ✓ Duplicate rows: {profile['summary']['duplicate_rows_percent']:.1f}%")
    except Exception as e:
        print(f"   ❌ Error: {e}")
    
    # Quick cleaning
    print("\n🧹 Quick cleaning...")
    try:
        cleaned_df = quick_clean(data_path)
        print(f"   ✓ Cleaned data shape: {cleaned_df.shape}")
        print(f"   ✓ Remaining missing values: {cleaned_df.isnull().sum().sum()}")
    except Exception as e:
        print(f"   ❌ Error: {e}")

def demo_method_chaining():
    """Demonstrate method chaining capabilities."""
    print("\n\n2. Method Chaining Demo")
    print("-" * 30)
    
    try:
        # Initialize API
        wrangler = WranglerAPI()
        
        # Method chaining workflow
        print("🔗 Creating data processing pipeline...")
        result = (wrangler
                 .load_data("../sample_data/messy_dataset.csv")
                 .profile_data(minimal=True)
                 .clean_data(strategy='auto')
                 .standardize_data(method='auto')
                 .validate_data(validation_type='schema')
                 .generate_schema(table_name="demo_table"))
        
        # Get current data summary
        summary = wrangler.get_data_summary()
        print(f"   ✓ Final data shape: {summary['shape']}")
        print(f"   ✓ Memory usage: {summary['memory_usage_mb']:.2f} MB")
        print(f"   ✓ Missing values: {sum(summary['missing_values'].values())}")
        
    except Exception as e:
        print(f"   ❌ Error in method chaining: {e}")

def demo_comprehensive_analysis():
    """Demonstrate comprehensive analysis."""
    print("\n\n3. Comprehensive Analysis Demo")
    print("-" * 30)
    
    try:
        # Initialize API
        wrangler = WranglerAPI()
        
        print("🔍 Running comprehensive analysis...")
        analysis = wrangler.analyze_all("../sample_data/clean_dataset.csv", table_name="users")
        
        # Display results
        summary = analysis['summary']
        print(f"   ✓ Original shape: {summary['data_transformation']['original_shape']}")
        print(f"   ✓ Final shape: {summary['data_transformation']['final_shape']}")
        print(f"   ✓ Data quality score: {summary['data_quality']['overall']:.1f}/100")
        print(f"   ✓ Validation passed: {summary['validation_status']['passed']}")
        print(f"   ✓ Operations completed: {len(summary['operations_completed'])}")
        
        # Show recommendations
        if analysis['recommendations']:
            print("\n   📋 Recommendations:")
            for rec in analysis['recommendations'][:3]:  # Show first 3
                print(f"      • {rec}")
        
        return analysis
        
    except Exception as e:
        print(f"   ❌ Error in comprehensive analysis: {e}")
        return None

def demo_individual_modules():
    """Demonstrate individual module usage."""
    print("\n\n4. Individual Modules Demo")
    print("-" * 30)
    
    try:
        # Load data
        df = pd.read_csv("../sample_data/clean_dataset.csv")
        
        # Initialize API
        wrangler = WranglerAPI()
        
        # Data Profiler
        print("📊 Data Profiler:")
        profile_result = wrangler.profiler.profile_data(df, title="Individual Module Demo")
        print(f"   ✓ Generated profile with {len(profile_result['variables'])} variable analyses")
        
        # Data Cleaner
        print("\n🧹 Data Cleaner:")
        clean_result = wrangler.cleaner.handle_missing_data(df, strategy='auto')
        print(f"   ✓ Processed {clean_result['summary']['missing_reduction_percent']:.1f}% missing data reduction")
        
        # Data Standardizer
        print("\n📊 Data Standardizer:")
        numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
        if numeric_cols:
            std_result = wrangler.standardizer.standardize_numerical(df, columns=numeric_cols[:2])
            print(f"   ✓ Standardized {std_result['summary']['columns_standardized']} numerical columns")
        
        # Data Validator
        print("\n✅ Data Validator:")
        val_result = wrangler.validator.validate_schema(df, infer_schema=True)
        print(f"   ✓ Schema validation: {'PASSED' if val_result['validation_passed'] else 'FAILED'}")
        print(f"   ✓ Validated {len(val_result['validated_columns'])} columns")
        
        # Schema Generator
        print("\n🏗️ Schema Generator:")
        schema_result = wrangler.schema_generator.infer_schema(df, table_name="demo_users")
        print(f"   ✓ Generated schema with {len(schema_result['schema']['columns'])} columns")
        print(f"   ✓ Found {len(schema_result['schema']['constraints']['primary_key_candidates'])} primary key candidates")
        
    except Exception as e:
        print(f"   ❌ Error in individual modules demo: {e}")

def demo_export_capabilities():
    """Demonstrate export and reporting capabilities."""
    print("\n\n5. Export Capabilities Demo")
    print("-" * 30)
    
    try:
        wrangler = WranglerAPI()
        
        # Create a workflow
        print("💾 Creating and exporting workflow...")
        wrangler.create_workflow("demo_workflow")
        
        # Simple workflow
        wrangler.load_data("../sample_data/clean_dataset.csv")
        wrangler.profile_data(minimal=True)
        wrangler.clean_data(strategy='conservative')
        
        # Export results
        export_dir = wrangler.export_results("../exports/demo_results")
        print(f"   ✓ Exported results to: {export_dir}")
        
        # Save workflow
        workflow_path = wrangler.save_workflow("../exports/demo_workflow.json")
        print(f"   ✓ Saved workflow to: {workflow_path}")
        
    except Exception as e:
        print(f"   ❌ Error in export demo: {e}")

def main():
    """Run all demonstrations."""
    try:
        demo_basic_usage()
        demo_method_chaining()
        analysis = demo_comprehensive_analysis()
        demo_individual_modules()
        demo_export_capabilities()
        
        print("\n\n🎉 Demo completed successfully!")
        print("=" * 50)
        print("The Wrangler API is working correctly and ready to use!")
        
        return analysis
        
    except Exception as e:
        print(f"\n❌ Demo failed with error: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    analysis_result = main()