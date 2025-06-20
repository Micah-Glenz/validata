"""
MVP Demo: Custom Pandas-Based Profiler
Demonstrates the working custom profiler without the problematic chain operations
"""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

import pandas as pd
from wrangler.profiler import CustomDataProfiler
import json

def demonstrate_mvp():
    """Demonstrate the MVP custom profiler capabilities."""
    print("🚀 Wrangler Custom Profiler MVP Demo")
    print("=" * 50)
    
    # Initialize profiler
    profiler = CustomDataProfiler()
    print("✓ Custom DataProfiler initialized")
    
    # Demo 1: Simple dataset
    print("\n📊 Demo 1: Simple Dataset Analysis")
    print("-" * 30)
    simple_data = pd.DataFrame({
        'user_id': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        'age': [25, 30, 35, 28, 32, 45, 29, 33, 27, 31],
        'department': ['Sales', 'HR', 'Sales', 'IT', 'HR', 'IT', 'Sales', 'HR', 'IT', 'Sales'],
        'salary': [50000, 60000, 70000, 55000, 65000, 85000, 52000, 63000, 75000, 68000],
        'is_active': [True, True, False, True, True, True, False, True, True, True]
    })
    
    # Quick profile
    quick_summary = profiler.quick_profile(simple_data)
    print(quick_summary)
    
    # Demo 2: Full profile analysis
    print("\n📈 Demo 2: Full Profile Analysis")
    print("-" * 30)
    full_profile = profiler.profile_data(simple_data, title="Employee Data", minimal=False)
    
    print(f"Dataset Title: {full_profile['title']}")
    print(f"Shape: {full_profile['summary']['shape']}")
    print(f"Memory Usage: {full_profile['summary']['memory_usage_mb']} MB")
    print(f"Missing Data: {full_profile['summary']['missing_cells_percent']}%")
    print(f"Quality Score: {full_profile['quality_report']['quality_score']}/100")
    
    # Show column analysis
    print(f"\nColumn Analysis:")
    for col, analysis in full_profile['columns'].items():
        print(f"  {col}: {analysis['type']} ({analysis['unique_count']} unique values)")
        if analysis['quality_issues']:
            print(f"    Issues: {', '.join(analysis['quality_issues'])}")
    
    # Show correlations if any
    if 'high_correlations' in full_profile['correlations'] and full_profile['correlations']['high_correlations']:
        print(f"\nHigh Correlations Found:")
        for corr in full_profile['correlations']['high_correlations']:
            print(f"  {corr['column1']} ↔ {corr['column2']}: {corr['correlation']:.3f}")
    else:
        print(f"\nNo high correlations detected")
    
    # Demo 3: Real sample data (if available)
    print(f"\n📁 Demo 3: Real Sample Data")
    print("-" * 30)
    if os.path.exists("sample_data/clean_dataset.csv"):
        sample_profile = profiler.profile_data("sample_data/clean_dataset.csv", minimal=True)
        print(f"Sample dataset profiled:")
        print(f"  Shape: {sample_profile['summary']['shape']}")
        print(f"  Data types: {sample_profile['summary']['data_types']}")
        print(f"  Quality score: {sample_profile['quality_report']['quality_score']}/100")
        print(f"  Issues found: {sample_profile['quality_report']['issues_found']}")
    else:
        print("Sample data not available")
    
    # Demo 4: Export functionality
    print(f"\n💾 Demo 4: Export Functionality")
    print("-" * 30)
    export_path = profiler.export_profile(full_profile, "demo_profile_export.json")
    print(f"Profile exported to: {export_path}")
    print(f"Export file size: {os.path.getsize(export_path)} bytes")
    
    # Demo 5: LLM-friendly output
    print(f"\n🤖 Demo 5: LLM-Friendly Output Structure")
    print("-" * 30)
    print("The profile output is structured for easy LLM consumption:")
    print(f"  - Summary section with key metrics")
    print(f"  - Column-by-column analysis with type detection")
    print(f"  - Quality assessment with specific issues")
    print(f"  - Correlation analysis for numerical data")
    print(f"  - Metadata with profiling parameters")
    
    print(f"\n🎉 MVP Demo Complete!")
    print("The custom pandas-based profiler is:")
    print("  ✓ Lightweight (only pandas/numpy dependencies)")
    print("  ✓ Fast and reliable")
    print("  ✓ LLM-optimized output format")
    print("  ✓ Comprehensive data analysis")
    print("  ✓ Export ready")
    
    return full_profile

if __name__ == "__main__":
    try:
        result = demonstrate_mvp()
        print("\n✅ MVP Demo: SUCCESS")
    except Exception as e:
        print(f"\n❌ MVP Demo failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)