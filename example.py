#!/usr/bin/env python3
"""
Example script demonstrating the Wandb Analysis Summary Tool

This script shows how to use the tool to analyze wandb runs and generate reports.
"""

import os
import sys
from pathlib import Path

# Add the wandb_summary directory to the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from main import WandbAnalysisOrchestrator
from config import load_config, create_default_config_file


def example_quick_analysis():
    """Example: Quick analysis of recent runs."""
    print("Running quick analysis example...")
    
    # Load or create configuration
    config = load_config()
    config.output.base_dir = "wandb_summary_example"
    
    # Initialize orchestrator
    orchestrator = WandbAnalysisOrchestrator(config=config)
    
    # Run quick analysis (you'll need to replace with your actual entity/project)
    results = orchestrator.quick_analysis(
        entity="your_username",  # Replace with your wandb username
        project="your_project",  # Replace with your project name
        limit=10
    )
    
    print(f"Analysis completed!")
    print(f"Processed {results.get('processed_runs', 0)} runs")
    print(f"Output directory: {results.get('output_dir', 'N/A')}")
    
    if 'report_paths' in results:
        print("\nGenerated reports:")
        for report_type, path in results['report_paths'].items():
            print(f"  {report_type}: {path}")
    
    return results


def example_complete_analysis():
    """Example: Complete analysis with custom filters."""
    print("Running complete analysis example...")
    
    # Load or create configuration
    config = load_config()
    config.output.base_dir = "wandb_summary_complete"
    
    # Initialize orchestrator
    orchestrator = WandbAnalysisOrchestrator(config=config)
    
    # Define filters for specific analysis
    filters = {
        "state": "finished",  # Only completed runs
        "tags": {"$in": ["experiment", "baseline"]}  # Runs with specific tags
    }
    
    # Run complete analysis
    results = orchestrator.run_complete_analysis(
        entity="your_username",  # Replace with your wandb username
        project="your_project",  # Replace with your project name
        filters=filters,
        limit=50
    )
    
    print(f"Complete analysis finished!")
    print(f"Processed {results.get('processed_runs', 0)} runs")
    
    return results


def example_run_comparison():
    """Example: Compare specific runs."""
    print("Running run comparison example...")
    
    # Load or create configuration
    config = load_config()
    config.output.base_dir = "wandb_summary_comparison"
    
    # Initialize orchestrator
    orchestrator = WandbAnalysisOrchestrator(config=config)
    
    # Compare specific run IDs (replace with actual run IDs)
    run_ids = ["run1", "run2", "run3"]  # Replace with actual run IDs
    
    results = orchestrator.compare_specific_runs(
        entity="your_username",  # Replace with your wandb username
        project="your_project",  # Replace with your project name
        run_ids=run_ids
    )
    
    print(f"Run comparison completed!")
    if 'comparison_analysis' in results:
        print("Comparison insights:")
        for insight in results['comparison_analysis'].get('insights', []):
            print(f"  - {insight}")
    
    return results


def example_custom_analysis():
    """Example: Custom analysis workflow."""
    print("Running custom analysis example...")
    
    from wandb_processor import WandbProcessor
    from ai_analyzer import AIAnalyzer
    from visualizer import WandbVisualizer
    from report_generator import ReportGenerator
    
    # Step 1: Process runs
    processor = WandbProcessor()
    runs = processor.process_multiple_runs(
        entity="your_username",  # Replace with your wandb username
        project="your_project",  # Replace with your project name
        limit=20
    )
    
    if not runs:
        print("No runs found!")
        return
    
    # Step 2: Extract summary
    summary_df = processor.extract_metrics_summary(runs)
    
    # Step 3: AI Analysis
    config = load_config()
    analyzer = AIAnalyzer(
        ollama_url=config.ai_model.ollama_url,
        model=config.ai_model.default_model
    )
    
    # Generate different types of analysis
    comprehensive_result = analyzer.analyze_runs_summary(summary_df, "comprehensive")
    performance_result = analyzer.generate_recommendations(summary_df, "performance")
    
    # Step 4: Visualizations
    config = load_config()
    visualizer = WandbVisualizer(output_dir="wandb_summary_custom_visualizations")
    
    # Convert runs to dictionary format
    runs_data = []
    for run in runs:
        run_dict = {
            'run_id': run.run_id,
            'name': run.name,
            'state': run.state,
            'summary': run.summary,
            'config': run.config
        }
        runs_data.append(run_dict)
    
    # Create visualizations
    viz_paths = visualizer.create_comprehensive_dashboard(summary_df, runs_data)
    
    # Step 5: Generate report
    report_generator = ReportGenerator(output_dir="wandb_summary_custom_reports")
    
    analysis_results = []
    if comprehensive_result:
        analysis_results.append({
            'analysis_type': 'comprehensive',
            'content': comprehensive_result.content,
            'insights': comprehensive_result.insights,
            'recommendations': comprehensive_result.recommendations,
            'metrics': comprehensive_result.metrics
        })
    
    if performance_result:
        analysis_results.append({
            'analysis_type': 'performance_recommendations',
            'content': performance_result.content,
            'insights': performance_result.insights,
            'recommendations': performance_result.recommendations,
            'metrics': performance_result.metrics
        })
    
    project_info = {
        'entity': 'your_username',
        'project': 'your_project'
    }
    
    report_path = report_generator.generate_report(
        summary_df, analysis_results, viz_paths, project_info
    )
    
    print(f"Custom analysis completed!")
    print(f"Generated report: {report_path}")
    
    return {
        'runs_processed': len(runs),
        'analysis_results': len(analysis_results),
        'visualizations': len(viz_paths),
        'report_path': report_path
    }


def main():
    """Main function to run examples."""
    print("Wandb Analysis Summary Tool - Examples")
    print("=" * 50)
    
    # Check if Ollama is available
    try:
        import requests
        response = requests.get("http://localhost:11434/api/tags", timeout=5)
        if response.status_code == 200:
            print("✓ Ollama server is running")
        else:
            print("⚠ Ollama server responded with error")
    except:
        print("⚠ Ollama server not available. AI analysis will not work.")
        print("  Please start Ollama: ollama serve")
    
    # Check if wandb is configured
    try:
        import wandb
        print("✓ Wandb is available")
    except ImportError:
        print("⚠ Wandb not installed. Please install: pip install wandb")
        return
    
    print("\nAvailable examples:")
    print("1. Quick analysis (10 runs)")
    print("2. Complete analysis with filters")
    print("3. Run comparison")
    print("4. Custom analysis workflow")
    print("5. Run all examples")
    
    choice = input("\nEnter your choice (1-5): ").strip()
    
    if choice == "1":
        example_quick_analysis()
    elif choice == "2":
        example_complete_analysis()
    elif choice == "3":
        example_run_comparison()
    elif choice == "4":
        example_custom_analysis()
    elif choice == "5":
        print("\nRunning all examples...")
        example_quick_analysis()
        print("\n" + "-" * 30)
        example_complete_analysis()
        print("\n" + "-" * 30)
        example_run_comparison()
        print("\n" + "-" * 30)
        example_custom_analysis()
    else:
        print("Invalid choice. Please run the script again.")


if __name__ == "__main__":
    main() 