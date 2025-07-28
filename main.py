"""
Main Orchestrator for Wandb Analysis

This module coordinates all components to process wandb runs, generate analysis,
visualizations, and comprehensive reports.
"""

import os
import sys
import argparse
import logging
from typing import Dict, List, Optional, Any
from pathlib import Path

# Add the wandb_summary directory to the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from wandb_processor import WandbProcessor, RunData
from ai_analyzer import AIAnalyzer, AnalysisResult
from visualizer import WandbVisualizer
from report_generator import ReportGenerator
from config import Config, load_config, create_default_config_file

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class WandbAnalysisOrchestrator:
    """Main orchestrator for wandb analysis workflow."""
    
    def __init__(self, 
                 config: Optional[Config] = None,
                 config_path: Optional[str] = None):
        """
        Initialize the orchestrator.
        
        Args:
            config: Configuration object
            config_path: Path to configuration file
        """
        # Load configuration
        if config is not None:
            self.config = config
        elif config_path is not None:
            self.config = load_config(config_path)
        else:
            self.config = load_config()
        
        # Initialize components with configuration
        self.processor = WandbProcessor()
        self.analyzer = AIAnalyzer(
            ollama_url=self.config.ai_model.ollama_url,
            model=self.config.ai_model.default_model
        )
        self.visualizer = WandbVisualizer(
            output_dir=self.config.output.get_visualizations_path()
        )
        self.report_generator = ReportGenerator(
            output_dir=self.config.output.get_reports_path()
        )
        
        # Create output directories
        os.makedirs(self.config.output.base_dir, exist_ok=True)
        os.makedirs(self.config.output.get_data_path(), exist_ok=True)
        os.makedirs(self.config.output.get_analysis_path(), exist_ok=True)
        os.makedirs(self.config.output.get_plots_path(), exist_ok=True)
        os.makedirs(self.config.output.get_interactive_path(), exist_ok=True)
        os.makedirs(self.config.output.get_reports_path(), exist_ok=True)
    
    def run_complete_analysis(self,
                            entity: str,
                            project: str,
                            filters: Optional[Dict] = None,
                            limit: Optional[int] = None,
                            generate_reports: bool = True) -> Dict[str, Any]:
        """
        Run complete analysis workflow.
        
        Args:
            entity: Wandb entity/username
            project: Project name
            filters: Optional filters for runs
            limit: Maximum number of runs to process
            generate_reports: Whether to generate HTML reports
            
        Returns:
            Dictionary with all results and file paths
        """
        logger.info(f"Starting complete analysis for {entity}/{project}")
        
        results = {
            'entity': entity,
            'project': project,
            'filters': filters,
            'limit': limit,
            'output_dir': self.config.output.base_dir
        }
        
        try:
            # Step 1: Process wandb runs
            logger.info("Step 1: Processing wandb runs...")
            runs = self.processor.process_multiple_runs(
                entity=entity,
                project=project,
                filters=filters,
                limit=limit
            )
            
            if not runs:
                logger.warning("No runs found or processed successfully")
                return results
            
            results['processed_runs'] = len(runs)
            logger.info(f"Processed {len(runs)} runs successfully")
            
            # Step 2: Save processed data
            logger.info("Step 2: Saving processed data...")
            data_paths = self.processor.save_processed_data(
                runs=runs,
                output_dir=self.config.output.get_data_path()
            )
            results['data_paths'] = data_paths
            
            # Step 3: Generate AI analysis
            logger.info("Step 3: Generating AI analysis...")
            summary_df = self.processor.extract_metrics_summary(runs)
            analysis_results = self._generate_ai_analysis(summary_df, runs)
            results['analysis_results'] = analysis_results
            
            # Step 4: Create visualizations
            logger.info("Step 4: Creating visualizations...")
            visualization_paths = self._create_visualizations(summary_df, runs)
            results['visualization_paths'] = visualization_paths
            
            # Step 5: Generate reports
            if generate_reports:
                logger.info("Step 5: Generating reports...")
                report_paths = self._generate_reports(
                    summary_df, analysis_results, visualization_paths, 
                    {'entity': entity, 'project': project}
                )
                results['report_paths'] = report_paths
            
            logger.info("Complete analysis finished successfully!")
            return results
            
        except Exception as e:
            logger.error(f"Error in complete analysis: {e}")
            results['error'] = str(e)
            return results
    
    def _generate_ai_analysis(self, 
                            summary_df: pd.DataFrame, 
                            runs: List[RunData]) -> List[Dict]:
        """Generate AI analysis for the runs."""
        analysis_results = []
        
        try:
            # Comprehensive analysis
            logger.info("Generating comprehensive analysis...")
            comprehensive_result = self.analyzer.analyze_runs_summary(
                summary_df, "comprehensive"
            )
            if comprehensive_result:
                analysis_results.append({
                    'type': 'comprehensive',
                    'result': comprehensive_result,
                    'file_path': self.analyzer.save_analysis_result(comprehensive_result)
                })
            
            # Performance recommendations
            logger.info("Generating performance recommendations...")
            performance_result = self.analyzer.generate_recommendations(
                summary_df, "performance"
            )
            if performance_result:
                analysis_results.append({
                    'type': 'performance_recommendations',
                    'result': performance_result,
                    'file_path': self.analyzer.save_analysis_result(performance_result)
                })
            
            # Model optimization recommendations
            logger.info("Generating model optimization recommendations...")
            optimization_result = self.analyzer.generate_recommendations(
                summary_df, "optimization"
            )
            if optimization_result:
                analysis_results.append({
                    'type': 'optimization_recommendations',
                    'result': optimization_result,
                    'file_path': self.analyzer.save_analysis_result(optimization_result)
                })
            
        except Exception as e:
            logger.error(f"Error generating AI analysis: {e}")
        
        return analysis_results
    
    def _create_visualizations(self, 
                             summary_df: pd.DataFrame, 
                             runs: List[RunData]) -> Dict[str, str]:
        """Create visualizations for the runs."""
        visualization_paths = {}
        
        try:
            # Convert runs to dictionary format for visualizer
            runs_data = []
            for run in runs:
                run_dict = {
                    'run_id': run.run_id,
                    'name': run.name,
                    'state': run.state,
                    'created_at': run.created_at,
                    'summary': run.summary,
                    'config': run.config,
                    'tags': run.tags,
                    'notes': run.notes,
                    'group': run.group,
                    'job_type': run.job_type
                }
                runs_data.append(run_dict)
            
            # Create comprehensive dashboard
            logger.info("Creating comprehensive dashboard...")
            dashboard_paths = self.visualizer.create_comprehensive_dashboard(
                summary_df, runs_data
            )
            visualization_paths.update(dashboard_paths)
            
        except Exception as e:
            logger.error(f"Error creating visualizations: {e}")
        
        return visualization_paths
    
    def _generate_reports(self,
                         summary_df: pd.DataFrame,
                         analysis_results: List[Dict],
                         visualization_paths: Dict[str, str],
                         project_info: Dict[str, str]) -> Dict[str, str]:
        """Generate reports from analysis results."""
        report_paths = {}
        
        try:
            # Prepare analysis results for report generator
            analysis_data = []
            for analysis in analysis_results:
                if 'result' in analysis:
                    result = analysis['result']
                    analysis_data.append({
                        'analysis_type': result.analysis_type,
                        'content': result.content,
                        'insights': result.insights,
                        'recommendations': result.recommendations,
                        'metrics': result.metrics
                    })
            
            # Generate HTML report
            logger.info("Generating HTML report...")
            html_report_path = self.report_generator.generate_report(
                summary_df=summary_df,
                analysis_results=analysis_data,
                visualization_paths=visualization_paths,
                project_info=project_info
            )
            if html_report_path:
                report_paths['html'] = html_report_path
            
            # Generate markdown report
            logger.info("Generating markdown report...")
            md_report_path = self.report_generator.generate_markdown_report(
                summary_df=summary_df,
                analysis_results=analysis_data,
                visualization_paths=visualization_paths,
                project_info=project_info
            )
            if md_report_path:
                report_paths['markdown'] = md_report_path
            
        except Exception as e:
            logger.error(f"Error generating reports: {e}")
        
        return report_paths
    
    def quick_analysis(self,
                      entity: str,
                      project: str,
                      limit: int = 10) -> Dict[str, Any]:
        """
        Run a quick analysis with limited runs.
        
        Args:
            entity: Wandb entity/username
            project: Project name
            limit: Maximum number of runs to analyze
            
        Returns:
            Dictionary with quick analysis results
        """
        logger.info(f"Running quick analysis for {entity}/{project} (max {limit} runs)")
        
        return self.run_complete_analysis(
            entity=entity,
            project=project,
            limit=limit,
            generate_reports=True
        )
    
    def compare_specific_runs(self,
                            entity: str,
                            project: str,
                            run_ids: List[str]) -> Dict[str, Any]:
        """
        Compare specific runs by their IDs.
        
        Args:
            entity: Wandb entity/username
            project: Project name
            run_ids: List of run IDs to compare
            
        Returns:
            Dictionary with comparison results
        """
        logger.info(f"Comparing specific runs: {run_ids}")
        
        results = {
            'entity': entity,
            'project': project,
            'run_ids': run_ids,
            'output_dir': self.output_dir
        }
        
        try:
            # Fetch specific runs
            runs = []
            for run_id in run_ids:
                try:
                    run = self.processor.api.run(f"{entity}/{project}/{run_id}")
                    processed_run = self.processor.process_run(run)
                    if processed_run:
                        runs.append(processed_run)
                except Exception as e:
                    logger.error(f"Error processing run {run_id}: {e}")
            
            if not runs:
                logger.warning("No runs could be processed")
                return results
            
            # Generate comparison analysis
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
            
            comparison_result = self.analyzer.compare_runs(runs_data)
            if comparison_result:
                results['comparison_analysis'] = {
                    'content': comparison_result.content,
                    'insights': comparison_result.insights,
                    'recommendations': comparison_result.recommendations,
                    'metrics': comparison_result.metrics
                }
                
                # Save comparison result
                file_path = self.analyzer.save_analysis_result(comparison_result)
                results['comparison_file'] = file_path
            
        except Exception as e:
            logger.error(f"Error in run comparison: {e}")
            results['error'] = str(e)
        
        return results


def main():
    """Main CLI interface."""
    parser = argparse.ArgumentParser(description="Wandb Analysis Tool")
    parser.add_argument("--entity", help="Wandb entity/username")
    parser.add_argument("--project", help="Wandb project name")
    parser.add_argument("--config", help="Path to configuration file")
    parser.add_argument("--create-config", action="store_true", help="Create default configuration file")
    parser.add_argument("--limit", type=int, help="Maximum number of runs to process")
    parser.add_argument("--quick", action="store_true", help="Run quick analysis (max 10 runs)")
    parser.add_argument("--compare-runs", nargs="+", help="Compare specific run IDs")
    parser.add_argument("--filters", help="JSON string of wandb filters")
    
    args = parser.parse_args()
    
    # Handle create-config option
    if args.create_config:
        config_path = args.config or "wandb_summary_config.yaml"
        create_default_config_file(config_path)
        print(f"Default configuration created at: {config_path}")
        return
    
    # Check if entity and project are provided for analysis
    if not args.entity or not args.project:
        parser.error("--entity and --project are required for analysis")
    
    # Parse filters if provided
    filters = None
    if args.filters:
        try:
            import json
            filters = json.loads(args.filters)
        except json.JSONDecodeError:
            logger.error("Invalid JSON in filters argument")
            return
    
    # Initialize orchestrator with configuration
    orchestrator = WandbAnalysisOrchestrator(config_path=args.config)
    
    # Run analysis
    if args.compare_runs:
        results = orchestrator.compare_specific_runs(
            entity=args.entity,
            project=args.project,
            run_ids=args.compare_runs
        )
    elif args.quick:
        results = orchestrator.quick_analysis(
            entity=args.entity,
            project=args.project,
            limit=args.limit or 10
        )
    else:
        results = orchestrator.run_complete_analysis(
            entity=args.entity,
            project=args.project,
            filters=filters,
            limit=args.limit
        )
    
    # Print results summary
    print("\n" + "="*50)
    print("ANALYSIS COMPLETE")
    print("="*50)
    print(f"Entity: {results.get('entity', 'N/A')}")
    print(f"Project: {results.get('project', 'N/A')}")
    print(f"Processed runs: {results.get('processed_runs', 0)}")
    print(f"Output directory: {results.get('output_dir', 'N/A')}")
    
    if 'error' in results:
        print(f"Error: {results['error']}")
    else:
        if 'report_paths' in results:
            print("\nGenerated Reports:")
            for report_type, path in results['report_paths'].items():
                print(f"  {report_type}: {path}")
        
        if 'visualization_paths' in results:
            print("\nGenerated Visualizations:")
            for viz_type, path in results['visualization_paths'].items():
                print(f"  {viz_type}: {path}")
    
    print("="*50)


if __name__ == "__main__":
    main() 