"""
Report Generator Module

This module generates comprehensive HTML reports from wandb analysis results,
including visualizations, AI insights, and recommendations.
"""

import os
import json
import pandas as pd
from typing import Dict, List, Optional, Any
import logging
from datetime import datetime
from pathlib import Path
import base64
from jinja2 import Template

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ReportGenerator:
    """Generate comprehensive HTML reports from wandb analysis."""
    
    def __init__(self, output_dir: str = "wandb_summary/reports"):
        """
        Initialize the report generator.
        
        Args:
            output_dir: Directory to save generated reports
        """
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        # HTML template for reports
        self.html_template = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{{ title }}</title>
    <style>
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            line-height: 1.6;
            margin: 0;
            padding: 20px;
            background-color: #f5f5f5;
        }
        .container {
            max-width: 1200px;
            margin: 0 auto;
            background-color: white;
            padding: 30px;
            border-radius: 10px;
            box-shadow: 0 0 20px rgba(0,0,0,0.1);
        }
        .header {
            text-align: center;
            border-bottom: 3px solid #007bff;
            padding-bottom: 20px;
            margin-bottom: 30px;
        }
        .header h1 {
            color: #007bff;
            margin: 0;
            font-size: 2.5em;
        }
        .header p {
            color: #666;
            margin: 10px 0 0 0;
            font-size: 1.1em;
        }
        .section {
            margin-bottom: 40px;
            padding: 20px;
            border: 1px solid #e0e0e0;
            border-radius: 8px;
            background-color: #fafafa;
        }
        .section h2 {
            color: #333;
            border-bottom: 2px solid #007bff;
            padding-bottom: 10px;
            margin-top: 0;
        }
        .section h3 {
            color: #555;
            margin-top: 25px;
        }
        .metrics-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 20px;
            margin: 20px 0;
        }
        .metric-card {
            background: white;
            padding: 20px;
            border-radius: 8px;
            border-left: 4px solid #007bff;
            box-shadow: 0 2px 5px rgba(0,0,0,0.1);
        }
        .metric-card h4 {
            margin: 0 0 10px 0;
            color: #007bff;
        }
        .metric-value {
            font-size: 1.5em;
            font-weight: bold;
            color: #333;
        }
        .insights-list, .recommendations-list {
            background: white;
            padding: 20px;
            border-radius: 8px;
            margin: 15px 0;
        }
        .insights-list li, .recommendations-list li {
            margin-bottom: 10px;
            padding: 10px;
            background: #f8f9fa;
            border-radius: 5px;
            border-left: 3px solid #28a745;
        }
        .visualization {
            text-align: center;
            margin: 20px 0;
            padding: 20px;
            background: white;
            border-radius: 8px;
            box-shadow: 0 2px 5px rgba(0,0,0,0.1);
        }
        .visualization img {
            max-width: 100%;
            height: auto;
            border-radius: 5px;
        }
        .visualization h4 {
            margin-bottom: 15px;
            color: #333;
        }
        .table-container {
            overflow-x: auto;
            margin: 20px 0;
        }
        table {
            width: 100%;
            border-collapse: collapse;
            background: white;
            border-radius: 8px;
            overflow: hidden;
            box-shadow: 0 2px 5px rgba(0,0,0,0.1);
        }
        th, td {
            padding: 12px;
            text-align: left;
            border-bottom: 1px solid #ddd;
        }
        th {
            background-color: #007bff;
            color: white;
            font-weight: bold;
        }
        tr:nth-child(even) {
            background-color: #f2f2f2;
        }
        tr:hover {
            background-color: #e9ecef;
        }
        .summary-stats {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 15px;
            margin: 20px 0;
        }
        .stat-item {
            background: white;
            padding: 15px;
            border-radius: 8px;
            text-align: center;
            border: 1px solid #e0e0e0;
        }
        .stat-value {
            font-size: 2em;
            font-weight: bold;
            color: #007bff;
        }
        .stat-label {
            color: #666;
            margin-top: 5px;
        }
        .footer {
            text-align: center;
            margin-top: 40px;
            padding-top: 20px;
            border-top: 1px solid #e0e0e0;
            color: #666;
        }
        .toc {
            background: white;
            padding: 20px;
            border-radius: 8px;
            margin-bottom: 30px;
            border: 1px solid #e0e0e0;
        }
        .toc h3 {
            margin-top: 0;
            color: #007bff;
        }
        .toc ul {
            list-style-type: none;
            padding-left: 0;
        }
        .toc li {
            padding: 5px 0;
            border-bottom: 1px solid #f0f0f0;
        }
        .toc a {
            text-decoration: none;
            color: #333;
        }
        .toc a:hover {
            color: #007bff;
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>{{ title }}</h1>
            <p>Generated on {{ generation_date }}</p>
            <p>Analysis of {{ total_runs }} wandb runs</p>
        </div>

        <div class="toc">
            <h3>Table of Contents</h3>
            <ul>
                <li><a href="#executive-summary">Executive Summary</a></li>
                <li><a href="#key-metrics">Key Metrics</a></li>
                <li><a href="#ai-insights">AI Analysis & Insights</a></li>
                <li><a href="#visualizations">Visualizations</a></li>
                <li><a href="#recommendations">Recommendations</a></li>
                <li><a href="#detailed-analysis">Detailed Analysis</a></li>
            </ul>
        </div>

        <div id="executive-summary" class="section">
            <h2>Executive Summary</h2>
            <p>{{ executive_summary }}</p>
            
            <div class="summary-stats">
                <div class="stat-item">
                    <div class="stat-value">{{ total_runs }}</div>
                    <div class="stat-label">Total Runs</div>
                </div>
                <div class="stat-item">
                    <div class="stat-value">{{ completed_runs }}</div>
                    <div class="stat-label">Completed</div>
                </div>
                <div class="stat-item">
                    <div class="stat-value">{{ failed_runs }}</div>
                    <div class="stat-label">Failed</div>
                </div>
                <div class="stat-item">
                    <div class="stat-value">{{ success_rate }}%</div>
                    <div class="stat-label">Success Rate</div>
                </div>
            </div>
        </div>

        <div id="key-metrics" class="section">
            <h2>Key Metrics</h2>
            <div class="metrics-grid">
                {% for metric in key_metrics %}
                <div class="metric-card">
                    <h4>{{ metric.name }}</h4>
                    <div class="metric-value">{{ metric.value }}</div>
                    <p>{{ metric.description }}</p>
                </div>
                {% endfor %}
            </div>
        </div>

        <div id="ai-insights" class="section">
            <h2>AI Analysis & Insights</h2>
            
            <h3>Key Insights</h3>
            <div class="insights-list">
                <ul>
                    {% for insight in insights %}
                    <li>{{ insight }}</li>
                    {% endfor %}
                </ul>
            </div>

            <h3>Detailed Analysis</h3>
            <div style="background: white; padding: 20px; border-radius: 8px; margin: 15px 0;">
                {{ detailed_analysis | safe }}
            </div>
        </div>

        <div id="visualizations" class="section">
            <h2>Visualizations</h2>
            {% for viz in visualizations %}
            <div class="visualization">
                <h4>{{ viz.title }}</h4>
                <img src="{{ viz.path }}" alt="{{ viz.title }}">
                <p>{{ viz.description }}</p>
            </div>
            {% endfor %}
        </div>

        <div id="recommendations" class="section">
            <h2>Recommendations</h2>
            <div class="recommendations-list">
                <ul>
                    {% for recommendation in recommendations %}
                    <li>{{ recommendation }}</li>
                    {% endfor %}
                </ul>
            </div>
        </div>

        <div id="detailed-analysis" class="section">
            <h2>Detailed Analysis</h2>
            
            <h3>Run Summary</h3>
            <div class="table-container">
                {{ runs_table | safe }}
            </div>

            <h3>Configuration Analysis</h3>
            <div class="table-container">
                {{ config_table | safe }}
            </div>
        </div>

        <div class="footer">
            <p>Report generated by Wandb Analysis Tool</p>
            <p>Generated on {{ generation_date }}</p>
        </div>
    </div>
</body>
</html>
"""
    
    def generate_report(self, 
                       summary_df: pd.DataFrame,
                       analysis_results: List[Dict],
                       visualization_paths: Dict[str, str],
                       project_info: Dict[str, Any]) -> str:
        """
        Generate a comprehensive HTML report.
        
        Args:
            summary_df: DataFrame with run summaries
            analysis_results: List of analysis results
            visualization_paths: Dictionary of visualization file paths
            project_info: Project information dictionary
            
        Returns:
            Path to generated HTML report
        """
        try:
            # Prepare data for template
            template_data = self._prepare_template_data(
                summary_df, analysis_results, visualization_paths, project_info
            )
            
            # Create template and render
            template = Template(self.html_template)
            html_content = template.render(**template_data)
            
            # Save report
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"wandb_analysis_report_{timestamp}.html"
            filepath = os.path.join(self.output_dir, filename)
            
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(html_content)
            
            logger.info(f"Generated report: {filepath}")
            return filepath
            
        except Exception as e:
            logger.error(f"Error generating report: {e}")
            return None
    
    def _prepare_template_data(self, 
                             summary_df: pd.DataFrame,
                             analysis_results: List[Dict],
                             visualization_paths: Dict[str, str],
                             project_info: Dict[str, Any]) -> Dict[str, Any]:
        """Prepare data for the HTML template."""
        
        # Basic information
        total_runs = len(summary_df)
        completed_runs = len(summary_df[summary_df['state'] == 'finished'])
        failed_runs = len(summary_df[summary_df['state'] == 'failed'])
        success_rate = round((completed_runs / total_runs) * 100, 1) if total_runs > 0 else 0
        
        # Executive summary
        executive_summary = self._generate_executive_summary(summary_df, analysis_results)
        
        # Key metrics
        key_metrics = self._extract_key_metrics(summary_df)
        
        # AI insights and recommendations
        insights = []
        recommendations = []
        detailed_analysis = ""
        
        for result in analysis_results:
            if 'insights' in result:
                insights.extend(result['insights'])
            if 'recommendations' in result:
                recommendations.extend(result['recommendations'])
            if 'content' in result:
                detailed_analysis += f"<h4>{result.get('analysis_type', 'Analysis')}</h4>"
                detailed_analysis += f"<p>{result['content']}</p>"
        
        # Prepare visualizations
        visualizations = []
        for viz_type, path in visualization_paths.items():
            if os.path.exists(path):
                visualizations.append({
                    'title': viz_type.replace('_', ' ').title(),
                    'path': path,
                    'description': f"Visualization showing {viz_type.replace('_', ' ')}"
                })
        
        # Prepare tables
        runs_table = self._create_runs_table(summary_df)
        config_table = self._create_config_table(summary_df)
        
        return {
            'title': f"Wandb Analysis Report - {project_info.get('project', 'Unknown Project')}",
            'generation_date': datetime.now().strftime("%B %d, %Y at %H:%M"),
            'total_runs': total_runs,
            'completed_runs': completed_runs,
            'failed_runs': failed_runs,
            'success_rate': success_rate,
            'executive_summary': executive_summary,
            'key_metrics': key_metrics,
            'insights': insights,
            'recommendations': recommendations,
            'detailed_analysis': detailed_analysis,
            'visualizations': visualizations,
            'runs_table': runs_table,
            'config_table': config_table
        }
    
    def _generate_executive_summary(self, 
                                  summary_df: pd.DataFrame,
                                  analysis_results: List[Dict]) -> str:
        """Generate an executive summary."""
        
        total_runs = len(summary_df)
        completed_runs = len(summary_df[summary_df['state'] == 'finished'])
        
        # Get date range
        if 'created_at' in summary_df.columns:
            summary_df['created_at'] = pd.to_datetime(summary_df['created_at'])
            date_range = f"from {summary_df['created_at'].min().strftime('%B %d, %Y')} to {summary_df['created_at'].max().strftime('%B %d, %Y')}"
        else:
            date_range = "across the analyzed period"
        
        # Get top performing runs
        summary_cols = [col for col in summary_df.columns if col.startswith('summary_')]
        if summary_cols:
            numeric_cols = [col for col in summary_cols 
                          if summary_df[col].dtype in ['int64', 'float64']]
            if numeric_cols:
                best_metric = numeric_cols[0]
                best_run = summary_df.loc[summary_df[best_metric].idxmax()]
                best_value = best_run[best_metric]
                best_name = best_run['name']
            else:
                best_metric = "performance"
                best_value = "N/A"
                best_name = "N/A"
        else:
            best_metric = "performance"
            best_value = "N/A"
            best_name = "N/A"
        
        summary = f"""
        This report analyzes {total_runs} wandb runs {date_range}. 
        {completed_runs} runs completed successfully, representing a {round((completed_runs/total_runs)*100, 1)}% success rate. 
        The best performing run was '{best_name}' with a {best_metric.replace('summary_', '')} of {best_value}.
        """
        
        return summary.strip()
    
    def _extract_key_metrics(self, summary_df: pd.DataFrame) -> List[Dict]:
        """Extract key metrics for display."""
        metrics = []
        
        # Basic metrics
        metrics.append({
            'name': 'Total Runs',
            'value': len(summary_df),
            'description': 'Total number of runs analyzed'
        })
        
        metrics.append({
            'name': 'Success Rate',
            'value': f"{round((len(summary_df[summary_df['state'] == 'finished']) / len(summary_df)) * 100, 1)}%",
            'description': 'Percentage of runs that completed successfully'
        })
        
        # Summary metrics
        summary_cols = [col for col in summary_df.columns if col.startswith('summary_')]
        numeric_summary_cols = [col for col in summary_cols 
                              if summary_df[col].dtype in ['int64', 'float64']]
        
        for col in numeric_summary_cols[:5]:  # Top 5 metrics
            metric_name = col.replace('summary_', '').replace('_', ' ').title()
            mean_value = summary_df[col].mean()
            metrics.append({
                'name': f'Average {metric_name}',
                'value': f"{mean_value:.4f}",
                'description': f'Mean {metric_name} across all runs'
            })
        
        return metrics
    
    def _create_runs_table(self, summary_df: pd.DataFrame) -> str:
        """Create HTML table for runs summary."""
        # Select key columns for display
        display_cols = ['name', 'state', 'created_at', 'group']
        available_cols = [col for col in display_cols if col in summary_df.columns]
        
        # Add some summary metrics
        summary_cols = [col for col in summary_df.columns if col.startswith('summary_')]
        numeric_summary_cols = [col for col in summary_cols 
                              if summary_df[col].dtype in ['int64', 'float64']]
        
        display_cols.extend(numeric_summary_cols[:3])  # Add top 3 metrics
        
        # Create table
        table_df = summary_df[display_cols].head(20)  # Show top 20 runs
        
        # Convert to HTML
        html_table = table_df.to_html(
            index=False,
            classes=['table', 'table-striped', 'table-hover'],
            float_format='%.4f'
        )
        
        return html_table
    
    def _create_config_table(self, summary_df: pd.DataFrame) -> str:
        """Create HTML table for configuration analysis."""
        # Get configuration columns
        config_cols = [col for col in summary_df.columns if col.startswith('config_')]
        
        if not config_cols:
            return "<p>No configuration data available.</p>"
        
        # Create summary of config values
        config_summary = []
        for col in config_cols[:10]:  # Top 10 config parameters
            param_name = col.replace('config_', '').replace('_', ' ').title()
            unique_values = summary_df[col].nunique()
            most_common = summary_df[col].mode().iloc[0] if len(summary_df[col].mode()) > 0 else "N/A"
            
            config_summary.append({
                'Parameter': param_name,
                'Unique Values': unique_values,
                'Most Common': str(most_common)[:50]  # Truncate long values
            })
        
        config_df = pd.DataFrame(config_summary)
        
        # Convert to HTML
        html_table = config_df.to_html(
            index=False,
            classes=['table', 'table-striped', 'table-hover']
        )
        
        return html_table
    
    def generate_markdown_report(self, 
                               summary_df: pd.DataFrame,
                               analysis_results: List[Dict],
                               visualization_paths: Dict[str, str],
                               project_info: Dict[str, Any]) -> str:
        """
        Generate a markdown report (alternative to HTML).
        
        Args:
            summary_df: DataFrame with run summaries
            analysis_results: List of analysis results
            visualization_paths: Dictionary of visualization file paths
            project_info: Project information dictionary
            
        Returns:
            Path to generated markdown report
        """
        try:
            # Prepare data
            template_data = self._prepare_template_data(
                summary_df, analysis_results, visualization_paths, project_info
            )
            
            # Create markdown content
            markdown_content = self._create_markdown_content(template_data)
            
            # Save report
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"wandb_analysis_report_{timestamp}.md"
            filepath = os.path.join(self.output_dir, filename)
            
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(markdown_content)
            
            logger.info(f"Generated markdown report: {filepath}")
            return filepath
            
        except Exception as e:
            logger.error(f"Error generating markdown report: {e}")
            return None
    
    def _create_markdown_content(self, template_data: Dict[str, Any]) -> str:
        """Create markdown content from template data."""
        
        content = f"""# {template_data['title']}

**Generated on:** {template_data['generation_date']}  
**Total Runs:** {template_data['total_runs']}

## Executive Summary

{template_data['executive_summary']}

## Key Metrics

"""
        
        for metric in template_data['key_metrics']:
            content += f"- **{metric['name']}:** {metric['value']} - {metric['description']}\n"
        
        content += "\n## AI Analysis & Insights\n\n"
        
        if template_data['insights']:
            content += "### Key Insights\n\n"
            for insight in template_data['insights']:
                content += f"- {insight}\n"
            content += "\n"
        
        if template_data['recommendations']:
            content += "### Recommendations\n\n"
            for recommendation in template_data['recommendations']:
                content += f"- {recommendation}\n"
            content += "\n"
        
        content += "## Visualizations\n\n"
        for viz in template_data['visualizations']:
            content += f"### {viz['title']}\n\n"
            content += f"![{viz['title']}]({viz['path']})\n\n"
            content += f"{viz['description']}\n\n"
        
        return content 