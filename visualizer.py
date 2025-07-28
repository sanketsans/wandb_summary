"""
Visualization Module

This module provides comprehensive visualization capabilities for wandb run data,
including performance comparisons, trend analysis, and statistical plots.
"""

import os
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Optional, Any, Tuple
import logging
from datetime import datetime
from pathlib import Path
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import warnings

# Configure logging and warnings
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
warnings.filterwarnings('ignore')

# Set style for matplotlib
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")


class WandbVisualizer:
    """Comprehensive visualizer for wandb run data."""
    
    def __init__(self, output_dir: str = "wandb_summary/visualizations"):
        """
        Initialize the visualizer.
        
        Args:
            output_dir: Directory to save visualizations
        """
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        # Create subdirectories
        self.plots_dir = os.path.join(output_dir, "plots")
        self.interactive_dir = os.path.join(output_dir, "interactive")
        os.makedirs(self.plots_dir, exist_ok=True)
        os.makedirs(self.interactive_dir, exist_ok=True)
    
    def create_comprehensive_dashboard(self, 
                                     summary_df: pd.DataFrame,
                                     runs_data: List[Dict] = None) -> Dict[str, str]:
        """
        Create a comprehensive dashboard with multiple visualizations.
        
        Args:
            summary_df: DataFrame with run summaries
            runs_data: Optional list of detailed run data
            
        Returns:
            Dictionary with file paths to generated visualizations
        """
        file_paths = {}
        
        # 1. Overview dashboard
        file_paths['overview'] = self.create_overview_dashboard(summary_df)
        
        # 2. Performance comparison
        file_paths['performance'] = self.create_performance_comparison(summary_df)
        
        # 3. Configuration analysis
        file_paths['config_analysis'] = self.create_configuration_analysis(summary_df)
        
        # 4. Timeline analysis
        file_paths['timeline'] = self.create_timeline_analysis(summary_df)
        
        # 5. Statistical analysis
        file_paths['statistics'] = self.create_statistical_analysis(summary_df)
        
        # 6. Interactive dashboard
        if runs_data:
            file_paths['interactive'] = self.create_interactive_dashboard(summary_df, runs_data)
        
        return file_paths
    
    def create_overview_dashboard(self, summary_df: pd.DataFrame) -> str:
        """Create an overview dashboard with key metrics."""
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Wandb Runs Overview Dashboard', fontsize=16, fontweight='bold')
        
        # 1. Run states distribution
        state_counts = summary_df['state'].value_counts()
        axes[0, 0].pie(state_counts.values, labels=state_counts.index, autopct='%1.1f%%')
        axes[0, 0].set_title('Run States Distribution')
        
        # 2. Runs over time
        summary_df['created_at'] = pd.to_datetime(summary_df['created_at'])
        daily_counts = summary_df.groupby(summary_df['created_at'].dt.date).size()
        axes[0, 1].plot(daily_counts.index, daily_counts.values, marker='o')
        axes[0, 1].set_title('Runs Over Time')
        axes[0, 1].set_xlabel('Date')
        axes[0, 1].set_ylabel('Number of Runs')
        axes[0, 1].tick_params(axis='x', rotation=45)
        
        # 3. Top tags
        all_tags = []
        for tags_str in summary_df['tags'].dropna():
            if tags_str:
                all_tags.extend(tags_str.split(','))
        tag_counts = pd.Series(all_tags).value_counts().head(10)
        axes[0, 2].barh(range(len(tag_counts)), tag_counts.values)
        axes[0, 2].set_yticks(range(len(tag_counts)))
        axes[0, 2].set_yticklabels(tag_counts.index)
        axes[0, 2].set_title('Top 10 Tags')
        axes[0, 2].set_xlabel('Count')
        
        # 4. Summary metrics distribution (if available)
        summary_cols = [col for col in summary_df.columns if col.startswith('summary_')]
        if summary_cols:
            # Select first numeric summary metric
            for col in summary_cols:
                if summary_df[col].dtype in ['int64', 'float64']:
                    axes[1, 0].hist(summary_df[col].dropna(), bins=20, alpha=0.7)
                    axes[1, 0].set_title(f'Distribution of {col.replace("summary_", "")}')
                    axes[1, 0].set_xlabel(col.replace('summary_', ''))
                    axes[1, 0].set_ylabel('Frequency')
                    break
        
        # 5. Run duration (if available)
        if 'summary_runtime' in summary_df.columns:
            axes[1, 1].hist(summary_df['summary_runtime'].dropna(), bins=20, alpha=0.7)
            axes[1, 1].set_title('Runtime Distribution')
            axes[1, 1].set_xlabel('Runtime (seconds)')
            axes[1, 1].set_ylabel('Frequency')
        
        # 6. Group distribution
        group_counts = summary_df['group'].value_counts().head(10)
        axes[1, 2].bar(range(len(group_counts)), group_counts.values)
        axes[1, 2].set_title('Top 10 Groups')
        axes[1, 2].set_xlabel('Group')
        axes[1, 2].set_ylabel('Count')
        axes[1, 2].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        
        # Save plot
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"overview_dashboard_{timestamp}.png"
        filepath = os.path.join(self.plots_dir, filename)
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
        
        return filepath
    
    def create_performance_comparison(self, summary_df: pd.DataFrame) -> str:
        """Create performance comparison visualizations."""
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Performance Comparison Analysis', fontsize=16, fontweight='bold')
        
        # Get summary metrics
        summary_cols = [col for col in summary_df.columns if col.startswith('summary_')]
        numeric_summary_cols = [col for col in summary_cols 
                              if summary_df[col].dtype in ['int64', 'float64']]
        
        if len(numeric_summary_cols) >= 2:
            # 1. Scatter plot of two main metrics
            metric1, metric2 = numeric_summary_cols[:2]
            axes[0, 0].scatter(summary_df[metric1], summary_df[metric2], alpha=0.6)
            axes[0, 0].set_xlabel(metric1.replace('summary_', ''))
            axes[0, 0].set_ylabel(metric2.replace('summary_', ''))
            axes[0, 0].set_title(f'{metric1.replace("summary_", "")} vs {metric2.replace("summary_", "")}')
            
            # Add trend line
            z = np.polyfit(summary_df[metric1].dropna(), summary_df[metric2].dropna(), 1)
            p = np.poly1d(z)
            axes[0, 0].plot(summary_df[metric1], p(summary_df[metric1]), "r--", alpha=0.8)
        
        # 2. Box plot of top metrics
        if numeric_summary_cols:
            top_metrics = numeric_summary_cols[:4]  # Top 4 metrics
            plot_data = []
            labels = []
            for col in top_metrics:
                plot_data.append(summary_df[col].dropna())
                labels.append(col.replace('summary_', ''))
            
            axes[0, 1].boxplot(plot_data, labels=labels)
            axes[0, 1].set_title('Distribution of Key Metrics')
            axes[0, 1].tick_params(axis='x', rotation=45)
        
        # 3. Performance by group
        if 'group' in summary_df.columns and numeric_summary_cols:
            group_metric = summary_df.groupby('group')[numeric_summary_cols[0]].mean().sort_values(ascending=False)
            top_groups = group_metric.head(10)
            axes[1, 0].bar(range(len(top_groups)), top_groups.values)
            axes[1, 0].set_title(f'Average {numeric_summary_cols[0].replace("summary_", "")} by Group')
            axes[1, 0].set_xlabel('Group')
            axes[1, 0].set_ylabel(numeric_summary_cols[0].replace('summary_', ''))
            axes[1, 0].tick_params(axis='x', rotation=45)
        
        # 4. Performance over time
        if 'created_at' in summary_df.columns and numeric_summary_cols:
            summary_df['created_at'] = pd.to_datetime(summary_df['created_at'])
            time_series = summary_df.groupby(summary_df['created_at'].dt.date)[numeric_summary_cols[0]].mean()
            axes[1, 1].plot(time_series.index, time_series.values, marker='o')
            axes[1, 1].set_title(f'{numeric_summary_cols[0].replace("summary_", "")} Over Time')
            axes[1, 1].set_xlabel('Date')
            axes[1, 1].set_ylabel(numeric_summary_cols[0].replace('summary_', ''))
            axes[1, 1].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        
        # Save plot
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"performance_comparison_{timestamp}.png"
        filepath = os.path.join(self.plots_dir, filename)
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
        
        return filepath
    
    def create_configuration_analysis(self, summary_df: pd.DataFrame) -> str:
        """Create configuration analysis visualizations."""
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Configuration Analysis', fontsize=16, fontweight='bold')
        
        # Get configuration columns
        config_cols = [col for col in summary_df.columns if col.startswith('config_')]
        
        if config_cols:
            # 1. Configuration parameter distributions
            numeric_config_cols = [col for col in config_cols 
                                 if summary_df[col].dtype in ['int64', 'float64']]
            
            if numeric_config_cols:
                # Plot distributions of numeric configs
                for i, col in enumerate(numeric_config_cols[:4]):
                    row, col_idx = i // 2, i % 2
                    axes[row, col_idx].hist(summary_df[col].dropna(), bins=20, alpha=0.7)
                    axes[row, col_idx].set_title(f'Distribution of {col.replace("config_", "")}')
                    axes[row, col_idx].set_xlabel(col.replace('config_', ''))
                    axes[row, col_idx].set_ylabel('Frequency')
            
            # 2. Categorical config analysis
            categorical_config_cols = [col for col in config_cols 
                                     if summary_df[col].dtype == 'object']
            
            if categorical_config_cols:
                # Plot top categorical values
                for i, col in enumerate(categorical_config_cols[:4]):
                    if i >= 4:  # Only plot first 4
                        break
                    value_counts = summary_df[col].value_counts().head(10)
                    row, col_idx = (i + 2) // 2, (i + 2) % 2
                    if row < 2:  # Ensure we don't exceed subplot bounds
                        axes[row, col_idx].bar(range(len(value_counts)), value_counts.values)
                        axes[row, col_idx].set_title(f'Top {col.replace("config_", "")} Values')
                        axes[row, col_idx].set_xlabel(col.replace('config_', ''))
                        axes[row, col_idx].set_ylabel('Count')
                        axes[row, col_idx].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        
        # Save plot
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"config_analysis_{timestamp}.png"
        filepath = os.path.join(self.plots_dir, filename)
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
        
        return filepath
    
    def create_timeline_analysis(self, summary_df: pd.DataFrame) -> str:
        """Create timeline analysis visualizations."""
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Timeline Analysis', fontsize=16, fontweight='bold')
        
        # Convert to datetime
        summary_df['created_at'] = pd.to_datetime(summary_df['created_at'])
        
        # 1. Runs over time
        daily_counts = summary_df.groupby(summary_df['created_at'].dt.date).size()
        axes[0, 0].plot(daily_counts.index, daily_counts.values, marker='o', linewidth=2)
        axes[0, 0].set_title('Runs Over Time')
        axes[0, 0].set_xlabel('Date')
        axes[0, 0].set_ylabel('Number of Runs')
        axes[0, 0].tick_params(axis='x', rotation=45)
        
        # 2. Cumulative runs
        cumulative_runs = daily_counts.cumsum()
        axes[0, 1].plot(cumulative_runs.index, cumulative_runs.values, marker='o', linewidth=2)
        axes[0, 1].set_title('Cumulative Runs')
        axes[0, 1].set_xlabel('Date')
        axes[0, 1].set_ylabel('Total Runs')
        axes[0, 1].tick_params(axis='x', rotation=45)
        
        # 3. Performance over time
        summary_cols = [col for col in summary_df.columns if col.startswith('summary_')]
        numeric_summary_cols = [col for col in summary_cols 
                              if summary_df[col].dtype in ['int64', 'float64']]
        
        if numeric_summary_cols:
            metric = numeric_summary_cols[0]
            time_series = summary_df.groupby(summary_df['created_at'].dt.date)[metric].mean()
            axes[1, 0].plot(time_series.index, time_series.values, marker='o', linewidth=2)
            axes[1, 0].set_title(f'{metric.replace("summary_", "")} Over Time')
            axes[1, 0].set_xlabel('Date')
            axes[1, 0].set_ylabel(metric.replace('summary_', ''))
            axes[1, 0].tick_params(axis='x', rotation=45)
        
        # 4. State changes over time
        state_time_series = summary_df.groupby([summary_df['created_at'].dt.date, 'state']).size().unstack(fill_value=0)
        state_time_series.plot(kind='bar', stacked=True, ax=axes[1, 1])
        axes[1, 1].set_title('Run States Over Time')
        axes[1, 1].set_xlabel('Date')
        axes[1, 1].set_ylabel('Number of Runs')
        axes[1, 1].tick_params(axis='x', rotation=45)
        axes[1, 1].legend(title='State')
        
        plt.tight_layout()
        
        # Save plot
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"timeline_analysis_{timestamp}.png"
        filepath = os.path.join(self.plots_dir, filename)
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
        
        return filepath
    
    def create_statistical_analysis(self, summary_df: pd.DataFrame) -> str:
        """Create statistical analysis visualizations."""
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Statistical Analysis', fontsize=16, fontweight='bold')
        
        # Get numeric summary columns
        summary_cols = [col for col in summary_df.columns if col.startswith('summary_')]
        numeric_summary_cols = [col for col in summary_cols 
                              if summary_df[col].dtype in ['int64', 'float64']]
        
        if numeric_summary_cols:
            # 1. Correlation matrix
            if len(numeric_summary_cols) > 1:
                corr_matrix = summary_df[numeric_summary_cols].corr()
                im = axes[0, 0].imshow(corr_matrix, cmap='coolwarm', aspect='auto')
                axes[0, 0].set_title('Correlation Matrix')
                axes[0, 0].set_xticks(range(len(numeric_summary_cols)))
                axes[0, 0].set_yticks(range(len(numeric_summary_cols)))
                axes[0, 0].set_xticklabels([col.replace('summary_', '') for col in numeric_summary_cols], rotation=45)
                axes[0, 0].set_yticklabels([col.replace('summary_', '') for col in numeric_summary_cols])
                plt.colorbar(im, ax=axes[0, 0])
            
            # 2. Q-Q plots for normality
            metric = numeric_summary_cols[0]
            from scipy import stats
            stats.probplot(summary_df[metric].dropna(), dist="norm", plot=axes[0, 1])
            axes[0, 1].set_title(f'Q-Q Plot for {metric.replace("summary_", "")}')
            
            # 3. Distribution with normal curve
            axes[1, 0].hist(summary_df[metric].dropna(), bins=30, density=True, alpha=0.7, label='Data')
            x = np.linspace(summary_df[metric].min(), summary_df[metric].max(), 100)
            mu, sigma = summary_df[metric].mean(), summary_df[metric].std()
            normal_curve = stats.norm.pdf(x, mu, sigma)
            axes[1, 0].plot(x, normal_curve, 'r-', linewidth=2, label='Normal Distribution')
            axes[1, 0].set_title(f'Distribution of {metric.replace("summary_", "")}')
            axes[1, 0].set_xlabel(metric.replace('summary_', ''))
            axes[1, 0].set_ylabel('Density')
            axes[1, 0].legend()
            
            # 4. Box plots for all metrics
            plot_data = []
            labels = []
            for col in numeric_summary_cols:
                plot_data.append(summary_df[col].dropna())
                labels.append(col.replace('summary_', ''))
            
            axes[1, 1].boxplot(plot_data, labels=labels)
            axes[1, 1].set_title('Distribution of All Metrics')
            axes[1, 1].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        
        # Save plot
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"statistical_analysis_{timestamp}.png"
        filepath = os.path.join(self.plots_dir, filename)
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
        
        return filepath
    
    def create_interactive_dashboard(self, 
                                   summary_df: pd.DataFrame, 
                                   runs_data: List[Dict]) -> str:
        """Create an interactive Plotly dashboard."""
        # Create subplots
        fig = make_subplots(
            rows=3, cols=2,
            subplot_titles=('Runs Over Time', 'Performance Comparison', 
                          'Configuration Analysis', 'Run States', 
                          'Metrics Distribution', 'Correlation Matrix'),
            specs=[[{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"secondary_y": False}]]
        )
        
        # 1. Runs over time
        summary_df['created_at'] = pd.to_datetime(summary_df['created_at'])
        daily_counts = summary_df.groupby(summary_df['created_at'].dt.date).size()
        fig.add_trace(
            go.Scatter(x=list(daily_counts.index), y=daily_counts.values, 
                      mode='lines+markers', name='Daily Runs'),
            row=1, col=1
        )
        
        # 2. Performance comparison
        summary_cols = [col for col in summary_df.columns if col.startswith('summary_')]
        numeric_summary_cols = [col for col in summary_cols 
                              if summary_df[col].dtype in ['int64', 'float64']]
        
        if len(numeric_summary_cols) >= 2:
            fig.add_trace(
                go.Scatter(x=summary_df[numeric_summary_cols[0]], 
                          y=summary_df[numeric_summary_cols[1]],
                          mode='markers', name='Performance Comparison',
                          text=summary_df['name']),
                row=1, col=2
            )
        
        # 3. Configuration analysis
        config_cols = [col for col in summary_df.columns if col.startswith('config_')]
        if config_cols:
            categorical_config_cols = [col for col in config_cols 
                                     if summary_df[col].dtype == 'object']
            if categorical_config_cols:
                value_counts = summary_df[categorical_config_cols[0]].value_counts().head(10)
                fig.add_trace(
                    go.Bar(x=list(value_counts.index), y=value_counts.values,
                          name='Config Values'),
                    row=2, col=1
                )
        
        # 4. Run states
        state_counts = summary_df['state'].value_counts()
        fig.add_trace(
            go.Pie(labels=state_counts.index, values=state_counts.values,
                  name='Run States'),
            row=2, col=2
        )
        
        # 5. Metrics distribution
        if numeric_summary_cols:
            fig.add_trace(
                go.Histogram(x=summary_df[numeric_summary_cols[0]], 
                           name='Metrics Distribution'),
                row=3, col=1
            )
        
        # 6. Correlation matrix
        if len(numeric_summary_cols) > 1:
            corr_matrix = summary_df[numeric_summary_cols].corr()
            fig.add_trace(
                go.Heatmap(z=corr_matrix.values, 
                          x=[col.replace('summary_', '') for col in numeric_summary_cols],
                          y=[col.replace('summary_', '') for col in numeric_summary_cols],
                          colorscale='RdBu', name='Correlation'),
                row=3, col=2
            )
        
        # Update layout
        fig.update_layout(
            title_text="Interactive Wandb Runs Dashboard",
            showlegend=False,
            height=1200
        )
        
        # Save interactive plot
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"interactive_dashboard_{timestamp}.html"
        filepath = os.path.join(self.interactive_dir, filename)
        fig.write_html(filepath)
        
        return filepath 