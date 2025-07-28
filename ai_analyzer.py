"""
AI-powered Analysis Module

This module provides AI-driven analysis of wandb run data, including
insights generation, run comparisons, and recommendations.
"""

import os
import json
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
import logging
from datetime import datetime
import requests
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class AnalysisResult:
    """Data structure to hold analysis results."""
    analysis_type: str
    content: str
    insights: List[str]
    recommendations: List[str]
    metrics: Dict[str, Any]
    generated_at: datetime


class AIAnalyzer:
    """AI-powered analyzer for wandb run data."""
    
    def __init__(self, 
                 ollama_url: str = "http://localhost:11434",
                 model: str = "llama3.2"):
        """
        Initialize the AI analyzer.
        
        Args:
            ollama_url: URL of the Ollama server
            model: Ollama model to use for analysis
        """
        self.ollama_url = ollama_url
        self.model = model
        
    def _create_analysis_prompt(self, 
                              data_summary: str, 
                              analysis_type: str,
                              specific_questions: Optional[List[str]] = None) -> str:
        """
        Create a prompt for AI analysis.
        
        Args:
            data_summary: Summary of the data to analyze
            analysis_type: Type of analysis to perform
            specific_questions: Specific questions to answer
            
        Returns:
            Formatted prompt string
        """
        base_prompt = f"""
You are an expert data scientist and machine learning engineer analyzing experiment results from Weights & Biases (wandb).

DATA SUMMARY:
{data_summary}

ANALYSIS TYPE: {analysis_type}

Please provide a comprehensive analysis including:

1. **Key Insights**: What are the most important findings from this data?
2. **Performance Analysis**: How do the models/runs compare in terms of performance?
3. **Trends and Patterns**: What trends or patterns do you observe?
4. **Anomalies**: Are there any unusual or concerning results?
5. **Recommendations**: What would you recommend based on this analysis?
6. **Next Steps**: What experiments or investigations should be pursued next?

Please structure your response in a clear, professional format with specific metrics and actionable insights.
"""
        
        if specific_questions:
            questions_text = "\n".join([f"- {q}" for q in specific_questions])
            base_prompt += f"\n\nSPECIFIC QUESTIONS TO ADDRESS:\n{questions_text}"
        
        return base_prompt
    
    def analyze_runs_summary(self, 
                           summary_df: pd.DataFrame,
                           analysis_type: str = "comprehensive") -> AnalysisResult:
        """
        Analyze a summary of multiple runs.
        
        Args:
            summary_df: DataFrame with run summaries
            analysis_type: Type of analysis to perform
            
        Returns:
            Analysis result
        """
        try:
            # Prepare data summary
            data_summary = self._prepare_data_summary(summary_df)
            
            # Create prompt
            prompt = self._create_analysis_prompt(data_summary, analysis_type)
            
            # Get AI response from Ollama
            response = self._call_ollama(prompt)
            
            content = response
            
            # Extract insights and recommendations
            insights, recommendations = self._extract_insights_and_recommendations(content)
            
            # Calculate key metrics
            metrics = self._calculate_key_metrics(summary_df)
            
            return AnalysisResult(
                analysis_type=analysis_type,
                content=content,
                insights=insights,
                recommendations=recommendations,
                metrics=metrics,
                generated_at=datetime.now()
            )
            
        except Exception as e:
            logger.error(f"Error in AI analysis: {e}")
            return None
    
    def compare_runs(self, 
                    runs_data: List[Dict],
                    comparison_criteria: List[str] = None) -> AnalysisResult:
        """
        Compare specific runs and provide insights.
        
        Args:
            runs_data: List of run data dictionaries
            comparison_criteria: Specific criteria to compare
            
        Returns:
            Analysis result
        """
        try:
            # Prepare comparison data
            comparison_summary = self._prepare_comparison_summary(runs_data)
            
            # Create comparison prompt
            prompt = self._create_analysis_prompt(
                comparison_summary, 
                "run_comparison",
                comparison_criteria
            )
            
            # Get AI response from Ollama
            response = self._call_ollama(prompt)
            
            content = response
            
            # Extract insights and recommendations
            insights, recommendations = self._extract_insights_and_recommendations(content)
            
            # Calculate comparison metrics
            metrics = self._calculate_comparison_metrics(runs_data)
            
            return AnalysisResult(
                analysis_type="run_comparison",
                content=content,
                insights=insights,
                recommendations=recommendations,
                metrics=metrics,
                generated_at=datetime.now()
            )
            
        except Exception as e:
            logger.error(f"Error in run comparison: {e}")
            return None
    
    def generate_recommendations(self, 
                               summary_df: pd.DataFrame,
                               focus_area: str = "performance") -> AnalysisResult:
        """
        Generate specific recommendations based on the data.
        
        Args:
            summary_df: DataFrame with run summaries
            focus_area: Area to focus recommendations on
            
        Returns:
            Analysis result
        """
        try:
            # Prepare data summary
            data_summary = self._prepare_data_summary(summary_df)
            
            # Create recommendations prompt
            prompt = f"""
You are an expert ML engineer providing recommendations based on experiment results.

DATA SUMMARY:
{data_summary}

FOCUS AREA: {focus_area}

Please provide specific, actionable recommendations for:
1. Model improvements
2. Hyperparameter tuning
3. Data preprocessing
4. Architecture changes
5. Training strategies
6. Evaluation approaches

For each recommendation, explain:
- Why it's recommended
- Expected impact
- Implementation difficulty
- Priority level

Please be specific and actionable.
"""
            
            # Get AI response from Ollama
            response = self._call_ollama(prompt)
            
            content = response
            
            # Extract insights and recommendations
            insights, recommendations = self._extract_insights_and_recommendations(content)
            
            # Calculate metrics
            metrics = self._calculate_key_metrics(summary_df)
            
            return AnalysisResult(
                analysis_type=f"recommendations_{focus_area}",
                content=content,
                insights=insights,
                recommendations=recommendations,
                metrics=metrics,
                generated_at=datetime.now()
            )
            
        except Exception as e:
            logger.error(f"Error generating recommendations: {e}")
            return None
    
    def _prepare_data_summary(self, summary_df: pd.DataFrame) -> str:
        """Prepare a text summary of the data for AI analysis."""
        summary_lines = []
        
        # Basic statistics
        summary_lines.append(f"Total runs: {len(summary_df)}")
        summary_lines.append(f"Date range: {summary_df['created_at'].min()} to {summary_df['created_at'].max()}")
        summary_lines.append(f"States: {summary_df['state'].value_counts().to_dict()}")
        
        # Summary metrics columns
        summary_cols = [col for col in summary_df.columns if col.startswith('summary_')]
        if summary_cols:
            summary_lines.append("\nSummary Metrics:")
            for col in summary_cols:
                metric_name = col.replace('summary_', '')
                values = summary_df[col].dropna()
                if len(values) > 0:
                    if values.dtype in ['int64', 'float64']:
                        summary_lines.append(f"  {metric_name}: mean={values.mean():.4f}, std={values.std():.4f}, min={values.min():.4f}, max={values.max():.4f}")
                    else:
                        summary_lines.append(f"  {metric_name}: {values.value_counts().head(3).to_dict()}")
        
        # Config columns
        config_cols = [col for col in summary_df.columns if col.startswith('config_')]
        if config_cols:
            summary_lines.append("\nConfiguration Parameters:")
            for col in config_cols:
                param_name = col.replace('config_', '')
                values = summary_df[col].dropna()
                if len(values) > 0:
                    if values.dtype in ['int64', 'float64']:
                        summary_lines.append(f"  {param_name}: mean={values.mean():.4f}, std={values.std():.4f}")
                    else:
                        summary_lines.append(f"  {param_name}: {values.value_counts().head(3).to_dict()}")
        
        return "\n".join(summary_lines)
    
    def _prepare_comparison_summary(self, runs_data: List[Dict]) -> str:
        """Prepare a summary for run comparison."""
        summary_lines = []
        
        for i, run in enumerate(runs_data):
            summary_lines.append(f"\nRun {i+1}: {run.get('name', 'Unknown')}")
            summary_lines.append(f"  ID: {run.get('run_id', 'Unknown')}")
            summary_lines.append(f"  State: {run.get('state', 'Unknown')}")
            
            # Add key metrics
            summary = run.get('summary', {})
            for key, value in summary.items():
                if isinstance(value, (int, float)):
                    summary_lines.append(f"  {key}: {value:.4f}")
                else:
                    summary_lines.append(f"  {key}: {value}")
        
        return "\n".join(summary_lines)
    
    def _extract_insights_and_recommendations(self, content: str) -> Tuple[List[str], List[str]]:
        """Extract insights and recommendations from AI response."""
        insights = []
        recommendations = []
        
        # Simple extraction based on common patterns
        lines = content.split('\n')
        in_insights = False
        in_recommendations = False
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
                
            if any(keyword in line.lower() for keyword in ['insight', 'finding', 'observation']):
                in_insights = True
                in_recommendations = False
            elif any(keyword in line.lower() for keyword in ['recommend', 'suggestion', 'next step']):
                in_insights = False
                in_recommendations = True
            elif line.startswith('#'):
                in_insights = False
                in_recommendations = False
            
            if in_insights and line and not line.startswith('#'):
                insights.append(line)
            elif in_recommendations and line and not line.startswith('#'):
                recommendations.append(line)
        
        return insights, recommendations
    
    def _calculate_key_metrics(self, summary_df: pd.DataFrame) -> Dict[str, Any]:
        """Calculate key metrics from the summary data."""
        metrics = {}
        
        # Basic statistics
        metrics['total_runs'] = len(summary_df)
        metrics['completed_runs'] = len(summary_df[summary_df['state'] == 'finished'])
        metrics['failed_runs'] = len(summary_df[summary_df['state'] == 'failed'])
        
        # Summary metrics statistics
        summary_cols = [col for col in summary_df.columns if col.startswith('summary_')]
        for col in summary_cols:
            metric_name = col.replace('summary_', '')
            values = summary_df[col].dropna()
            if len(values) > 0 and values.dtype in ['int64', 'float64']:
                metrics[f'{metric_name}_mean'] = float(values.mean())
                metrics[f'{metric_name}_std'] = float(values.std())
                metrics[f'{metric_name}_min'] = float(values.min())
                metrics[f'{metric_name}_max'] = float(values.max())
        
        return metrics
    
    def _calculate_comparison_metrics(self, runs_data: List[Dict]) -> Dict[str, Any]:
        """Calculate comparison metrics between runs."""
        metrics = {}
        
        # Extract common metrics for comparison
        common_metrics = set()
        for run in runs_data:
            summary = run.get('summary', {})
            for key in summary.keys():
                if isinstance(summary[key], (int, float)):
                    common_metrics.add(key)
        
        # Calculate comparison statistics
        for metric in common_metrics:
            values = []
            for run in runs_data:
                value = run.get('summary', {}).get(metric)
                if value is not None and isinstance(value, (int, float)):
                    values.append(value)
            
            if values:
                metrics[f'{metric}_values'] = values
                metrics[f'{metric}_mean'] = float(np.mean(values))
                metrics[f'{metric}_std'] = float(np.std(values))
                metrics[f'{metric}_range'] = float(max(values) - min(values))
        
        return metrics
    
    def save_analysis_result(self, 
                           result: AnalysisResult, 
                           output_dir: str = "wandb_summary/analysis") -> str:
        """
        Save analysis result to file.
        
        Args:
            result: Analysis result to save
            output_dir: Directory to save the result
            
        Returns:
            Path to saved file
        """
        os.makedirs(output_dir, exist_ok=True)
        
        # Create filename
        timestamp = result.generated_at.strftime("%Y%m%d_%H%M%S")
        filename = f"analysis_{result.analysis_type}_{timestamp}.json"
        filepath = os.path.join(output_dir, filename)
        
        # Prepare data for saving
        data = {
            'analysis_type': result.analysis_type,
            'content': result.content,
            'insights': result.insights,
            'recommendations': result.recommendations,
            'metrics': result.metrics,
            'generated_at': result.generated_at.isoformat()
        }
        
        # Save to file
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2, default=str)
        
        logger.info(f"Saved analysis result to {filepath}")
        return filepath
    
    def _call_ollama(self, prompt: str) -> str:
        """
        Call Ollama API to get AI response.
        
        Args:
            prompt: The prompt to send to the model
            
        Returns:
            AI response text
        """
        try:
            url = f"{self.ollama_url}/api/generate"
            
            payload = {
                "model": self.model,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": 0.3,
                    "top_p": 0.9,
                    "num_predict": 2000
                }
            }
            
            response = requests.post(url, json=payload, timeout=120)
            response.raise_for_status()
            
            result = response.json()
            return result.get('response', 'No response generated')
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Error calling Ollama API: {e}")
            return f"Error: Could not connect to Ollama server at {self.ollama_url}. Please ensure Ollama is running and the model '{self.model}' is available."
        except Exception as e:
            logger.error(f"Unexpected error in Ollama API call: {e}")
            return f"Error: {str(e)}" 