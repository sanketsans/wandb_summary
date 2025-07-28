"""
Wandb Run Processor

This module provides functionality to fetch and process multiple wandb runs,
extracting metrics, configurations, and other relevant data for analysis.
"""

import os
import json
import pandas as pd
import wandb
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from datetime import datetime
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class RunData:
    """Data structure to hold processed run information."""
    run_id: str
    name: str
    state: str
    created_at: datetime
    config: Dict[str, Any]
    summary: Dict[str, Any]
    history: pd.DataFrame
    tags: List[str]
    notes: Optional[str]
    group: Optional[str]
    job_type: Optional[str]


class WandbProcessor:
    """Process multiple wandb runs and extract relevant data."""
    
    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize the wandb processor.
        
        Args:
            api_key: Optional wandb API key. If not provided, uses environment variable.
        """
        if api_key:
            os.environ["WANDB_API_KEY"] = api_key
        
        self.api = wandb.Api()
        
    def get_runs(self, 
                 entity: str, 
                 project: str, 
                 filters: Optional[Dict] = None,
                 limit: Optional[int] = None) -> List[wandb.apis.public.Run]:
        """
        Fetch runs from wandb.
        
        Args:
            entity: Wandb entity/username
            project: Project name
            filters: Optional filters to apply
            limit: Maximum number of runs to fetch
            
        Returns:
            List of wandb runs
        """
        try:
            runs = self.api.runs(f"{entity}/{project}", filters=filters)
            if limit:
                runs = list(runs)[:limit]
            else:
                runs = list(runs)
            
            logger.info(f"Fetched {len(runs)} runs from {entity}/{project}")
            return runs
        except Exception as e:
            logger.error(f"Error fetching runs: {e}")
            return []
    
    def process_run(self, run: wandb.apis.public.Run) -> RunData:
        """
        Process a single wandb run and extract relevant data.
        
        Args:
            run: Wandb run object
            
        Returns:
            Processed run data
        """
        try:
            # Extract basic run information
            run_id = run.id
            name = run.name
            state = run.state
            created_at = datetime.fromisoformat(run.created_at.replace('Z', '+00:00'))
            
            # Extract configuration
            config = dict(run.config) if run.config else {}
            
            # Extract summary metrics
            summary = dict(run.summary) if run.summary else {}
            
            # Extract history (all logged metrics)
            history_data = []
            for row in run.scan_history():
                history_data.append(row)
            
            history_df = pd.DataFrame(history_data) if history_data else pd.DataFrame()
            
            # Extract metadata
            tags = run.tags if run.tags else []
            notes = run.notes if run.notes else None
            group = run.group if run.group else None
            job_type = run.job_type if run.job_type else None
            
            return RunData(
                run_id=run_id,
                name=name,
                state=state,
                created_at=created_at,
                config=config,
                summary=summary,
                history=history_df,
                tags=tags,
                notes=notes,
                group=group,
                job_type=job_type
            )
            
        except Exception as e:
            logger.error(f"Error processing run {run.id}: {e}")
            return None
    
    def process_multiple_runs(self, 
                            entity: str, 
                            project: str,
                            filters: Optional[Dict] = None,
                            limit: Optional[int] = None) -> List[RunData]:
        """
        Process multiple wandb runs.
        
        Args:
            entity: Wandb entity/username
            project: Project name
            filters: Optional filters to apply
            limit: Maximum number of runs to fetch
            
        Returns:
            List of processed run data
        """
        runs = self.get_runs(entity, project, filters, limit)
        processed_runs = []
        
        for run in runs:
            processed_run = self.process_run(run)
            if processed_run:
                processed_runs.append(processed_run)
        
        logger.info(f"Successfully processed {len(processed_runs)} runs")
        return processed_runs
    
    def extract_metrics_summary(self, runs: List[RunData]) -> pd.DataFrame:
        """
        Extract a summary of metrics across all runs.
        
        Args:
            runs: List of processed run data
            
        Returns:
            DataFrame with metrics summary
        """
        summary_data = []
        
        for run in runs:
            run_summary = {
                'run_id': run.run_id,
                'name': run.name,
                'state': run.state,
                'created_at': run.created_at,
                'tags': ','.join(run.tags) if run.tags else '',
                'group': run.group or '',
                'job_type': run.job_type or '',
                'notes': run.notes or ''
            }
            
            # Add summary metrics
            for key, value in run.summary.items():
                if isinstance(value, (int, float, str, bool)):
                    run_summary[f'summary_{key}'] = value
            
            # Add config values
            for key, value in run.config.items():
                if isinstance(value, (int, float, str, bool)):
                    run_summary[f'config_{key}'] = value
            
            summary_data.append(run_summary)
        
        return pd.DataFrame(summary_data)
    
    def save_processed_data(self, 
                          runs: List[RunData], 
                          output_dir: str = "wandb_summary/data") -> Dict[str, str]:
        """
        Save processed run data to files.
        
        Args:
            runs: List of processed run data
            output_dir: Directory to save data
            
        Returns:
            Dictionary with file paths
        """
        os.makedirs(output_dir, exist_ok=True)
        
        file_paths = {}
        
        # Save summary metrics
        summary_df = self.extract_metrics_summary(runs)
        summary_path = os.path.join(output_dir, "runs_summary.csv")
        summary_df.to_csv(summary_path, index=False)
        file_paths['summary'] = summary_path
        
        # Save individual run data
        runs_dir = os.path.join(output_dir, "runs")
        os.makedirs(runs_dir, exist_ok=True)
        
        for run in runs:
            run_dir = os.path.join(runs_dir, run.run_id)
            os.makedirs(run_dir, exist_ok=True)
            
            # Save config
            config_path = os.path.join(run_dir, "config.json")
            with open(config_path, 'w') as f:
                json.dump(run.config, f, indent=2, default=str)
            
            # Save summary
            summary_path = os.path.join(run_dir, "summary.json")
            with open(summary_path, 'w') as f:
                json.dump(run.summary, f, indent=2, default=str)
            
            # Save history
            if not run.history.empty:
                history_path = os.path.join(run_dir, "history.csv")
                run.history.to_csv(history_path, index=False)
            
            # Save metadata
            metadata = {
                'run_id': run.run_id,
                'name': run.name,
                'state': run.state,
                'created_at': run.created_at.isoformat(),
                'tags': run.tags,
                'notes': run.notes,
                'group': run.group,
                'job_type': run.job_type
            }
            metadata_path = os.path.join(run_dir, "metadata.json")
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2, default=str)
        
        file_paths['runs_dir'] = runs_dir
        logger.info(f"Saved processed data to {output_dir}")
        
        return file_paths 