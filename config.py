"""
Configuration file for Wandb Analysis Summary Tool

This file contains all configurable settings for the tool, including:
- AI model settings
- Output directories
- Visualization preferences
- Analysis parameters
- Report templates
"""

import os
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field
import yaml
import json


@dataclass
class AIModelConfig:
    """Configuration for AI models and analysis."""
    
    # Ollama settings
    ollama_url: str = "http://localhost:11434"
    default_model: str = "llama3.2"
    temperature: float = 0.3
    top_p: float = 0.9
    max_tokens: int = 2000
    timeout: int = 120
    
    # Alternative models (if Ollama is not available)
    fallback_models: List[str] = field(default_factory=lambda: [
        "llama3.2",
        "llama3.1",
        "mistral",
        "codellama",
        "phi3"
    ])
    
    # Analysis types and their specific prompts
    analysis_types: Dict[str, str] = field(default_factory=lambda: {
        "comprehensive": "comprehensive_analysis",
        "performance": "performance_analysis", 
        "optimization": "optimization_analysis",
        "debugging": "debugging_analysis",
        "comparison": "run_comparison"
    })


@dataclass
class OutputConfig:
    """Configuration for output directories and file naming."""
    
    # Base output directory
    base_dir: str = "wandb_summary"
    
    # Subdirectories
    data_dir: str = "data"
    analysis_dir: str = "analysis"
    visualizations_dir: str = "visualizations"
    reports_dir: str = "reports"
    plots_dir: str = "plots"
    interactive_dir: str = "interactive"
    
    # File naming patterns
    timestamp_format: str = "%Y%m%d_%H%M%S"
    run_summary_filename: str = "runs_summary.csv"
    analysis_filename_pattern: str = "analysis_{type}_{timestamp}.json"
    report_filename_pattern: str = "wandb_analysis_report_{timestamp}.{ext}"
    
    # Create full paths
    def get_data_path(self) -> str:
        return os.path.join(self.base_dir, self.data_dir)
    
    def get_analysis_path(self) -> str:
        return os.path.join(self.base_dir, self.analysis_dir)
    
    def get_visualizations_path(self) -> str:
        return os.path.join(self.base_dir, self.visualizations_dir)
    
    def get_reports_path(self) -> str:
        return os.path.join(self.base_dir, self.reports_dir)
    
    def get_plots_path(self) -> str:
        return os.path.join(self.base_dir, self.visualizations_dir, self.plots_dir)
    
    def get_interactive_path(self) -> str:
        return os.path.join(self.base_dir, self.visualizations_dir, self.interactive_dir)


@dataclass
class VisualizationConfig:
    """Configuration for visualizations and plots."""
    
    # Plot settings
    figure_size: tuple = (16, 12)
    dpi: int = 300
    style: str = "seaborn-v0_8"
    color_palette: str = "husl"
    
    # Chart types to generate
    generate_overview: bool = True
    generate_performance: bool = True
    generate_config_analysis: bool = True
    generate_timeline: bool = True
    generate_statistics: bool = True
    generate_interactive: bool = True
    
    # Interactive dashboard settings
    interactive_height: int = 1200
    interactive_width: int = 1200
    
    # Specific visualization settings
    max_runs_for_comparison: int = 20
    max_metrics_to_plot: int = 5
    max_config_params_to_show: int = 10


@dataclass
class ReportConfig:
    """Configuration for report generation."""
    
    # Report types to generate
    generate_html: bool = True
    generate_markdown: bool = True
    generate_pdf: bool = False  # Future feature
    
    # Report content
    include_executive_summary: bool = True
    include_key_metrics: bool = True
    include_ai_insights: bool = True
    include_visualizations: bool = True
    include_recommendations: bool = True
    include_detailed_analysis: bool = True
    
    # Report styling
    html_template: str = "default"
    css_theme: str = "modern"
    include_toc: bool = True
    
    # Report metadata
    company_name: str = "Your Company"
    report_title: str = "Wandb Analysis Report"
    report_subtitle: str = "AI-Powered Experiment Analysis"
    
    # Custom branding
    logo_path: Optional[str] = None
    custom_css: Optional[str] = None


@dataclass
class AnalysisConfig:
    """Configuration for analysis parameters."""
    
    # Run processing
    max_runs_to_process: int = 1000
    default_limit: int = 50
    quick_analysis_limit: int = 10
    
    # Metrics to focus on
    priority_metrics: List[str] = field(default_factory=lambda: [
        "accuracy", "loss", "f1_score", "precision", "recall",
        "training_time", "inference_time", "model_size"
    ])
    
    # Configuration parameters to track
    important_configs: List[str] = field(default_factory=lambda: [
        "learning_rate", "batch_size", "epochs", "model_type",
        "optimizer", "scheduler", "dropout", "weight_decay"
    ])
    
    # Analysis depth
    extract_history: bool = True
    extract_configs: bool = True
    extract_summary: bool = True
    extract_metadata: bool = True
    
    # Comparison settings
    comparison_metrics: List[str] = field(default_factory=lambda: [
        "accuracy", "loss", "training_time"
    ])
    
    # Filtering defaults
    default_filters: Dict[str, Any] = field(default_factory=lambda: {
        "state": "finished"
    })


@dataclass
class LoggingConfig:
    """Configuration for logging and debugging."""
    
    # Logging level
    log_level: str = "INFO"
    log_format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    
    # File logging
    log_to_file: bool = True
    log_file: str = "wandb_analysis.log"
    
    # Console output
    verbose: bool = True
    show_progress: bool = True
    
    # Debug settings
    debug_mode: bool = False
    save_intermediate_results: bool = False


@dataclass
class WandbConfig:
    """Configuration for Wandb integration."""
    
    # API settings
    api_timeout: int = 30
    max_retries: int = 3
    retry_delay: int = 1
    
    # Data extraction
    extract_files: bool = False
    extract_artifacts: bool = False
    extract_system_metrics: bool = True
    
    # Rate limiting
    requests_per_minute: int = 60
    batch_size: int = 10


@dataclass
class Config:
    """Main configuration class that combines all settings."""
    
    # AI model configuration
    ai_model: AIModelConfig = field(default_factory=AIModelConfig)
    
    # Output configuration
    output: OutputConfig = field(default_factory=OutputConfig)
    
    # Visualization configuration
    visualization: VisualizationConfig = field(default_factory=VisualizationConfig)
    
    # Report configuration
    report: ReportConfig = field(default_factory=ReportConfig)
    
    # Analysis configuration
    analysis: AnalysisConfig = field(default_factory=AnalysisConfig)
    
    # Logging configuration
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    
    # Wandb configuration
    wandb: WandbConfig = field(default_factory=WandbConfig)
    
    # Custom settings
    custom_settings: Dict[str, Any] = field(default_factory=dict)
    
    @classmethod
    def from_file(cls, config_path: str) -> 'Config':
        """Load configuration from a YAML or JSON file."""
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
        
        with open(config_path, 'r') as f:
            if config_path.endswith('.yaml') or config_path.endswith('.yml'):
                data = yaml.safe_load(f)
            elif config_path.endswith('.json'):
                data = json.load(f)
            else:
                raise ValueError("Configuration file must be YAML or JSON")
        
        return cls.from_dict(data)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Config':
        """Create configuration from dictionary."""
        config = cls()
        
        # Update nested configurations
        for key, value in data.items():
            if hasattr(config, key):
                attr = getattr(config, key)
                if isinstance(attr, (AIModelConfig, OutputConfig, VisualizationConfig, 
                                   ReportConfig, AnalysisConfig, LoggingConfig, WandbConfig)):
                    # Update nested config
                    for sub_key, sub_value in value.items():
                        if hasattr(attr, sub_key):
                            setattr(attr, sub_key, sub_value)
                else:
                    setattr(config, key, value)
        
        return config
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return {
            'ai_model': self.ai_model.__dict__,
            'output': self.output.__dict__,
            'visualization': self.visualization.__dict__,
            'report': self.report.__dict__,
            'analysis': self.analysis.__dict__,
            'logging': self.logging.__dict__,
            'wandb': self.wandb.__dict__,
            'custom_settings': self.custom_settings
        }
    
    def save_to_file(self, config_path: str):
        """Save configuration to a file."""
        os.makedirs(os.path.dirname(config_path), exist_ok=True)
        
        with open(config_path, 'w') as f:
            if config_path.endswith('.yaml') or config_path.endswith('.yml'):
                yaml.dump(self.to_dict(), f, default_flow_style=False, indent=2)
            elif config_path.endswith('.json'):
                json.dump(self.to_dict(), f, indent=2)
            else:
                raise ValueError("Configuration file must be YAML or JSON")
    
    def update(self, **kwargs):
        """Update configuration with new values."""
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                self.custom_settings[key] = value


# Default configuration instance
default_config = Config()


def load_config(config_path: Optional[str] = None) -> Config:
    """
    Load configuration from file or return default.
    
    Args:
        config_path: Path to configuration file (YAML or JSON)
        
    Returns:
        Configuration object
    """
    if config_path and os.path.exists(config_path):
        return Config.from_file(config_path)
    else:
        return default_config


def create_default_config_file(config_path: str = "wandb_summary_config.yaml"):
    """
    Create a default configuration file.
    
    Args:
        config_path: Path where to save the configuration file
    """
    default_config.save_to_file(config_path)
    print(f"Default configuration saved to: {config_path}")


# Example usage and configuration templates
if __name__ == "__main__":
    # Create default config file
    create_default_config_file()
    
    # Example of loading and modifying config
    config = load_config()
    
    # Update AI model settings
    config.ai_model.default_model = "llama3.2"
    config.ai_model.temperature = 0.4
    
    # Update output directory
    config.output.base_dir = "my_wandb_analysis"
    
    # Save modified config
    config.save_to_file("custom_config.yaml") 