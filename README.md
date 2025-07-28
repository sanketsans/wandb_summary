# Wandb Analysis Summary Tool

A comprehensive tool for analyzing multiple Weights & Biases (wandb) runs, generating AI-powered insights, creating visualizations, and producing detailed reports.

## Features

- **Multi-run Processing**: Fetch and process multiple wandb runs from any project
- **AI-Powered Analysis**: Generate insights and recommendations using Ollama
- **Comprehensive Visualizations**: Create static and interactive charts
- **Report Generation**: Produce HTML and Markdown reports
- **Flexible Filtering**: Filter runs by various criteria
- **Run Comparison**: Compare specific runs in detail

## Installation

1. Install the required dependencies:
```bash
pip install -r requirements.txt
```

2. Install and start Ollama (for AI analysis):
```bash
# Install Ollama from https://ollama.ai
ollama pull llama3.2  # or any other model you prefer
ollama serve
```

3. Create and configure your settings:
```bash
# Create default configuration
python wandb_summary/main.py --create-config

# Or use the configuration manager
python wandb_summary/config_manager.py --create
```

## Quick Start

### Basic Usage

```python
from wandb_summary.main import WandbAnalysisOrchestrator

# Initialize the orchestrator (uses default config)
orchestrator = WandbAnalysisOrchestrator()

# Or specify a custom config file
orchestrator = WandbAnalysisOrchestrator(config_path="my_config.yaml")

# Run complete analysis
results = orchestrator.run_complete_analysis(
    entity="your_username",
    project="your_project",
    limit=50  # Optional: limit number of runs
)
```

### Command Line Interface

```bash
# Create default configuration
python wandb_summary/main.py --create-config

# Quick analysis (max 10 runs)
python wandb_summary/main.py --entity your_username --project your_project --quick

# Full analysis with custom limit
python wandb_summary/main.py --entity your_username --project your_project --limit 100

# Use custom configuration
python wandb_summary/main.py --entity your_username --project your_project --config my_config.yaml

# Compare specific runs
python wandb_summary/main.py --entity your_username --project your_project --compare-runs run1 run2 run3
```

## Components

### 1. WandbProcessor (`wandb_processor.py`)

Fetches and processes wandb runs, extracting:
- Run metadata (ID, name, state, timestamps)
- Configuration parameters
- Summary metrics
- Training history
- Tags and notes

```python
from wandb_summary.wandb_processor import WandbProcessor

processor = WandbProcessor()
runs = processor.process_multiple_runs(
    entity="username",
    project="project_name",
    limit=50
)
```

### 2. AIAnalyzer (`ai_analyzer.py`)

Uses Ollama to generate AI-powered insights:
- Comprehensive run analysis
- Performance comparisons
- Recommendations for improvements
- Trend identification

```python
from wandb_summary.ai_analyzer import AIAnalyzer

analyzer = AIAnalyzer(
    ollama_url="http://localhost:11434",
    model="llama3.2"
)

result = analyzer.analyze_runs_summary(summary_df, "comprehensive")
```

### 3. WandbVisualizer (`visualizer.py`)

Creates comprehensive visualizations:
- Overview dashboards
- Performance comparisons
- Configuration analysis
- Timeline analysis
- Statistical plots
- Interactive charts

```python
from wandb_summary.visualizer import WandbVisualizer

visualizer = WandbVisualizer(output_dir="visualizations")
paths = visualizer.create_comprehensive_dashboard(summary_df, runs_data)
```

### 4. ReportGenerator (`report_generator.py`)

Generates professional reports:
- HTML reports with interactive elements
- Markdown reports for documentation
- Executive summaries
- Detailed analysis sections

```python
from wandb_summary.report_generator import ReportGenerator

generator = ReportGenerator(output_dir="reports")
report_path = generator.generate_report(
    summary_df, analysis_results, visualization_paths, project_info
)
```

## Configuration

The tool uses a comprehensive configuration system that allows you to customize all aspects of the analysis. You can manage your configuration through:

### 1. Configuration Manager (Recommended)

```bash
# Interactive configuration editor
python wandb_summary/config_manager.py

# Show current configuration
python wandb_summary/config_manager.py --show

# Edit configuration interactively
python wandb_summary/config_manager.py --edit

# Validate configuration
python wandb_summary/config_manager.py --validate

# Export configuration
python wandb_summary/config_manager.py --export yaml
```

### 2. Configuration File

The configuration is stored in `wandb_summary_config.yaml` and includes:

#### AI Model Settings
```yaml
ai_model:
  ollama_url: "http://localhost:11434"
  default_model: "llama3.2"
  temperature: 0.3
  max_tokens: 2000
  fallback_models:
    - "llama3.2"
    - "mistral"
    - "codellama"
```

#### Output Settings
```yaml
output:
  base_dir: "wandb_summary"
  data_dir: "data"
  analysis_dir: "analysis"
  visualizations_dir: "visualizations"
  reports_dir: "reports"
```

#### Visualization Settings
```yaml
visualization:
  generate_overview: true
  generate_performance: true
  generate_interactive: true
  figure_size: [16, 12]
  dpi: 300
```

#### Report Settings
```yaml
report:
  generate_html: true
  generate_markdown: true
  company_name: "Your Company"
  report_title: "Wandb Analysis Report"
```

### 3. Ollama Setup

1. Install Ollama from [https://ollama.ai](https://ollama.ai)
2. Start the Ollama server: `ollama serve`
3. Pull a model: `ollama pull llama3.2`
4. Configure the model in your configuration file:

```yaml
ai_model:
  ollama_url: "http://localhost:11434"
  default_model: "llama3.2"
```

### Wandb Authentication

Ensure you're logged into wandb:
```bash
wandb login
```

Or set your API key:
```bash
export WANDB_API_KEY=your_api_key_here
```

## Advanced Usage

### Custom Filters

```python
# Filter runs by tags
filters = {"tags": {"$in": ["experiment", "baseline"]}}

# Filter by date range
filters = {
    "created_at": {
        "$gte": "2024-01-01",
        "$lte": "2024-12-31"
    }
}

results = orchestrator.run_complete_analysis(
    entity="username",
    project="project",
    filters=filters
)
```

### Custom Analysis

```python
# Generate specific recommendations
performance_result = analyzer.generate_recommendations(
    summary_df, focus_area="performance"
)

# Compare specific runs
comparison_result = analyzer.compare_runs(
    runs_data, 
    comparison_criteria=["accuracy", "loss", "training_time"]
)
```

### Custom Visualizations

```python
# Create specific visualizations
overview_path = visualizer.create_overview_dashboard(summary_df)
performance_path = visualizer.create_performance_comparison(summary_df)
timeline_path = visualizer.create_timeline_analysis(summary_df)
```

## Output Structure

```
wandb_summary/
├── data/
│   ├── runs_summary.csv
│   └── runs/
│       ├── run_id_1/
│       │   ├── config.json
│       │   ├── summary.json
│       │   ├── history.csv
│       │   └── metadata.json
│       └── ...
├── analysis/
│   ├── analysis_comprehensive_20241201_143022.json
│   ├── analysis_performance_recommendations_20241201_143023.json
│   └── ...
├── visualizations/
│   ├── plots/
│   │   ├── overview_dashboard_20241201_143024.png
│   │   ├── performance_comparison_20241201_143025.png
│   │   └── ...
│   └── interactive/
│       └── interactive_dashboard_20241201_143026.html
└── reports/
    ├── wandb_analysis_report_20241201_143027.html
    └── wandb_analysis_report_20241201_143028.md
```

## Examples

### Example 1: Quick Project Overview

```python
# Get a quick overview of your latest experiments
results = orchestrator.quick_analysis(
    entity="your_username",
    project="ml_experiments",
    limit=20
)
```

### Example 2: Compare Model Variants

```python
# Compare different model configurations
filters = {"tags": {"$in": ["model_variant"]}}
results = orchestrator.run_complete_analysis(
    entity="your_username",
    project="model_comparison",
    filters=filters
)
```

### Example 3: Debug Failed Runs

```python
# Analyze failed runs to identify patterns
filters = {"state": "failed"}
results = orchestrator.run_complete_analysis(
    entity="your_username",
    project="debug_project",
    filters=filters
)
```

## Troubleshooting

### Common Issues

1. **Ollama Connection Error**
   - Ensure Ollama is running: `ollama serve`
   - Check the URL: `http://localhost:11434`
   - Verify model is available: `ollama list`

2. **Wandb Authentication Error**
   - Run `wandb login`
   - Check API key: `echo $WANDB_API_KEY`

3. **No Runs Found**
   - Verify entity and project names
   - Check if runs exist in the project
   - Ensure you have access to the project

4. **Memory Issues**
   - Reduce the number of runs with `--limit`
   - Use `--quick` for faster analysis

### Performance Tips

- Use `--limit` to process fewer runs for faster analysis
- Use `--quick` for rapid overview
- Consider using smaller Ollama models for faster AI analysis
- Process runs in batches for large projects

## Contributing

To extend the tool:

1. Add new analysis types in `ai_analyzer.py`
2. Create new visualizations in `visualizer.py`
3. Extend report templates in `report_generator.py`
4. Add new CLI options in `main.py`

## License

This tool is part of the ml-dev project and follows the same license terms. 