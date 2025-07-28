#!/usr/bin/env python3
"""
Configuration Manager for Wandb Analysis Summary Tool

This script provides an interactive way to manage configuration settings.
"""

import os
import sys
import json
import yaml
from pathlib import Path
from typing import Dict, Any, Optional

from config import Config, load_config, create_default_config_file


class ConfigManager:
    """Interactive configuration manager."""
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize the configuration manager.
        
        Args:
            config_path: Path to configuration file
        """
        self.config_path = config_path or "wandb_summary_config.yaml"
        self.config = load_config(self.config_path)
    
    def show_current_config(self):
        """Display current configuration."""
        print("🔧 Current Configuration")
        print("=" * 50)
        
        # AI Model settings
        print("\n🤖 AI Model Settings:")
        print(f"  Ollama URL: {self.config.ai_model.ollama_url}")
        print(f"  Default Model: {self.config.ai_model.default_model}")
        print(f"  Temperature: {self.config.ai_model.temperature}")
        print(f"  Max Tokens: {self.config.ai_model.max_tokens}")
        
        # Output settings
        print("\n📁 Output Settings:")
        print(f"  Base Directory: {self.config.output.base_dir}")
        print(f"  Data Directory: {self.config.output.get_data_path()}")
        print(f"  Analysis Directory: {self.config.output.get_analysis_path()}")
        print(f"  Visualizations Directory: {self.config.output.get_visualizations_path()}")
        print(f"  Reports Directory: {self.config.output.get_reports_path()}")
        
        # Visualization settings
        print("\n📊 Visualization Settings:")
        print(f"  Generate Overview: {self.config.visualization.generate_overview}")
        print(f"  Generate Performance: {self.config.visualization.generate_performance}")
        print(f"  Generate Interactive: {self.config.visualization.generate_interactive}")
        print(f"  Figure Size: {self.config.visualization.figure_size}")
        print(f"  DPI: {self.config.visualization.dpi}")
        
        # Report settings
        print("\n📄 Report Settings:")
        print(f"  Generate HTML: {self.config.report.generate_html}")
        print(f"  Generate Markdown: {self.config.report.generate_markdown}")
        print(f"  Company Name: {self.config.report.company_name}")
        print(f"  Report Title: {self.config.report.report_title}")
        
        # Analysis settings
        print("\n🔍 Analysis Settings:")
        print(f"  Max Runs to Process: {self.config.analysis.max_runs_to_process}")
        print(f"  Default Limit: {self.config.analysis.default_limit}")
        print(f"  Quick Analysis Limit: {self.config.analysis.quick_analysis_limit}")
        print(f"  Priority Metrics: {', '.join(self.config.analysis.priority_metrics[:5])}...")
        
        # Logging settings
        print("\n📝 Logging Settings:")
        print(f"  Log Level: {self.config.logging.log_level}")
        print(f"  Log to File: {self.config.logging.log_to_file}")
        print(f"  Verbose: {self.config.logging.verbose}")
    
    def edit_config(self):
        """Interactive configuration editor."""
        print("\n✏️  Configuration Editor")
        print("=" * 50)
        
        while True:
            print("\nSelect a section to edit:")
            print("1. AI Model Settings")
            print("2. Output Settings")
            print("3. Visualization Settings")
            print("4. Report Settings")
            print("5. Analysis Settings")
            print("6. Logging Settings")
            print("7. Save and Exit")
            print("8. Exit without saving")
            
            choice = input("\nEnter your choice (1-8): ").strip()
            
            if choice == "1":
                self._edit_ai_model_settings()
            elif choice == "2":
                self._edit_output_settings()
            elif choice == "3":
                self._edit_visualization_settings()
            elif choice == "4":
                self._edit_report_settings()
            elif choice == "5":
                self._edit_analysis_settings()
            elif choice == "6":
                self._edit_logging_settings()
            elif choice == "7":
                self.save_config()
                print("✅ Configuration saved!")
                break
            elif choice == "8":
                print("❌ Exiting without saving changes.")
                break
            else:
                print("❌ Invalid choice. Please try again.")
    
    def _edit_ai_model_settings(self):
        """Edit AI model settings."""
        print("\n🤖 AI Model Settings")
        print("-" * 30)
        
        # Ollama URL
        new_url = input(f"Ollama URL (current: {self.config.ai_model.ollama_url}): ").strip()
        if new_url:
            self.config.ai_model.ollama_url = new_url
        
        # Default model
        new_model = input(f"Default Model (current: {self.config.ai_model.default_model}): ").strip()
        if new_model:
            self.config.ai_model.default_model = new_model
        
        # Temperature
        try:
            new_temp = input(f"Temperature (current: {self.config.ai_model.temperature}): ").strip()
            if new_temp:
                self.config.ai_model.temperature = float(new_temp)
        except ValueError:
            print("❌ Invalid temperature value. Keeping current value.")
        
        # Max tokens
        try:
            new_tokens = input(f"Max Tokens (current: {self.config.ai_model.max_tokens}): ").strip()
            if new_tokens:
                self.config.ai_model.max_tokens = int(new_tokens)
        except ValueError:
            print("❌ Invalid max tokens value. Keeping current value.")
    
    def _edit_output_settings(self):
        """Edit output settings."""
        print("\n📁 Output Settings")
        print("-" * 30)
        
        # Base directory
        new_base = input(f"Base Directory (current: {self.config.output.base_dir}): ").strip()
        if new_base:
            self.config.output.base_dir = new_base
        
        # Data directory
        new_data = input(f"Data Directory (current: {self.config.output.data_dir}): ").strip()
        if new_data:
            self.config.output.data_dir = new_data
        
        # Analysis directory
        new_analysis = input(f"Analysis Directory (current: {self.config.output.analysis_dir}): ").strip()
        if new_analysis:
            self.config.output.analysis_dir = new_analysis
    
    def _edit_visualization_settings(self):
        """Edit visualization settings."""
        print("\n📊 Visualization Settings")
        print("-" * 30)
        
        # Generate overview
        overview = input(f"Generate Overview (current: {self.config.visualization.generate_overview}) [y/n]: ").strip().lower()
        if overview in ['y', 'yes']:
            self.config.visualization.generate_overview = True
        elif overview in ['n', 'no']:
            self.config.visualization.generate_overview = False
        
        # Generate interactive
        interactive = input(f"Generate Interactive (current: {self.config.visualization.generate_interactive}) [y/n]: ").strip().lower()
        if interactive in ['y', 'yes']:
            self.config.visualization.generate_interactive = True
        elif interactive in ['n', 'no']:
            self.config.visualization.generate_interactive = False
        
        # DPI
        try:
            new_dpi = input(f"DPI (current: {self.config.visualization.dpi}): ").strip()
            if new_dpi:
                self.config.visualization.dpi = int(new_dpi)
        except ValueError:
            print("❌ Invalid DPI value. Keeping current value.")
    
    def _edit_report_settings(self):
        """Edit report settings."""
        print("\n📄 Report Settings")
        print("-" * 30)
        
        # Company name
        new_company = input(f"Company Name (current: {self.config.report.company_name}): ").strip()
        if new_company:
            self.config.report.company_name = new_company
        
        # Report title
        new_title = input(f"Report Title (current: {self.config.report.report_title}): ").strip()
        if new_title:
            self.config.report.report_title = new_title
        
        # Generate HTML
        html = input(f"Generate HTML (current: {self.config.report.generate_html}) [y/n]: ").strip().lower()
        if html in ['y', 'yes']:
            self.config.report.generate_html = True
        elif html in ['n', 'no']:
            self.config.report.generate_html = False
        
        # Generate Markdown
        md = input(f"Generate Markdown (current: {self.config.report.generate_markdown}) [y/n]: ").strip().lower()
        if md in ['y', 'yes']:
            self.config.report.generate_markdown = True
        elif md in ['n', 'no']:
            self.config.report.generate_markdown = False
    
    def _edit_analysis_settings(self):
        """Edit analysis settings."""
        print("\n🔍 Analysis Settings")
        print("-" * 30)
        
        # Max runs
        try:
            new_max = input(f"Max Runs to Process (current: {self.config.analysis.max_runs_to_process}): ").strip()
            if new_max:
                self.config.analysis.max_runs_to_process = int(new_max)
        except ValueError:
            print("❌ Invalid max runs value. Keeping current value.")
        
        # Default limit
        try:
            new_limit = input(f"Default Limit (current: {self.config.analysis.default_limit}): ").strip()
            if new_limit:
                self.config.analysis.default_limit = int(new_limit)
        except ValueError:
            print("❌ Invalid default limit value. Keeping current value.")
        
        # Quick analysis limit
        try:
            new_quick = input(f"Quick Analysis Limit (current: {self.config.analysis.quick_analysis_limit}): ").strip()
            if new_quick:
                self.config.analysis.quick_analysis_limit = int(new_quick)
        except ValueError:
            print("❌ Invalid quick analysis limit value. Keeping current value.")
    
    def _edit_logging_settings(self):
        """Edit logging settings."""
        print("\n📝 Logging Settings")
        print("-" * 30)
        
        # Log level
        levels = ["DEBUG", "INFO", "WARNING", "ERROR"]
        print(f"Available log levels: {', '.join(levels)}")
        new_level = input(f"Log Level (current: {self.config.logging.log_level}): ").strip().upper()
        if new_level in levels:
            self.config.logging.log_level = new_level
        elif new_level:
            print("❌ Invalid log level. Keeping current value.")
        
        # Verbose
        verbose = input(f"Verbose (current: {self.config.logging.verbose}) [y/n]: ").strip().lower()
        if verbose in ['y', 'yes']:
            self.config.logging.verbose = True
        elif verbose in ['n', 'no']:
            self.config.logging.verbose = False
        
        # Log to file
        log_file = input(f"Log to File (current: {self.config.logging.log_to_file}) [y/n]: ").strip().lower()
        if log_file in ['y', 'yes']:
            self.config.logging.log_to_file = True
        elif log_file in ['n', 'no']:
            self.config.logging.log_to_file = False
    
    def save_config(self):
        """Save configuration to file."""
        self.config.save_to_file(self.config_path)
    
    def export_config(self, format: str = "yaml"):
        """Export configuration in different formats."""
        if format.lower() == "json":
            output_path = self.config_path.replace(".yaml", ".json")
            with open(output_path, 'w') as f:
                json.dump(self.config.to_dict(), f, indent=2)
        else:
            output_path = self.config_path.replace(".json", ".yaml")
            with open(output_path, 'w') as f:
                yaml.dump(self.config.to_dict(), f, default_flow_style=False, indent=2)
        
        print(f"✅ Configuration exported to: {output_path}")
    
    def validate_config(self):
        """Validate configuration settings."""
        print("\n🔍 Validating Configuration")
        print("=" * 50)
        
        issues = []
        
        # Check AI model settings
        if not self.config.ai_model.ollama_url.startswith(("http://", "https://")):
            issues.append("Ollama URL should start with http:// or https://")
        
        if self.config.ai_model.temperature < 0 or self.config.ai_model.temperature > 2:
            issues.append("Temperature should be between 0 and 2")
        
        if self.config.ai_model.max_tokens <= 0:
            issues.append("Max tokens should be positive")
        
        # Check output settings
        if not self.config.output.base_dir:
            issues.append("Base directory cannot be empty")
        
        # Check analysis settings
        if self.config.analysis.max_runs_to_process <= 0:
            issues.append("Max runs to process should be positive")
        
        if self.config.analysis.default_limit <= 0:
            issues.append("Default limit should be positive")
        
        if self.config.analysis.quick_analysis_limit <= 0:
            issues.append("Quick analysis limit should be positive")
        
        if issues:
            print("❌ Configuration issues found:")
            for issue in issues:
                print(f"  - {issue}")
        else:
            print("✅ Configuration is valid!")
        
        return len(issues) == 0


def main():
    """Main function for configuration manager."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Wandb Analysis Configuration Manager")
    parser.add_argument("--config", help="Path to configuration file")
    parser.add_argument("--show", action="store_true", help="Show current configuration")
    parser.add_argument("--edit", action="store_true", help="Edit configuration interactively")
    parser.add_argument("--validate", action="store_true", help="Validate configuration")
    parser.add_argument("--export", choices=["yaml", "json"], help="Export configuration")
    parser.add_argument("--create", action="store_true", help="Create default configuration")
    
    args = parser.parse_args()
    
    if args.create:
        config_path = args.config or "wandb_summary_config.yaml"
        create_default_config_file(config_path)
        print(f"✅ Default configuration created at: {config_path}")
        return
    
    # Initialize config manager
    manager = ConfigManager(args.config)
    
    if args.show:
        manager.show_current_config()
    elif args.edit:
        manager.edit_config()
    elif args.validate:
        manager.validate_config()
    elif args.export:
        manager.export_config(args.export)
    else:
        # Interactive mode
        print("🔧 Wandb Analysis Configuration Manager")
        print("=" * 50)
        
        while True:
            print("\nSelect an option:")
            print("1. Show current configuration")
            print("2. Edit configuration")
            print("3. Validate configuration")
            print("4. Export configuration")
            print("5. Exit")
            
            choice = input("\nEnter your choice (1-5): ").strip()
            
            if choice == "1":
                manager.show_current_config()
            elif choice == "2":
                manager.edit_config()
            elif choice == "3":
                manager.validate_config()
            elif choice == "4":
                format_choice = input("Export format (yaml/json): ").strip().lower()
                if format_choice in ["yaml", "json"]:
                    manager.export_config(format_choice)
                else:
                    print("❌ Invalid format. Use 'yaml' or 'json'.")
            elif choice == "5":
                print("👋 Goodbye!")
                break
            else:
                print("❌ Invalid choice. Please try again.")


if __name__ == "__main__":
    main() 