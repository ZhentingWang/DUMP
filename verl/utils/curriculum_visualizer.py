"""
Visualization tools for curriculum learning in VERL.
These tools help monitor the curriculum learning process through wandb visualizations.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import io
from typing import Dict, List, Any, Optional
import logging
import json

logger = logging.getLogger(__name__)

class CurriculumVisualizer:
    """Visualizer for curriculum learning metrics with wandb integration."""
    
    def __init__(self):
        """Initialize the visualizer."""
        self.history = {
            'weights': {},
            'expected_values': {},
            'reward_rates': {},
            'sample_counts': {},
            'steps': []
        }
    
    def update_history(self, step: int, source_stats: Dict[str, Dict[str, float]]):
        """Update history with new data for visualization."""
        self.history['steps'].append(step)
        
        # Extract and store relevant metrics
        for source, stats in source_stats.items():
            if source not in self.history['weights']:
                # Initialize history for new sources
                self.history['weights'][source] = []
                self.history['expected_values'][source] = []
                self.history['reward_rates'][source] = []
                self.history['sample_counts'][source] = []
            
            # Append current values
            self.history['weights'][source].append(stats.get('sampling_weight', 0.0))
            self.history['expected_values'][source].append(stats.get('expected_value', 0.0))
            self.history['reward_rates'][source].append(stats.get('reward_rate', 0.0))
            self.history['sample_counts'][source].append(stats.get('n_samples', 0))
    
    def create_visualizations(self) -> Dict[str, Any]:
        """Create visualizations for wandb."""
        if not self.history['steps']:
            return {}
        
        visualizations = {}
        
        try:
            import wandb
            # Add weight distribution pie chart
            visualizations.update(self._create_weight_pie_chart())
            
            # Add weight history line chart
            visualizations.update(self._create_weight_line_chart())
            
            # Add expected value history
            visualizations.update(self._create_expected_value_chart())
            
            # Add learning progress chart
            visualizations.update(self._create_learning_progress_chart())
            
            # Add sample distribution chart
            visualizations.update(self._create_sample_distribution_chart())
            
            # Add interactive task analysis dashboard
            visualizations.update(self._create_interactive_dashboard())
            
        except ImportError:
            logger.warning("wandb not installed, skipping curriculum visualizations")
        except Exception as e:
            logger.error(f"Error creating curriculum visualizations: {e}")
        
        return visualizations
    
    def _create_weight_pie_chart(self) -> Dict[str, Any]:
        """Create a pie chart of current weights."""
        import wandb
        
        # Get the latest weights
        weights = {}
        for source, weight_history in self.history['weights'].items():
            if weight_history:  # Check if not empty
                weights[source] = weight_history[-1]
        
        # Create the pie chart
        fig, ax = plt.subplots(figsize=(8, 8))
        labels = list(weights.keys())
        sizes = list(weights.values())
        
        # Ensure all weights are positive for the pie chart
        sizes = [max(0.0001, size) for size in sizes]
        
        # Create a colorful pie chart
        ax.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90, 
               shadow=True, explode=[0.05] * len(labels))
        ax.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle
        plt.title('Current Curriculum Sampling Weights', fontsize=16)
        
        # Convert to wandb Image
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        plt.close(fig)
        
        return {"curriculum/weight_pie_chart": wandb.Image(buf)}
    
    def _create_weight_line_chart(self) -> Dict[str, Any]:
        """Create a line chart of weight history."""
        import wandb
        import matplotlib.cm as cm
        
        # Create the line chart
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Use a colormap with distinct colors
        colormap = cm.get_cmap('tab10', len(self.history['weights']))
        
        for i, (source, weight_history) in enumerate(self.history['weights'].items()):
            color = colormap(i)
            ax.plot(self.history['steps'], weight_history, 
                    label=source, linewidth=2, marker='o', markersize=4, color=color)
        
        ax.set_xlabel('Training Steps', fontsize=12)
        ax.set_ylabel('Sampling Weight', fontsize=12)
        ax.set_title('Curriculum Sampling Weights Over Time', fontsize=16)
        ax.grid(True, alpha=0.3)
        ax.legend(loc='best')
        
        # Convert to wandb Image
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        plt.close(fig)
        
        return {"curriculum/weight_history": wandb.Image(buf)}
    
    def _create_expected_value_chart(self) -> Dict[str, Any]:
        """Create a line chart of expected value history."""
        import wandb
        import matplotlib.cm as cm
        
        # Create the line chart
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Use a colormap with distinct colors
        colormap = cm.get_cmap('tab10', len(self.history['expected_values']))
        
        for i, (source, value_history) in enumerate(self.history['expected_values'].items()):
            color = colormap(i)
            ax.plot(self.history['steps'], value_history, 
                    label=source, linewidth=2, marker='o', markersize=4, color=color)
        
        ax.set_xlabel('Training Steps', fontsize=12)
        ax.set_ylabel('Expected Value', fontsize=12)
        ax.set_title('Task Expected Values Over Time', fontsize=16)
        ax.grid(True, alpha=0.3)
        ax.legend(loc='best')
        
        # Convert to wandb Image
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        plt.close(fig)
        
        return {"curriculum/expected_value_history": wandb.Image(buf)}
    
    def _create_learning_progress_chart(self) -> Dict[str, Any]:
        """Create a chart showing learning progress through reward rates."""
        import wandb
        import matplotlib.cm as cm
        
        # Create the line chart
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Use a colormap with distinct colors
        colormap = cm.get_cmap('tab10', len(self.history['reward_rates']))
        
        for i, (source, rate_history) in enumerate(self.history['reward_rates'].items()):
            color = colormap(i)
            ax.plot(self.history['steps'], rate_history, 
                    label=source, linewidth=2, marker='o', markersize=4, color=color)
        
        ax.set_xlabel('Training Steps', fontsize=12)
        ax.set_ylabel('Reward Rate', fontsize=12)
        ax.set_title('Task Learning Progress', fontsize=16)
        ax.grid(True, alpha=0.3)
        ax.legend(loc='best')
        
        # Convert to wandb Image
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        plt.close(fig)
        
        return {"curriculum/learning_progress": wandb.Image(buf)}
    
    def _create_sample_distribution_chart(self) -> Dict[str, Any]:
        """Create a stacked bar chart of sample counts."""
        import wandb
        
        # Get the latest sample counts
        sample_counts = {}
        for source, count_history in self.history['sample_counts'].items():
            if count_history:  # Check if not empty
                sample_counts[source] = count_history[-1]
        
        # Create the bar chart
        fig, ax = plt.subplots(figsize=(10, 6))
        
        sources = list(sample_counts.keys())
        counts = list(sample_counts.values())
        
        # Create horizontal bar chart
        y_pos = np.arange(len(sources))
        ax.barh(y_pos, counts)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(sources)
        ax.invert_yaxis()  # Labels read top-to-bottom
        ax.set_xlabel('Number of Samples Seen', fontsize=12)
        ax.set_title('Curriculum Sample Distribution', fontsize=16)
        
        # Add count labels to the bars
        for i, v in enumerate(counts):
            ax.text(v + 0.1, i, str(v), va='center')
        
        # Convert to wandb Image
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        plt.close(fig)
        
        return {"curriculum/sample_distribution": wandb.Image(buf)}
    
    def _create_interactive_dashboard(self) -> Dict[str, Any]:
        """Create an interactive dashboard for task analysis using wandb's custom charts."""
        import wandb
        
        # Extract the latest data for each source
        latest_data = {}
        sources = list(self.history['weights'].keys())
        
        for source in sources:
            if self.history['weights'][source]:  # Check if not empty
                latest_data[source] = {
                    'weight': self.history['weights'][source][-1],
                    'expected_value': self.history['expected_values'][source][-1],
                    'reward_rate': self.history['reward_rates'][source][-1],
                    'sample_count': self.history['sample_counts'][source][-1]
                }
        
        # Create a wandb Table for the task analysis
        columns = ["data_source", "sampling_weight", "expected_value", "reward_rate", "sample_count"]
        task_table = wandb.Table(columns=columns)
        
        for source, data in latest_data.items():
            task_table.add_data(
                source,
                data['weight'],
                data['expected_value'],
                data['reward_rate'],
                data['sample_count']
            )
        
        # Create scatter plot custom chart
        scatter_chart = wandb.plot_table(
            "wandb/scatter/v0",
            task_table,
            {"x": "expected_value", "y": "reward_rate", "size": "sample_count", "color": "sampling_weight"},
            {"title": "Task Analysis: Reward Rate vs Expected Value by Task"}
        )
        
        # Create bar chart for weights
        bar_chart = wandb.plot_table(
            "wandb/bar/v0",
            task_table,
            {"x": "data_source", "y": "sampling_weight"},
            {"title": "Sampling Weight by Task"}
        )
        
        # Create bubble chart for task difficulty vs progress
        bubble_chart = wandb.plot_table(
            "wandb/scatter/v0",
            task_table,
            {"x": "reward_rate", "y": "sampling_weight", "size": "sample_count"},
            {"title": "Task Difficulty vs Sampling Priority"}
        )
        
        # Create a JSON object for the dashboard state
        dashboard_data = {
            "sources": sources,
            "latest_step": self.history['steps'][-1] if self.history['steps'] else 0,
            "metrics": {
                source: {
                    "weight": self.history['weights'][source][-1] if self.history['weights'][source] else 0,
                    "expected_value": self.history['expected_values'][source][-1] if self.history['expected_values'][source] else 0,
                    "reward_rate": self.history['reward_rates'][source][-1] if self.history['reward_rates'][source] else 0,
                    "sample_count": self.history['sample_counts'][source][-1] if self.history['sample_counts'][source] else 0
                } for source in sources
            }
        }
        
        # Return visualizations
        return {
            "curriculum/task_analysis_table": task_table,
            "curriculum/task_analysis_scatter": scatter_chart,
            "curriculum/task_weight_bar": bar_chart,
            "curriculum/task_difficulty_bubble": bubble_chart,
            "curriculum/dashboard_json": wandb.Html(json.dumps(dashboard_data, indent=2))
        }


def create_curriculum_wandb_plots(curriculum_stats: Dict[str, Dict[str, float]], step: int) -> Dict[str, Any]:
    """Create wandb visualizations for curriculum learning.
    
    This function creates various visualizations for curriculum learning metrics
    and returns a dictionary of wandb objects ready to be logged.
    
    Args:
        curriculum_stats: Dictionary of curriculum statistics by data source
        step: Current training step
        
    Returns:
        Dictionary of wandb visualization objects
    """
    # Create a static visualizer
    visualizer = CurriculumVisualizer()
    
    # Update with current data
    visualizer.update_history(step, curriculum_stats)
    
    # Generate visualizations
    return visualizer.create_visualizations() 