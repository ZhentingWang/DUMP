# Curriculum Learning for VERL

This document explains how to use the curriculum learning feature in the VERL framework.

## Overview

Curriculum learning is a training strategy where the model learns from easier tasks first and gradually progresses to more difficult tasks. This approach can improve the model's overall performance and training efficiency.

In the VERL framework, curriculum learning is implemented using a Bayesian approach that:

1. Tracks performance metrics for each data source (task type)
2. Estimates the learning potential of each task
3. Dynamically adjusts sampling weights to favor tasks with the highest learning potential

## How It Works

The curriculum learning module consists of three main components:

1. **BayesianDifficultyEstimator**: Estimates task difficulty using Bayesian statistics, tracking performance metrics and updating based on rewards and advantages.

2. **BayesianCurriculumController**: Manages multiple difficulty estimators and computes optimal sampling weights using an Upper Confidence Bound (UCB) approach combined with Thompson sampling.

3. **CurriculumSampler**: Implements curriculum-based sampling, dynamically adjusting weights to focus on tasks with the highest learning potential.

The system balances exploration (trying different tasks) and exploitation (focusing on promising tasks) to optimize learning.

## How to Use

### 1. Enable Curriculum Learning

In your configuration file (`ppo_trainer.yaml`), enable curriculum learning by setting:

```yaml
data:
  # ... other data settings ...
  enable_curriculum_learning: True
  data_source_key: "task_type"  # Replace with your dataset's column name that identifies the task
```

### 2. Prepare Your Dataset

Your dataset must include a column that identifies different data sources or task types. This column should be specified in the `data_source_key` configuration.

For example, your parquet file might have columns:
- `prompt`: The input prompt
- `completion`: The target completion
- `task_type`: The type or source of the task (e.g., "math", "coding", "reasoning")

### 3. Monitor Training

During training, the system will automatically:
- Track performance on different tasks
- Adjust sampling weights based on learning progress
- Log statistics about the curriculum learning process

You can monitor these statistics in your training logs or visualization tools like wandb, including:
- Expected values for each task
- Reward rates for each task
- Sampling weights for each task

## Enhanced Visualizations

The curriculum learning module now includes rich visualizations to help you understand and monitor the training process. When using wandb as a logging backend, you'll get access to:

### Task Weight Distribution

A pie chart showing the current sampling weight distribution across different tasks:

![Weight Distribution](https://i.imgur.com/example1.png)

### Learning Progress Tracking

Line charts showing how weights, expected values, and reward rates change over time:

![Learning Progress](https://i.imgur.com/example2.png)

### Interactive Task Analysis Dashboard

An interactive dashboard that allows you to analyze the relationship between:
- Task difficulty (reward rate)
- Learning potential (expected value)
- Sampling priority (weight)
- Training exposure (sample count)

![Task Analysis](https://i.imgur.com/example3.png)

### Sample Distribution Visualization

A bar chart showing how many samples from each task type have been processed:

![Sample Distribution](https://i.imgur.com/example4.png)

## Advanced Configuration

The curriculum learning system is designed to work with minimal configuration. However, you can adjust parameters in the `BayesianDifficultyEstimator` and `BayesianCurriculumController` classes for more fine-grained control:

- `alpha` and `beta`: Prior parameters for the Beta distribution
- `mu` and `sigma`: Prior parameters for the normal distribution
- `window_size`: Number of recent updates to consider when calculating trends
- `exploration_weight`: Balance between exploration and exploitation

## Limitations and Considerations

- Curriculum learning works best when the dataset contains a variety of task difficulties
- The system needs some validation steps to gather initial performance data
- Performance may vary depending on the nature and distribution of your tasks 