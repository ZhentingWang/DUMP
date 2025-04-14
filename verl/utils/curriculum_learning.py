"""
Curriculum Learning for PPO Training

This module provides curriculum learning utilities for training on various data sources 
with learnability-based sampling. It implements a Bayesian approach to estimate task learnability
and dynamically adjust sampling weights to focus on tasks with the highest learning potential.

How to use:
1. Set 'enable_curriculum_learning: True' in your config file
2. Set 'data_source_key' to the column in your dataset that identifies different data sources/tasks
3. The system will automatically track performance on different tasks and adjust sampling weights

The curriculum learning system includes:
- LearnabilityEstimator: Tracks performance metrics for individual tasks
- CurriculumController: Manages multiple estimators and computes optimal sampling weights
- CurriculumSampler: Implements the torch.utils.data.Sampler interface for curriculum-based sampling

The system uses a combination of Upper Confidence Bound (UCB) and Thompson sampling to balance
exploration and exploitation, focusing on tasks that are challenging but achievable.
"""

import numpy as np
import torch
import logging
import random
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any, Union
from scipy.stats import beta as beta_dist
from scipy.stats import norm as norm_dist
from collections import defaultdict

logger = logging.getLogger(__name__)

@dataclass
class LearnabilityEstimator:
    """Learnability estimator for curriculum learning."""
    # Beta distribution parameters (for reward modeling)
    alpha: float = 1.0
    beta: float = 1.0
    # Normal distribution parameters (for advantage function modeling)
    mu: float = 0.0
    # Additional statistics
    n_samples: int = 0
    total_reward: float = 0.0
    # Sliding window for recent performance tracking
    window_size: int = 300  # sliding window size
    recent_rewards: List[float] = field(default_factory=list)
    recent_advantages: List[float] = field(default_factory=list)
    
    def update(self, rewards: np.ndarray, advantages: np.ndarray):
        """update based on new data."""
        n = len(rewards)
        if n == 0:
            return
            
        # Update recent rewards and advantages
        self.recent_rewards.extend(rewards.tolist())
        self.recent_advantages.extend(advantages.tolist())
        
        # Maintain window size
        if len(self.recent_rewards) > self.window_size:
            self.recent_rewards = self.recent_rewards[-self.window_size:]
            self.recent_advantages = self.recent_advantages[-self.window_size:]
        
        # Update Beta distribution parameters based on successes/failures
        successes = np.sum(rewards >= 3)  # Consider reward >= 1 as success
        failures = len(rewards) - successes
        
        # Accumulate Beta parameters (without resetting)
        self.alpha += successes
        self.beta += failures
        
        # Update normal distribution using window data with absolute advantages
        window_advantages = np.abs(np.array(self.recent_advantages)) # Take absolute value here
        self.mu = np.mean(window_advantages)
        
        # Update statistics
        self.n_samples += n
        self.total_reward += np.sum(rewards)
    
    @property
    def expected_value(self) -> float:
        """Calculate expected value from the current distribution."""
        reward_rate = self.alpha / (self.alpha + self.beta)
        return reward_rate * self.mu

    @property
    def recent_reward_rate(self) -> float:
        """Calculate the recent success rate within the sliding window."""
        if not self.recent_rewards:
            return 0.0
        window_rewards = np.array(self.recent_rewards)
        successes = np.sum(window_rewards >= 1)
        return successes / len(window_rewards) if len(window_rewards) > 0 else 0.0


class CurriculumController:
    """Controller for task curriculum learning."""
    def __init__(self, data_sources: List[str], trend_window: int = 50):
        """Initialize the curriculum controller."""
        self.data_sources = list(set(data_sources))  # Ensure unique sources
        self.trend_window = trend_window
        self.estimators = {
            source: LearnabilityEstimator()
            for source in self.data_sources
        }
        
        # Store weight history for each source
        self.weight_history = {
            source: []
            for source in self.data_sources
        }
    
    def compute_trend(self, history):
        """Calculate trend using linear regression on weight history."""
        if len(history) < 2:
            return 0.0
            
        # Use only the most recent trend_window data points
        history = history[-self.trend_window:]
        n = len(history)
        x = np.arange(n)
        y = np.array(history)
        
        # Calculate linear regression slope
        x_mean = x.mean()
        y_mean = y.mean()
        
        # Avoid division by zero
        denominator = np.sum((x - x_mean) ** 2)
        if denominator == 0:
            return 0.0
            
        slope = np.sum((x - x_mean) * (y - y_mean)) / denominator
        return slope
        
    def compute_sampling_weights(self) -> Dict[str, float]:
        """Compute sampling weights for each data source."""
        stats = self.get_source_stats()
        
        # Calculate UCB scores for each source
        base_scores = []
        source_to_index = {}
        
        # Calculate total samples across all sources
        total_samples = sum(stat['n_samples'] for stat in stats.values())
        total_samples = max(1, total_samples)  # Avoid division by zero
        
        for i, source in enumerate(self.data_sources):
            source_to_index[source] = i
            stat = stats[source]
            
            # Weight parameters
            uncertainty_weight = 1.0  # uncertainty term weight
            exploration_weight = 1.0  # exploration term weight
            
            # Base UCB score using the advantage mean
            ucb_score = stat['advantage_mean']
            
            # Add exploration bonus (more exploration for less sampled sources)
            exploration_bonus = exploration_weight * np.sqrt(
                2 * np.log(total_samples + 1) / (stat['n_samples'] + 1)
            )
            ucb_score += exploration_bonus
            
            base_scores.append(ucb_score)
            
            logger.debug(f"Source {source} UCB calculation:")
            logger.debug(f"  Advantage mean: {stat['advantage_mean']:.3f}")
            logger.debug(f"  Exploration bonus: {exploration_bonus:.3f}")
            logger.debug(f"  Final UCB score: {ucb_score:.3f}")
        
        # Convert scores to a probability distribution using softmax
        temperature = 0.1  # controls distribution smoothness
        scores_array = np.array(base_scores)
        
        # Softmax calculation with temperature
        # Subtract max for numerical stability
        if len(scores_array) > 0:
            max_score = np.max(scores_array)
            exp_scores = np.exp((scores_array - max_score) / temperature)
            weights_array = exp_scores / np.sum(exp_scores)
            
            # Create a dictionary mapping sources to weights
            weights = {source: weights_array[source_to_index[source]] for source in self.data_sources}
        else:
            # Equal weights if no scores available
            weights = {source: 1.0/len(self.data_sources) for source in self.data_sources}
        
        # Update weight history
        for source, weight in weights.items():
            self.weight_history[source].append(weight)
            # Keep history within window size
            if len(self.weight_history[source]) > self.trend_window:
                self.weight_history[source] = self.weight_history[source][-self.trend_window:]
        
        return weights
    
    def get_source_stats(self) -> Dict[str, Dict[str, float]]:
        """Get statistics for each data source."""
        stats = {}
        for source in self.data_sources:
            estimator = self.estimators[source]
            stats[source] = {
                'expected_value': estimator.expected_value,
                'reward_rate': estimator.alpha / (estimator.alpha + estimator.beta),
                'recent_reward_rate': estimator.recent_reward_rate,
                'advantage_mean': estimator.mu,
                'n_samples': estimator.n_samples
            }
        return stats


class CurriculumSampler:
    """Sampler that implements curriculum learning based on task difficulty."""
    def __init__(self, 
                 dataset,
                 data_source_key: str = 'data_source',
                 batch_size: int = 1,
                 seed: Optional[int] = None):
        """Initialize the curriculum sampler."""
        self.dataset = dataset
        self.data_source_key = data_source_key
        self.batch_size = batch_size
        self.rng = random.Random(seed)
        
        # Group dataset items by data source
        self.source_indices = defaultdict(list)
        
        # Keep track of problematic indices for debugging
        self.problematic_indices = []
        
        # Pre-check prompt format validity to avoid sampling problematic examples
        for i in range(len(dataset)):
            try:
                item = dataset[i]
                if isinstance(item, dict) and self.data_source_key in item:
                    source = item[self.data_source_key]
                    
                    # Verify that this sample won't cause problems during training
                    # Try to access the prompt from the dataset - this will verify format
                    # We don't need to actually use the result, just check it works
                    _ = dataset[i]
                    
                    # If successful, add to source indices
                    self.source_indices[source].append(i)
                else:
                    # Handle items without a data source by creating a default source
                    # But still verify it won't cause problems
                    _ = dataset[i]
                    self.source_indices["default"].append(i)
            except Exception as e:
                # If any error occurs during checking, track it but don't include the sample
                self.problematic_indices.append((i, str(e)))
                print(f"Warning: index {i} skipped due to error: {e}")
                continue
        
        # Report number of problematic indices
        if self.problematic_indices:
            print(f"Warning: {len(self.problematic_indices)} samples excluded from curriculum sampler due to format issues")
            
        # If no valid sources were found, create a default one with valid samples
        if not self.source_indices:
            print("Warning: No valid source groups found, creating default source with all valid samples")
            
            # Identify all valid indices by directly testing each sample
            valid_indices = []
            for i in range(len(dataset)):
                if i not in [x[0] for x in self.problematic_indices]:
                    try:
                        _ = dataset[i]
                        valid_indices.append(i)
                    except:
                        pass
                        
            self.source_indices["default"] = valid_indices
            
        # Initialize the curriculum controller with all data sources
        self.curriculum_controller = CurriculumController(
            data_sources=list(self.source_indices.keys())
        )
        
        # Initialize sampling weights equally
        self._weights = {source: 1.0 / len(self.source_indices) 
                        for source in self.source_indices}
    
    def update_weights(self, 
                      source_rewards: Dict[str, List[float]], 
                      source_advantages: Dict[str, List[float]]):
        """Update the curriculum controller with new rewards and advantages."""
        for source in source_rewards:
            if source in self.curriculum_controller.estimators:
                rewards = np.array(source_rewards[source])
                advantages = np.array(source_advantages[source])
                self.curriculum_controller.estimators[source].update(rewards, advantages)
        
        # Recompute sampling weights
        self._weights = self.curriculum_controller.compute_sampling_weights()
    
    def get_source_stats(self):
        """Get statistics for each data source by delegating to the curriculum controller."""
        stats = self.curriculum_controller.get_source_stats()
        
        # Add sampling weights to the statistics
        for source in stats:
            stats[source]['sampling_weight'] = self._weights.get(source, 0.0)
        
        # Calculate weight trend if possible (could add this as a separate field)
        weight_trend = 0.0
        if hasattr(self.curriculum_controller, 'compute_trend') and hasattr(self.curriculum_controller, 'weight_history'):
            trends = {}
            for source, history in self.curriculum_controller.weight_history.items():
                if history:  # Only compute trend if we have history
                    trends[source] = self.curriculum_controller.compute_trend(history)
            
            # Average trend across all sources
            if trends:
                weight_trend = sum(trends.values()) / len(trends)
        
        # Add overall stats
        stats['weight_trend'] = weight_trend
        
        return stats
    
    def __iter__(self):
        """Create an iterator for sampling from the dataset."""
        # Convert weights to a list of (source, weight) pairs for selection
        sources = list(self._weights.keys())
        weights = [self._weights[source] for source in sources]
        
        # Yield indices in batches
        n = len(self.dataset)
        indices = []
        
        total_batches = len(self)
        batches_yielded = 0
        
        # Keep track of errors during sampling
        sampling_errors = 0
        max_errors = 100  # Maximum number of errors before giving up
        
        while batches_yielded < total_batches and sampling_errors < max_errors:
            # Sample a data source according to weights
            if sources and weights and sum(len(self.source_indices[s]) for s in sources) > 0:
                try:
                    # Sample source with non-zero indices
                    valid_sources = [s for s in sources if len(self.source_indices[s]) > 0]
                    if not valid_sources:
                        # If no valid sources, skip this iteration
                        sampling_errors += 1
                        continue
                        
                    valid_weights = [self._weights[s] for s in valid_sources]
                    # Normalize weights
                    weight_sum = sum(valid_weights)
                    if weight_sum > 0:
                        valid_weights = [w/weight_sum for w in valid_weights]
                    else:
                        valid_weights = [1.0/len(valid_sources) for _ in valid_sources]
                        
                    source = self.rng.choices(valid_sources, weights=valid_weights, k=1)[0]
                    
                    # If this source has no samples, skip it (should not happen with filtering above)
                    if not self.source_indices[source]:
                        sampling_errors += 1
                        continue
                        
                    # Sample an index from this source
                    idx = self.rng.choice(self.source_indices[source])
                    
                    # Verify this index still works (double-check)
                    _ = self.dataset[idx]
                    
                    indices.append(idx)
                except Exception as e:
                    # If error during sampling, try again
                    sampling_errors += 1
                    print(f"Error during sampling: {e}")
                    continue
            else:
                # Fallback to random sampling if no sources/weights
                try:
                    # Ensure we only sample valid indices
                    all_valid_indices = []
                    for source_indices in self.source_indices.values():
                        all_valid_indices.extend(source_indices)
                    
                    if not all_valid_indices:
                        sampling_errors += 1
                        continue
                        
                    idx = self.rng.choice(all_valid_indices)
                    indices.append(idx)
                except Exception as e:
                    sampling_errors += 1
                    print(f"Error during fallback sampling: {e}")
                    continue
            
            # When we have a full batch, yield it
            if len(indices) == self.batch_size:
                yield from indices  
                indices = []
                batches_yielded += 1
                
        # If we hit max errors, warn the user
        if sampling_errors >= max_errors:
            print(f"WARNING: Hit maximum number of sampling errors ({max_errors}). Some batches may be incomplete.")
    
    def __len__(self):
        """Return the total number of batches."""
        return len(self.dataset)