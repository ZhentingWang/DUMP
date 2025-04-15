# DUMP

This is code repo for paper "DUMP: Automated Distribution-Level Curriculum Learning for RL-based LLM Post-training"

## Project Overview

Recent advances in reinforcement learning (RL)-based post-training have led to notable improvements in large language models (LLMs), particularly in enhancing their reasoning capabilities. However, most existing methods treat the training data as a unified whole, overlooking the fact that modern LLM training often involves a mixture of data from diverse distributions—varying in both source and difficulty.

This project presents **DUMP**, a principled automated curriculum learning framework grounded in the notion of distribution-level learnability. Our core insight is that the magnitude of policy advantages reflects how much a model can still benefit from further training on a given distribution. Based on this, we propose:

- A distribution-level curriculum learning framework for RL-based LLM post-training
- An approach leveraging the Upper Confidence Bound (UCB) principle to dynamically adjust sampling probabilities for different distributions
- A method that prioritizes distributions with either high average advantage (exploitation) or low sample count (exploration)

We instantiate our curriculum learning framework with GRPO (Generalized Reward Proximal Optimization) as the underlying RL algorithm and demonstrate its effectiveness on logic reasoning datasets with multiple difficulties and sources. Our experiments show that our framework significantly improves convergence speed and final performance, highlighting the value of distribution-aware curriculum strategies in LLM post-training.

## Key Components

The project builds upon and integrates several existing components:

1. **verl** - An existing framework for implementing reinforcement learning training pipelines, with custom curriculum learning modifications
2. **kk/data_gen_kk.py** - Script used for generating Knights and Knaves logic puzzles datasets
3. **combined_logic_dataset** - Custom tools for combining multiple datasets

## Requirements

- Python 3.9
- CUDA support
- 8 A100/H100 GPUs for training (recommended)

## Installation

Follow these steps to set up the environment:

```bash
# Create and activate conda environment
conda create -n dump python=3.9
conda activate dump

# Clone the repository
git clone https://github.com/ZhentingWang/DUMP.git
cd DUMP

# Install PyTorch
pip install torch==2.4.0 --index-url https://download.pytorch.org/whl/cu121

# Install vllm and ray
pip3 install vllm==0.5.4 ray

# Install flash-attention
pip3 install flash-attn --no-build-isolation

# Install project dependencies
pip install -e .  # For verl integration

# Install additional tools
pip install wandb IPython matplotlib
pip install tensordict==0.5.0
pip install scipy
```

## Service Authentication

Before using the project, you need to authenticate with Weights & Biases (for experiment tracking) and Hugging Face (for model uploading):

### Weights & Biases (Wandb)

```bash
# Install wandb if you haven't already
pip install wandb

# Log in to wandb
wandb login

# Follow the instructions to enter your API key
```

You can find your Wandb API key in your [Wandb account settings](https://wandb.ai/settings).

### Hugging Face

```bash
# Install huggingface_hub if you haven't already
pip install huggingface_hub

# Log in to Hugging Face
huggingface-cli login

# Follow the instructions to enter your token
```

You can find or create your Hugging Face token in your [Hugging Face account settings](https://huggingface.co/settings/tokens).

The training scripts automatically use these credentials for:
- Logging training metrics and model performance to Wandb
- Optionally uploading trained models to Hugging Face (configured in the training scripts via `trainer.hf_account`)

**Important**: Before running the training scripts, you need to modify the `trainer.hf_account` parameter in the `.sh` files from `xxx` to your own Hugging Face username. For example:
```bash
# Change this line in the training scripts
trainer.hf_account=xxx  # Change to your Hugging Face username
```

## Knights and Knaves (K&K) Puzzles

Knights and Knaves puzzles are classic logical reasoning problems where:
- An island is inhabited by knights (who always tell the truth) and knaves (who always lie)
- The solver must determine who is a knight and who is a knave based on statements made by the inhabitants
- The puzzles require deductive reasoning and logical inference

The project uses K&K puzzles of varying complexity (from 3 to 14 people) to train and evaluate LLMs' logical reasoning capabilities.

## Training Details

- **Base Model**: Qwen2.5-7B-Instruct-1M
- **Sequence Length**: Long sequence training with 4K token prompts and up to 20K token responses
- **Training Strategy**: Two variants are available:
  - With curriculum learning (`combinedkk.sh`) - Custom implementation focus of this project
  - Without curriculum learning (`combinedkk_nocl.sh`) - For comparative evaluation
- **Hardware Requirements**: 8 GPUs per node (recommended)

## Training Scripts

The project includes two primary training scripts:

### With DUMP Curriculum Learning
```bash
conda activate dump
./main_grpo_Qwen2.5-7B-Instruct-1M_combined_logic_longseq_combinedkk.sh
```

### Without DUMP Curriculum Learning
```bash
conda activate dump
./main_grpo_Qwen2.5-7B-Instruct-1M_combined_logic_longseq_combinedkk_nocl.sh
```

**Important**: Before running these scripts, remember to modify the `trainer.hf_account` parameter in the scripts from `xxx` to your own Hugging Face username to enable model uploading.

## Dataset Generation Process (Optional)


The dataset generation process is optional, you can directly use our generated data located in ./combined_logic_dataset/generate_combined_kk. The dataset generation process consists of the following steps:

1. **Generate K&K puzzles**:
   ```bash
   cd kk
   conda env create -f environment.yml
   conda activate kk
   cd ..
   python kk/data_prep/data_gen_kk.py
   ```
   This generates various Knights and Knaves puzzles in JSONL format.

2. **Move generated files to combined_logic_dataset**:
   ```bash
   # Move all generated JSONL files to the appropriate directory
   mv data/*/clean/*.jsonl combined_logic_dataset/kk/
   ```

3. **Generate combined dataset**:
   ```bash
   # Run the dataset combiner in background
   conda activate dump
   nohup python ./combined_logic_dataset/generate_combined_kk.py --local_dir ./combined_logic_dataset/generate_combined_kk > generate_combined_kk.log 2>&1 &
   ```
   This processes the JSONL files into parquet files with carefully formatted prompts suitable for instruction-tuned models.

## Project Structure

```
├── verl/                  # Reinforcement learning framework (external dependency with modifications)
│   ├── trainer/           # RL training implementation
│   └── ...
├── kk/                    # Knights and Knaves utilities
│   ├── data_prep/         # Data preparation utilities
│   │   └── data_gen_kk.py # Main data generation script used in this project
│   └── ...                # Other utilities (not directly used)
├── combined_logic_dataset/  # Combined dataset generation
│   ├── kk/                # Location for generated KK dataset files
│   ├── generate_combined_kk/ # Output directory for processed datasets
│   └── generate_combined_kk.py # Dataset combination script
└── main_grpo_*.sh         # Training scripts
```

## Usage

1. **Generate K&K dataset (Optional)**:
   ```bash
   cd kk
   conda env create -f environment.yml
   conda activate kk
   cd ..
   python ./kk/data_prep/data_gen_kk.py
   ```

2. **Move generated files (Optional)**:
   ```bash
   mv data/train/clean/*.jsonl combined_logic_dataset/kk/
   ```

3. **Generate combined dataset (Optional)**:
   ```bash
   conda activate dump
   nohup python ./combined_logic_dataset/generate_combined_kk.py --local_dir ./combined_logic_dataset/generate_combined_kk > generate_combined_kk.log 2>&1 &
   ```

4. **Running experiments**:

    - Before running the training scripts, make sure to modify the `trainer.hf_account=xxx` parameter in the `.sh` files to your own Hugging Face username.
    - The training requires significant GPU resources (8x A100/H100 GPUs recommended).
    - Dataset generation is optional as pre-processed data is already available in `./combined_logic_dataset/generate_combined_kk/`.
    - Ensure you've logged into both Weights & Biases and Hugging Face before starting training to enable experiment tracking and model uploading.

    *Start training with DUMP curriculum learning*:

   ```bash
   conda activate dump
   ./main_grpo_Qwen2.5-7B-Instruct-1M_combined_logic_longseq_combinedkk.sh
   ```

    *Start training without DUMP curriculum learning (for comparison)*:
   ```bash
   conda activate dump
   ./main_grpo_Qwen2.5-7B-Instruct-1M_combined_logic_longseq_combinedkk_nocl.sh
   ```

## Primary Contribution: Curriculum Learning

The main contribution of this project is the implementation of curriculum learning strategies in the verl reinforcement learning framework. The curriculum learning approach enables more effective training by:

- Organizing training samples based on difficulty
- Progressively introducing more complex examples
- Using the `data_source_key` parameter to identify different data sources
- Dynamically adjusting the learning process based on model performance

## Curriculum Learning Implementation

Our implementation in `verl/utils/curriculum_learning.py` provides a robust framework for distribution-level curriculum learning in RL-based LLM training. The system dynamically adjusts sampling probabilities to focus on distributions with the highest learning potential.

### Key Components of the Implementation

#### 1. LearnabilityEstimator

The `LearnabilityEstimator` tracks performance metrics for individual data distributions:

```python
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
```

- Uses a sliding window approach to track recent model performance
- Maintains a Normal distribution to model the advantage function
- Tracks both reward rates and advantage magnitudes

#### 2. CurriculumController

The `CurriculumController` manages multiple estimators and computes optimal sampling weights:

```python
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
```

- Implements the Upper Confidence Bound (UCB) principle for balancing exploration and exploitation
- The `advantage_mean` serves as the exploitation term, prioritizing distributions with higher potential gains
- The exploration bonus favors less frequently sampled distributions
- Uses softmax with temperature to convert scores into sampling probabilities

#### 3. CurriculumSampler

The `CurriculumSampler` implements the actual sampling logic:

```python
class CurriculumSampler:
    """Sampler that implements curriculum learning based on task difficulty."""
    def __init__(self, 
                 dataset,
                 data_source_key: str = 'data_source',
                 batch_size: int = 1,
                 seed: Optional[int] = None):
```

- Integrates with PyTorch's data loading infrastructure
- Groups dataset items by their data source
- Dynamically updates sampling weights based on training progress
- Implements robust error handling to manage problematic samples

### How It Works

1. **Data Source Identification**: Training data is tagged with a `data_source_key` to identify different distributions.

2. **Performance Tracking**: During training, the system tracks:
   - Reward values achieved for each distribution
   - Advantage function magnitudes, which indicate learning potential

3. **Dynamic Weight Adjustment**: The UCB-based algorithm adjusts sampling weights to:
   - Prioritize distributions with high advantage magnitudes (high learning potential)
   - Ensure sufficient exploration of all distributions
   - Gradually shift focus as the model improves on specific distributions

4. **Robust Sampling**: The sampler handles practical implementation challenges like:
   - Filtering problematic samples
   - Gracefully managing empty distributions
   - Providing fallback mechanisms when errors occur

This implementation realizes the theoretical framework proposed in our paper, creating a principled approach to curriculum learning that adapts to the model's changing capabilities during training.

## Acknowledgement
https://github.com/volcengine/verl

https://github.com/AlphaPav/mem-kk-logic

https://github.com/Unakar/Logic-RL